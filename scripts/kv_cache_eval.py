#!/usr/bin/env python3
"""KV-cache robustness evaluation for quantized models."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv_eval import HFKVRunner, OllamaKVRunner, RunConfig, aggregate, build_prompt_set
from src.kv_eval.report import make_run_dir, write_raw_jsonl, write_summary_json, write_tables_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KV-cache robustness evaluation")
    p.add_argument("--track", choices=["ollama", "hf", "hybrid"], default="hybrid")
    p.add_argument("--models", type=str, default="gemma3:1b,qwen3:1.7b", help="Comma-separated Ollama model names")
    p.add_argument("--hf-models", type=str, default="gemma3-1b,qwen3-1.7b", help="Comma-separated model keys from configs/models.yaml")
    p.add_argument("--models-config", type=str, default="configs/models.yaml")
    p.add_argument("--attack-modes", type=str, default="none,prefill,cache_contam,cache_reuse")
    p.add_argument("--harmful-samples", type=int, default=100)
    p.add_argument("--benign-samples", type=int, default=100)
    p.add_argument("--output-root", type=str, default="artifacts/kv_cache_eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--reuse-session", action="store_true", default=True)
    p.add_argument("--no-reuse-session", dest="reuse_session", action="store_false")
    p.add_argument("--reset-between-turns", action="store_true", default=False)
    p.add_argument("--keep-alive-s", type=int, default=300)
    p.add_argument("--defense-grid", type=str, default="0.3:0.5,0.7:1.5", help="p_weak:alpha pairs")
    p.add_argument("--num-candidates", type=int, default=3)
    p.add_argument("--run-self-tests", action="store_true", help="Run lightweight internal checks before eval")
    return p.parse_args()


def parse_defense_grid(s: str, num_candidates: int) -> List[Dict[str, float]]:
    out = []
    for part in s.split(","):
        pw, al = part.strip().split(":")
        out.append({"p_weak": float(pw), "alpha": float(al), "num_candidates": int(num_candidates)})
    return out


def load_models_config(path: str) -> Dict[str, Any]:
    try:
        import yaml  # local import to keep Ollama-only runs dependency-light
    except ImportError as e:
        raise RuntimeError("PyYAML is required for HF/hybrid track. Install with `pip install pyyaml`.") from e

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_self_tests() -> None:
    prompts = build_prompt_set(4, 4, seed=7)
    assert len(prompts) == 8
    assert any(p.is_harmful for p in prompts)
    assert any(not p.is_harmful for p in prompts)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.run_self_tests:
        run_self_tests()
        print("[ok] self-tests passed")

    attack_modes = [a.strip() for a in args.attack_modes.split(",") if a.strip()]
    defense_grid = parse_defense_grid(args.defense_grid, num_candidates=args.num_candidates)

    config = RunConfig(
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        track=args.track,
        attack_modes=attack_modes,
        harmful_samples=args.harmful_samples,
        benign_samples=args.benign_samples,
        output_root=args.output_root,
        seed=args.seed,
        defense_params=defense_grid,
        decode_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        cache_params={
            "reuse_session": args.reuse_session,
            "keep_alive_s": args.keep_alive_s,
            "reset_between_turns": args.reset_between_turns,
        },
    )

    prompts = build_prompt_set(args.harmful_samples, args.benign_samples, seed=args.seed)
    run_dir = make_run_dir(args.output_root)

    all_records = []
    notes: Dict[str, str] = {}

    use_ollama = args.track in {"ollama", "hybrid"}
    use_hf = args.track in {"hf", "hybrid"}

    if use_ollama:
        ollama = OllamaKVRunner()
        available = set(ollama.list_models())

        for model in config.models:
            if model not in available:
                notes[f"ollama/{model}"] = "skipped: model not in ollama list"
                continue

            for attack_mode in attack_modes:
                baseline = ollama.run_eval(
                    model=model,
                    prompts=prompts,
                    attack_mode=attack_mode,
                    defense_name="baseline",
                    defense_on=False,
                    defense_cfg={},
                    decode_params=config.decode_params or {},
                    cache_params=config.cache_params or {},
                    seed=config.seed,
                )
                all_records.extend(baseline)

                for d in defense_grid:
                    name = f"deeprefusal_p{d['p_weak']}_a{d['alpha']}"
                    rows = ollama.run_eval(
                        model=model,
                        prompts=prompts,
                        attack_mode=attack_mode,
                        defense_name=name,
                        defense_on=True,
                        defense_cfg=d,
                        decode_params=config.decode_params or {},
                        cache_params=config.cache_params or {},
                        seed=config.seed,
                    )
                    all_records.extend(rows)

    if use_hf:
        models_cfg = load_models_config(args.models_config)
        hf = HFKVRunner(models_cfg)
        hf_models = [m.strip() for m in args.hf_models.split(",") if m.strip()]

        for model in hf_models:
            for attack_mode in attack_modes:
                baseline, err = hf.run_eval(
                    model_name=model,
                    prompts=prompts,
                    attack_mode=attack_mode,
                    defense_name="baseline",
                    defense_on=False,
                    defense_cfg={},
                    decode_params=config.decode_params or {},
                    cache_params=config.cache_params or {},
                    seed=config.seed,
                )
                if err:
                    notes[f"hf/{model}"] = f"skipped: {err}"
                    break
                all_records.extend(baseline)

                for d in defense_grid:
                    name = f"deeprefusal_p{d['p_weak']}_a{d['alpha']}"
                    rows, err = hf.run_eval(
                        model_name=model,
                        prompts=prompts,
                        attack_mode=attack_mode,
                        defense_name=name,
                        defense_on=True,
                        defense_cfg=d,
                        decode_params=config.decode_params or {},
                        cache_params=config.cache_params or {},
                        seed=config.seed,
                    )
                    if err:
                        notes[f"hf/{model}"] = f"skipped during defense run: {err}"
                        break
                    all_records.extend(rows)

    # aggregate per (model,track,attack_mode,defense_name)
    buckets: Dict[tuple, List[Any]] = defaultdict(list)
    for rec in all_records:
        buckets[(rec.model, rec.track, rec.attack_mode, rec.defense_name)].append(rec)

    summaries = [aggregate(rows, seed=args.seed) for rows in buckets.values()]
    summaries.sort(key=lambda x: (x.track, x.model, x.attack_mode, x.defense_name))

    write_raw_jsonl(run_dir / "raw.jsonl", all_records)
    write_summary_json(run_dir / "summary.json", config, summaries, notes)
    write_tables_csv(run_dir / "tables.csv", summaries)

    # convenience pointer
    latest = Path(args.output_root) / "latest.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir)}, f, indent=2)

    print(f"[done] run_dir={run_dir}")
    print(f"[done] records={len(all_records)} summaries={len(summaries)}")
    if notes:
        print("[notes]")
        for k, v in notes.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
