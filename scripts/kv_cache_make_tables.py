#!/usr/bin/env python3
"""Generate paper-ready CSV tables from kv_cache_eval summary.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create paper-ready tables from KV eval summary")
    p.add_argument("--run-dir", type=str, default="", help="Run dir with summary.json; if empty use artifacts/kv_cache_eval/latest.json")
    return p.parse_args()


def resolve_run_dir(arg: str) -> Path:
    if arg:
        return Path(arg)
    latest = Path("artifacts/kv_cache_eval/latest.json")
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return Path(payload["run_dir"])


def load_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    p = run_dir / "summary.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("summaries", [])


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_core_asr_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append(
            {
                "model": r["model"],
                "track": r["track"],
                "attack_mode": r["attack_mode"],
                "defense_name": r["defense_name"],
                "asr": r["asr"],
                "asr_ci_low": r["asr_ci_low"],
                "asr_ci_high": r["asr_ci_high"],
                "over_refusal_rate": r["over_refusal_rate"],
                "utility_rate": r["utility_rate"],
            }
        )
    return out


def build_pareto_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append(
            {
                "model": r["model"],
                "track": r["track"],
                "attack_mode": r["attack_mode"],
                "defense_name": r["defense_name"],
                "asr": r["asr"],
                "over_refusal_rate": r["over_refusal_rate"],
                "p50_latency_ms": r["p50_latency_ms"],
                "p95_latency_ms": r["p95_latency_ms"],
                "toks_per_sec": r["toks_per_sec"],
            }
        )
    return out


def build_baseline_delta_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idx = {(r["model"], r["track"], r["attack_mode"], r["defense_name"]): r for r in rows}
    out: List[Dict[str, Any]] = []

    for r in rows:
        if r["defense_name"] == "baseline":
            continue
        key_base = (r["model"], r["track"], r["attack_mode"], "baseline")
        base = idx.get(key_base)
        if not base:
            continue
        out.append(
            {
                "model": r["model"],
                "track": r["track"],
                "attack_mode": r["attack_mode"],
                "defense_name": r["defense_name"],
                "asr_baseline": base["asr"],
                "asr_defense": r["asr"],
                "asr_relative_reduction": ((base["asr"] - r["asr"]) / base["asr"]) if base["asr"] > 0 else 0.0,
                "over_refusal_baseline": base["over_refusal_rate"],
                "over_refusal_defense": r["over_refusal_rate"],
                "p50_latency_baseline": base["p50_latency_ms"],
                "p50_latency_defense": r["p50_latency_ms"],
                "p50_latency_overhead_frac": ((r["p50_latency_ms"] - base["p50_latency_ms"]) / base["p50_latency_ms"]) if base["p50_latency_ms"] > 0 else 0.0,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    rows = load_summaries(run_dir)

    core = build_core_asr_table(rows)
    pareto = build_pareto_table(rows)
    delta = build_baseline_delta_table(rows)

    write_csv(run_dir / "table_core_asr.csv", core, list(core[0].keys()) if core else [])
    write_csv(run_dir / "figure_pareto.csv", pareto, list(pareto[0].keys()) if pareto else [])
    write_csv(run_dir / "table_baseline_delta.csv", delta, list(delta[0].keys()) if delta else [])

    print(f"[done] generated tables in {run_dir}")
    print("- table_core_asr.csv")
    print("- figure_pareto.csv")
    print("- table_baseline_delta.csv")


if __name__ == "__main__":
    main()
