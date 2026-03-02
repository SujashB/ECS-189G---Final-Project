#!/usr/bin/env python3
"""
Unified experiment runner for the DeepRefusal evaluation framework.

Supports:
  --evaluate_paper_protocol : 6 attack types + over-refusal + MMLU/GSM8k
  --evaluate_kv_extension   : KV-cache attacks (cache_reuse, cache_contam, prefill)
  --p_sweep                 : PAA sensitivity sweep over p values

Example:
  python run_full_evaluation.py \
      --model_name gemma3-1b \
      --evaluate_paper_protocol \
      --output_dir artifacts/paper_eval \
      --mmlu_subset 100 --gsm8k_subset 100
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.types import EvalConfig
from evaluation.attack_runner import run_all_paper_attacks
from evaluation.asr import compute_asr, compute_asr_with_ci, compute_delta
from evaluation.over_refusal import compute_over_refusal, load_benign_dataset
from evaluation.capability import evaluate_mmlu, evaluate_gsm8k
from evaluation.report import generate_all_tables
from src.model_utils import load_model_and_tokenizer, load_config, cleanup_memory
from src.refusal_detector import RefusalDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepRefusal full evaluation runner")

    # Model
    p.add_argument("--model_name", type=str, required=True,
                   help="Model name as defined in configs/models.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to DeepRefusal fine-tuned checkpoint")
    p.add_argument("--baseline_checkpoint", type=str, default=None,
                   help="Path to baseline (non-DeepRefusal) checkpoint")
    p.add_argument("--models_config", type=str, default="configs/models.yaml",
                   help="Path to models YAML config")

    # Evaluation modes
    p.add_argument("--evaluate_paper_protocol", action="store_true",
                   help="Run the full paper evaluation protocol")
    p.add_argument("--evaluate_kv_extension", action="store_true",
                   help="Run KV-cache extension evaluation")
    p.add_argument("--p_sweep", type=float, nargs="+", default=None,
                   help="PAA probability values for sensitivity sweep")

    # Output
    p.add_argument("--output_dir", type=str, default="artifacts/eval_output",
                   help="Directory for output tables and metadata")

    # Generation parameters
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--deterministic", action="store_true",
                   help="Force do_sample=False, temperature=1.0")
    p.add_argument("--batch_size", type=int, default=1)

    # Subset sizes for fast runs
    p.add_argument("--mmlu_subset", type=int, default=None,
                   help="Number of MMLU examples to evaluate (None = full)")
    p.add_argument("--gsm8k_subset", type=int, default=None,
                   help="Number of GSM8k examples to evaluate (None = full)")

    return p.parse_args()


def _save_metadata(args: argparse.Namespace, output_dir: Path) -> None:
    """Save run metadata to JSON."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "baseline_checkpoint": args.baseline_checkpoint,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "deterministic": args.deterministic,
        "p_sweep": args.p_sweep,
        "mmlu_subset": args.mmlu_subset,
        "gsm8k_subset": args.gsm8k_subset,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)


def _load_model(args: argparse.Namespace, checkpoint: str = None):
    """Load model and tokenizer, optionally from a checkpoint."""
    models_config = load_config(args.models_config)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, models_config, load_in_4bit=True,
    )
    if checkpoint:
        from peft import PeftModel
        logger.info("Loading checkpoint from %s", checkpoint)
        model = PeftModel.from_pretrained(model, checkpoint)
    return model, tokenizer


def run_paper_protocol(args: argparse.Namespace, output_dir: Path) -> None:
    """Run the full paper evaluation protocol."""
    eval_cfg = EvalConfig(
        seed=args.seed, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens, deterministic=args.deterministic,
    )
    gen_params = eval_cfg.get_gen_params()
    detector = RefusalDetector()

    # --- Core ASR ---
    logger.info("=== Phase 1: Core ASR evaluation ===")
    core_asr_rows = []

    # Baseline model
    if args.baseline_checkpoint:
        logger.info("Loading baseline model...")
        model, tokenizer = _load_model(args, checkpoint=args.baseline_checkpoint)
        baseline_results = run_all_paper_attacks(
            model, tokenizer, gen_params=gen_params, detector=detector,
            batch_size=args.batch_size,
        )
        cleanup_memory()
    else:
        logger.info("Loading base model (no baseline checkpoint)...")
        model, tokenizer = _load_model(args)
        baseline_results = run_all_paper_attacks(
            model, tokenizer, gen_params=gen_params, detector=detector,
            batch_size=args.batch_size,
        )
        cleanup_memory()

    # DeepRefusal model
    if args.checkpoint:
        logger.info("Loading DeepRefusal model...")
        dr_model, dr_tokenizer = _load_model(args, checkpoint=args.checkpoint)
        dr_results = run_all_paper_attacks(
            dr_model, dr_tokenizer, gen_params=gen_params, detector=detector,
            batch_size=args.batch_size,
        )
        cleanup_memory()
    else:
        # If no checkpoint, just use the same baseline results
        logger.warning("No --checkpoint provided; DeepRefusal ASR = baseline ASR.")
        dr_results = baseline_results

    # Build table rows
    baseline_map = {r.attack_name: r for r in baseline_results}
    dr_map = {r.attack_name: r for r in dr_results}
    all_attacks = sorted(set(baseline_map) | set(dr_map))
    for atk in all_attacks:
        b_asr = baseline_map[atk].asr if atk in baseline_map else 0.0
        d_asr = dr_map[atk].asr if atk in dr_map else 0.0
        core_asr_rows.append({
            "model": args.model_name,
            "attack": atk,
            "baseline_asr": f"{b_asr:.4f}",
            "deeprefusal_asr": f"{d_asr:.4f}",
            "delta_asr": f"{compute_delta(b_asr, d_asr):.4f}",
        })

    # --- Over-refusal ---
    logger.info("=== Phase 2: Over-refusal evaluation ===")
    benign_prompts = load_benign_dataset(project_root=_ROOT)
    overrefusal_rows = []

    if benign_prompts:
        from src.model_utils import generate_text

        # Baseline over-refusal
        if args.baseline_checkpoint or not args.checkpoint:
            b_model, b_tok = (model, tokenizer) if not args.checkpoint else _load_model(
                args, checkpoint=args.baseline_checkpoint
            )
        else:
            b_model, b_tok = _load_model(args)

        b_responses = generate_text(b_model, b_tok, benign_prompts, **gen_params)
        b_or = compute_over_refusal(benign_prompts, b_responses)
        cleanup_memory()

        # DeepRefusal over-refusal
        if args.checkpoint:
            dr_model2, dr_tok2 = _load_model(args, checkpoint=args.checkpoint)
            dr_responses = generate_text(dr_model2, dr_tok2, benign_prompts, **gen_params)
            dr_or = compute_over_refusal(benign_prompts, dr_responses)
            cleanup_memory()
        else:
            dr_or = b_or

        overrefusal_rows.append({
            "model": args.model_name,
            "baseline_overrefusal": f"{b_or.over_refusal_rate:.4f}",
            "deeprefusal_overrefusal": f"{dr_or.over_refusal_rate:.4f}",
            "delta": f"{dr_or.over_refusal_rate - b_or.over_refusal_rate:.4f}",
        })
    else:
        logger.warning("Skipping over-refusal evaluation (no benign prompts).")

    # --- Capability ---
    logger.info("=== Phase 3: Capability benchmarks ===")
    capability_rows = []

    # Use base model for capability (no checkpoint needed for MMLU/GSM8k)
    cap_model, cap_tok = _load_model(args)

    b_mmlu = evaluate_mmlu(
        cap_model, cap_tok, subset_size=args.mmlu_subset, seed=args.seed,
    )
    b_gsm = evaluate_gsm8k(
        cap_model, cap_tok, subset_size=args.gsm8k_subset, seed=args.seed,
    )

    if args.checkpoint:
        cleanup_memory()
        dr_cap_model, dr_cap_tok = _load_model(args, checkpoint=args.checkpoint)
        dr_mmlu = evaluate_mmlu(
            dr_cap_model, dr_cap_tok, subset_size=args.mmlu_subset, seed=args.seed,
        )
        dr_gsm = evaluate_gsm8k(
            dr_cap_model, dr_cap_tok, subset_size=args.gsm8k_subset, seed=args.seed,
        )
        cleanup_memory()
    else:
        dr_mmlu, dr_gsm = b_mmlu, b_gsm

    capability_rows.append({
        "model": args.model_name,
        "baseline_mmlu": f"{b_mmlu.accuracy:.4f}",
        "deeprefusal_mmlu": f"{dr_mmlu.accuracy:.4f}",
        "baseline_gsm8k": f"{b_gsm.accuracy:.4f}",
        "deeprefusal_gsm8k": f"{dr_gsm.accuracy:.4f}",
    })

    # --- Generate tables ---
    generate_all_tables(
        core_asr_results=core_asr_rows,
        overrefusal_results=overrefusal_rows,
        capability_results=capability_rows,
        output_dir=str(output_dir),
    )


def run_kv_extension(args: argparse.Namespace, output_dir: Path) -> None:
    """Run KV-cache extension evaluation using existing kv_eval module."""
    from src.kv_eval.hf_runner import HFKVRunner
    from src.kv_eval.metrics import aggregate
    from evaluation.report import generate_kv_extension_table

    eval_cfg = EvalConfig(
        seed=args.seed, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens, deterministic=args.deterministic,
    )

    kv_rows = []
    for attack_mode in ["cache_reuse", "cache_contam", "prefill"]:
        logger.info("Running KV extension: %s", attack_mode)
        runner = HFKVRunner()
        models_config = load_config(args.models_config)
        model_config = models_config["models"][args.model_name]
        load_result = runner.load(model_config["path"])

        records = runner.run_eval(
            attack_mode=attack_mode,
            n_samples=20,
        )
        metrics = aggregate(records)
        kv_rows.append({
            "model": args.model_name,
            "attack_mode": attack_mode,
            "asr": f"{metrics.asr:.4f}",
            "refusal_rate": f"{metrics.refusal_rate:.4f}",
            "utility_rate": f"{metrics.utility_rate:.4f}",
        })
        cleanup_memory()

    generate_kv_extension_table(kv_rows, str(output_dir))


def run_p_sweep(args: argparse.Namespace, output_dir: Path) -> None:
    """Run PAA sensitivity sweep over p values."""
    from evaluation.report import generate_p_sweep_table

    if not args.p_sweep:
        return

    eval_cfg = EvalConfig(
        seed=args.seed, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens, deterministic=args.deterministic,
    )
    gen_params = eval_cfg.get_gen_params()
    detector = RefusalDetector()

    sweep_rows = []
    for p_val in args.p_sweep:
        logger.info("=== P-sweep: p=%.3f ===", p_val)

        # Load model with checkpoint if available
        model, tokenizer = _load_model(args, checkpoint=args.checkpoint)

        results = run_all_paper_attacks(
            model, tokenizer, gen_params=gen_params, detector=detector,
            batch_size=args.batch_size,
        )

        for r in results:
            sweep_rows.append({
                "model": args.model_name,
                "p_value": f"{p_val:.3f}",
                "attack": r.attack_name,
                "asr": f"{r.asr:.4f}",
                "total": r.total,
                "successes": r.successes,
            })

        cleanup_memory()

    generate_p_sweep_table(sweep_rows, str(output_dir))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_metadata(args, output_dir)

    if not (args.evaluate_paper_protocol or args.evaluate_kv_extension or args.p_sweep):
        logger.warning(
            "No evaluation mode selected. Use --evaluate_paper_protocol, "
            "--evaluate_kv_extension, or --p_sweep."
        )
        return

    if args.evaluate_paper_protocol:
        run_paper_protocol(args, output_dir)

    if args.evaluate_kv_extension:
        run_kv_extension(args, output_dir)

    if args.p_sweep:
        run_p_sweep(args, output_dir)

    logger.info("All evaluations complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
