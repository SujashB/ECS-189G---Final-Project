#!/usr/bin/env python3
"""
Run full DeepRefusal evaluation across all 3 models, generate CSV tables,
plots, radar charts, and a RESULTS.md report.
"""

import csv
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model_utils import load_model_and_tokenizer, load_config, generate_text, cleanup_memory
from src.refusal_detector import RefusalDetector
from src.data_utils import load_jsonl
from evaluation.types import EvalConfig, AttackResult, OverRefusalResult
from evaluation.attack_runner import run_attack
from evaluation.asr import compute_asr_with_ci
from evaluation.over_refusal import classify_response, ResponseClass, compute_over_refusal
from evaluation.capability import evaluate_mmlu, evaluate_gsm8k
from evaluation.report import generate_all_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT / "artifacts" / "paper_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gemma3-1b", "qwen3-1.7b", "deepseek-r1-1.5b"]
CHECKPOINT_MAP = {
    "gemma3-1b": str(ROOT / "runs" / "gemma3-1b" / "deeprefusal"),
    "qwen3-1.7b": str(ROOT / "runs" / "qwen3-1.7b" / "deeprefusal"),
    "deepseek-r1-1.5b": str(ROOT / "runs" / "deepseek-r1-1.5b" / "deeprefusal"),
}
BASELINE_CHECKPOINT_MAP = {
    "gemma3-1b": str(ROOT / "runs" / "gemma3-1b" / "standard_refusal_training" / "final"),
}

MMLU_SUBSET = 100
GSM8K_SUBSET = 100
SEED = 42
MAX_ATTACK_SAMPLES = 50  # Cap attack prompts for speed
MAX_OVERREFUSAL_SAMPLES = 50
MAX_UTILITY_SAMPLES = 30

# Attack datasets (use harmbench + advbench as the attack prompts)
ATTACK_DATASETS = {
    "harmbench": str(ROOT / "data" / "processed" / "harmbench_standard_eval.jsonl"),
    "advbench": str(ROOT / "data" / "processed" / "advbench_harmful_eval.jsonl"),
}

# Over-refusal dataset
XSTEST_PATH = str(ROOT / "data" / "processed" / "xstest_eval.jsonl")
ULTRACHAT_PATH = str(ROOT / "data" / "processed" / "ultrachat_benign_eval.jsonl")


def load_model(model_name, checkpoint=None):
    """Load a model with optional PEFT checkpoint."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    models_config = load_config(str(ROOT / "configs" / "models.yaml"))
    model_config = models_config["models"][model_name]
    model_path = model_config["path"]
    dtype = getattr(torch, model_config.get("dtype", "bfloat16"))

    # HF hub tokenizer fallback map (extends model_utils for qwen3/deepseek)
    _tokenizer_hub = {
        "gemma3-1b": "google/gemma-3-1b-it",
        "qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    }

    # Load tokenizer (local → HF hub fallback)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        logger.info("Loaded tokenizer from local: %s", model_path)
    except Exception as e:
        logger.warning("Local tokenizer failed: %s", e)
        hub_path = _tokenizer_hub.get(model_name)
        if hub_path:
            tokenizer = AutoTokenizer.from_pretrained(hub_path, trust_remote_code=True)
            logger.info("Loaded tokenizer from HF: %s", hub_path)

    if tokenizer is None:
        raise RuntimeError(f"Cannot load tokenizer for {model_name}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config,
        device_map="auto", torch_dtype=dtype, trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    if checkpoint and Path(checkpoint).exists():
        from peft import PeftModel
        logger.info("Loading PEFT adapter from %s", checkpoint)
        model = PeftModel.from_pretrained(model, checkpoint)
        model.eval()
    return model, tokenizer


def free_model(*args):
    """Delete model objects and free GPU memory."""
    for obj in args:
        del obj
    gc.collect()
    cleanup_memory()


# ── Evaluation Functions ─────────────────────────────────────────────────────

def eval_asr_on_dataset(model, tokenizer, dataset_path, gen_params, detector):
    """Evaluate ASR on a harmful dataset."""
    records = load_jsonl(dataset_path)
    prompts = [r["prompt"] for r in records][:MAX_ATTACK_SAMPLES]
    if not prompts:
        return 0.0, 0.0, 0.0, 0, []

    responses = generate_text(model, tokenizer, prompts, **gen_params)
    # harmful_label_fn: True if model did NOT refuse (attack success)
    harmful_fn = lambda r: not detector.is_refusal(r)
    asr, ci_lo, ci_hi = compute_asr_with_ci(responses, harmful_fn, n_boot=1000, seed=SEED)
    return asr, ci_lo, ci_hi, len(prompts), responses


def eval_over_refusal(model, tokenizer, gen_params, detector):
    """Run over-refusal evaluation on XSTest safe prompts."""
    records = load_jsonl(XSTEST_PATH)
    safe_records = [r for r in records if r.get("label") == "safe"]
    prompts = [r["prompt"] for r in safe_records][:MAX_OVERREFUSAL_SAMPLES]
    if not prompts:
        return None, []

    responses = generate_text(model, tokenizer, prompts, **gen_params)
    result = compute_over_refusal(prompts, responses)
    return result, responses


def eval_utility(model, tokenizer, gen_params, detector):
    """Evaluate utility on benign UltraChat prompts."""
    records = load_jsonl(ULTRACHAT_PATH)[:MAX_UTILITY_SAMPLES]
    prompts = [r["prompt"] for r in records]
    if not prompts:
        return 0.0

    responses = generate_text(model, tokenizer, prompts, **gen_params)
    refusal_rate = detector.get_refusal_rate(responses)
    return 1.0 - refusal_rate


# ── Main Evaluation Loop ─────────────────────────────────────────────────────

def run_all():
    eval_cfg = EvalConfig(seed=SEED, deterministic=True, max_new_tokens=128)
    gen_params = eval_cfg.get_gen_params()
    detector = RefusalDetector()

    all_results = {}

    for model_name in MODELS:
        logger.info("=" * 60)
        logger.info("EVALUATING: %s", model_name)
        logger.info("=" * 60)

        results = {"model": model_name}

        # ─── Baseline model ────────────────────────────────────────
        logger.info("--- Baseline (base model) ---")
        model, tokenizer = load_model(model_name)

        for ds_name, ds_path in ATTACK_DATASETS.items():
            asr, ci_lo, ci_hi, n, _ = eval_asr_on_dataset(
                model, tokenizer, ds_path, gen_params, detector
            )
            results[f"baseline_{ds_name}_asr"] = asr
            results[f"baseline_{ds_name}_ci"] = (ci_lo, ci_hi)
            results[f"baseline_{ds_name}_n"] = n
            logger.info("  %s baseline ASR: %.4f [%.4f, %.4f] (n=%d)",
                        ds_name, asr, ci_lo, ci_hi, n)

        # Over-refusal on baseline
        or_result, _ = eval_over_refusal(model, tokenizer, gen_params, detector)
        results["baseline_overrefusal"] = or_result.over_refusal_rate if or_result else 0.0
        results["baseline_overrefusal_detail"] = {
            "direct_answer": or_result.direct_answer if or_result else 0,
            "indirect_answer": or_result.indirect_answer if or_result else 0,
            "direct_refusal": or_result.direct_refusal if or_result else 0,
            "indirect_refusal": or_result.indirect_refusal if or_result else 0,
        }

        # Utility on baseline
        results["baseline_utility"] = eval_utility(model, tokenizer, gen_params, detector)

        # Capability benchmarks on baseline
        logger.info("  Running MMLU (subset=%d)...", MMLU_SUBSET)
        mmlu = evaluate_mmlu(model, tokenizer, subset_size=MMLU_SUBSET, seed=SEED)
        results["baseline_mmlu"] = mmlu.accuracy
        results["baseline_mmlu_n"] = mmlu.total

        logger.info("  Running GSM8k (subset=%d)...", GSM8K_SUBSET)
        gsm = evaluate_gsm8k(model, tokenizer, subset_size=GSM8K_SUBSET, seed=SEED)
        results["baseline_gsm8k"] = gsm.accuracy
        results["baseline_gsm8k_n"] = gsm.total

        free_model(model, tokenizer)

        # ─── DeepRefusal model ─────────────────────────────────────
        checkpoint = CHECKPOINT_MAP.get(model_name)
        if checkpoint and Path(checkpoint).exists():
            logger.info("--- DeepRefusal checkpoint: %s ---", checkpoint)
            model, tokenizer = load_model(model_name, checkpoint=checkpoint)

            for ds_name, ds_path in ATTACK_DATASETS.items():
                asr, ci_lo, ci_hi, n, _ = eval_asr_on_dataset(
                    model, tokenizer, ds_path, gen_params, detector
                )
                results[f"deeprefusal_{ds_name}_asr"] = asr
                results[f"deeprefusal_{ds_name}_ci"] = (ci_lo, ci_hi)
                logger.info("  %s DeepRefusal ASR: %.4f [%.4f, %.4f]",
                            ds_name, asr, ci_lo, ci_hi)

            or_result, _ = eval_over_refusal(model, tokenizer, gen_params, detector)
            results["deeprefusal_overrefusal"] = or_result.over_refusal_rate if or_result else 0.0
            results["deeprefusal_overrefusal_detail"] = {
                "direct_answer": or_result.direct_answer if or_result else 0,
                "indirect_answer": or_result.indirect_answer if or_result else 0,
                "direct_refusal": or_result.direct_refusal if or_result else 0,
                "indirect_refusal": or_result.indirect_refusal if or_result else 0,
            }

            results["deeprefusal_utility"] = eval_utility(model, tokenizer, gen_params, detector)

            logger.info("  Running MMLU on DeepRefusal...")
            mmlu_dr = evaluate_mmlu(model, tokenizer, subset_size=MMLU_SUBSET, seed=SEED)
            results["deeprefusal_mmlu"] = mmlu_dr.accuracy

            logger.info("  Running GSM8k on DeepRefusal...")
            gsm_dr = evaluate_gsm8k(model, tokenizer, subset_size=GSM8K_SUBSET, seed=SEED)
            results["deeprefusal_gsm8k"] = gsm_dr.accuracy

            free_model(model, tokenizer)
        else:
            logger.warning("No DeepRefusal checkpoint for %s", model_name)
            for ds_name in ATTACK_DATASETS:
                results[f"deeprefusal_{ds_name}_asr"] = results[f"baseline_{ds_name}_asr"]
                results[f"deeprefusal_{ds_name}_ci"] = results[f"baseline_{ds_name}_ci"]
            results["deeprefusal_overrefusal"] = results["baseline_overrefusal"]
            results["deeprefusal_overrefusal_detail"] = results["baseline_overrefusal_detail"]
            results["deeprefusal_utility"] = results["baseline_utility"]
            results["deeprefusal_mmlu"] = results["baseline_mmlu"]
            results["deeprefusal_gsm8k"] = results["baseline_gsm8k"]

        all_results[model_name] = results

        # Incremental save after each model
        def _serialize(obj):
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        inc_path = OUTPUT_DIR / "all_results_partial.json"
        with open(inc_path, "w") as f:
            json.dump(all_results, f, indent=2, default=_serialize)
        logger.info("Saved incremental results to %s", inc_path)

    return all_results


# ── CSV Table Generation ────────────────────────────────────────────────────

def write_tables(all_results):
    """Write all CSV tables."""
    # Table 1: Core ASR
    core_rows = []
    for m, r in all_results.items():
        for ds in ATTACK_DATASETS:
            core_rows.append({
                "model": m,
                "attack": ds,
                "baseline_asr": f"{r[f'baseline_{ds}_asr']:.4f}",
                "deeprefusal_asr": f"{r[f'deeprefusal_{ds}_asr']:.4f}",
                "delta_asr": f"{r[f'deeprefusal_{ds}_asr'] - r[f'baseline_{ds}_asr']:.4f}",
            })

    # Table 2: Over-refusal
    or_rows = []
    for m, r in all_results.items():
        or_rows.append({
            "model": m,
            "baseline_overrefusal": f"{r['baseline_overrefusal']:.4f}",
            "deeprefusal_overrefusal": f"{r['deeprefusal_overrefusal']:.4f}",
            "delta": f"{r['deeprefusal_overrefusal'] - r['baseline_overrefusal']:.4f}",
        })

    # Table 3: Capability
    cap_rows = []
    for m, r in all_results.items():
        cap_rows.append({
            "model": m,
            "baseline_mmlu": f"{r['baseline_mmlu']:.4f}",
            "deeprefusal_mmlu": f"{r['deeprefusal_mmlu']:.4f}",
            "baseline_gsm8k": f"{r['baseline_gsm8k']:.4f}",
            "deeprefusal_gsm8k": f"{r['deeprefusal_gsm8k']:.4f}",
        })

    paths = generate_all_tables(
        core_asr_results=core_rows,
        overrefusal_results=or_rows,
        capability_results=cap_rows,
        output_dir=str(OUTPUT_DIR),
    )
    return paths


# ── Plotting ─────────────────────────────────────────────────────────────────

def hex_colors():
    return {
        "gemma3-1b": "#4285F4",
        "qwen3-1.7b": "#EA4335",
        "deepseek-r1-1.5b": "#34A853",
    }


def plot_asr_comparison(all_results):
    """Bar chart: baseline vs DeepRefusal ASR per model per attack."""
    colors = hex_colors()
    datasets = list(ATTACK_DATASETS.keys())
    models = list(all_results.keys())
    n_models = len(models)
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    bar_width = 0.3
    for ax, ds in zip(axes, datasets):
        x = np.arange(n_models)
        baseline_vals = [all_results[m][f"baseline_{ds}_asr"] for m in models]
        dr_vals = [all_results[m][f"deeprefusal_{ds}_asr"] for m in models]

        bars1 = ax.bar(x - bar_width / 2, baseline_vals, bar_width,
                       label="Baseline", color="#BBDEFB", edgecolor="#1565C0", linewidth=0.8)
        bars2 = ax.bar(x + bar_width / 2, dr_vals, bar_width,
                       label="DeepRefusal", color="#C8E6C9", edgecolor="#2E7D32", linewidth=0.8)

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=9)
        ax.set_title(f"ASR on {ds.upper()}", fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Attack Success Rate")
    fig.suptitle("Attack Success Rate: Baseline vs DeepRefusal", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = str(OUTPUT_DIR / "plot_asr_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_overrefusal(all_results):
    """Stacked bar chart for 4-class over-refusal."""
    models = list(all_results.keys())
    categories = ["direct_answer", "indirect_answer", "direct_refusal", "indirect_refusal"]
    cat_labels = ["Direct Answer", "Indirect Answer", "Direct Refusal", "Indirect Refusal"]
    cat_colors = ["#4CAF50", "#8BC34A", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, variant, title in [
        (axes[0], "baseline", "Baseline"),
        (axes[1], "deeprefusal", "DeepRefusal"),
    ]:
        bottoms = np.zeros(len(models))
        for cat, label, color in zip(categories, cat_labels, cat_colors):
            vals = []
            for m in models:
                detail = all_results[m].get(f"{variant}_overrefusal_detail", {})
                vals.append(detail.get(cat, 0))
            vals = np.array(vals, dtype=float)
            total = np.array([
                sum(all_results[m].get(f"{variant}_overrefusal_detail", {}).values())
                for m in models
            ], dtype=float)
            total = np.where(total == 0, 1, total)
            fracs = vals / total
            ax.bar(models, fracs, bottom=bottoms, label=label, color=color, edgecolor="white", linewidth=0.5)
            bottoms += fracs

        ax.set_title(f"{title} Over-Refusal Breakdown", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=9)

    fig.suptitle("4-Class Response Distribution on Benign Prompts", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = str(OUTPUT_DIR / "plot_overrefusal.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_capability(all_results):
    """Grouped bar chart for MMLU and GSM8k."""
    models = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bar_width = 0.3
    x = np.arange(len(models))

    for ax, bench, title in [
        (axes[0], "mmlu", "MMLU Accuracy"),
        (axes[1], "gsm8k", "GSM8k Accuracy"),
    ]:
        baseline = [all_results[m][f"baseline_{bench}"] for m in models]
        dr = [all_results[m][f"deeprefusal_{bench}"] for m in models]

        bars1 = ax.bar(x - bar_width / 2, baseline, bar_width,
                       label="Baseline", color="#BBDEFB", edgecolor="#1565C0", linewidth=0.8)
        bars2 = ax.bar(x + bar_width / 2, dr, bar_width,
                       label="DeepRefusal", color="#C8E6C9", edgecolor="#2E7D32", linewidth=0.8)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(max(baseline), max(dr)) * 1.3 + 0.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Capability Benchmarks: Baseline vs DeepRefusal", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = str(OUTPUT_DIR / "plot_capability.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_radar(all_results):
    """Radar plot comparing models across all dimensions."""
    models = list(all_results.keys())
    colors_map = hex_colors()

    # Axes: lower ASR is better → use (1 - ASR) for safety
    # higher utility is better, higher MMLU/GSM8k is better, lower over-refusal is better
    categories = [
        "HarmBench\nSafety",
        "AdvBench\nSafety",
        "Utility",
        "MMLU",
        "GSM8k",
        "Low\nOver-Refusal",
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

    for ax, variant, title in [
        (axes[0], "baseline", "Baseline"),
        (axes[1], "deeprefusal", "DeepRefusal"),
    ]:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)

        for model_name in models:
            r = all_results[model_name]
            values = [
                1.0 - r[f"{variant}_harmbench_asr"],   # safety = 1 - ASR
                1.0 - r[f"{variant}_advbench_asr"],
                r[f"{variant}_utility"],
                r[f"{variant}_mmlu"],
                r[f"{variant}_gsm8k"],
                1.0 - r[f"{variant}_overrefusal"],
            ]
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=1.5, label=model_name,
                    color=colors_map[model_name], markersize=4)
            ax.fill(angles, values, alpha=0.1, color=colors_map[model_name])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontweight="bold", pad=20)
        ax.legend(loc="lower right", fontsize=7, bbox_to_anchor=(1.3, -0.1))

    fig.suptitle("Model Radar: Safety, Utility & Capability", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = str(OUTPUT_DIR / "plot_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_delta_heatmap(all_results):
    """Heatmap of DeepRefusal delta (improvement) across models and metrics."""
    models = list(all_results.keys())
    metrics = [
        ("HarmBench ASR", "harmbench_asr", -1),   # negative delta is better
        ("AdvBench ASR", "advbench_asr", -1),
        ("Over-Refusal", "overrefusal", -1),
        ("Utility", "utility", 1),                  # positive delta is better
        ("MMLU", "mmlu", 1),
        ("GSM8k", "gsm8k", 1),
    ]

    data = np.zeros((len(models), len(metrics)))
    for i, m in enumerate(models):
        for j, (label, key, direction) in enumerate(metrics):
            delta = all_results[m][f"deeprefusal_{key}"] - all_results[m][f"baseline_{key}"]
            data[i, j] = delta * direction  # positive = improvement

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(metrics)):
            raw_delta = all_results[models[i]][f"deeprefusal_{metrics[j][1]}"] - \
                        all_results[models[i]][f"baseline_{metrics[j][1]}"]
            color = "white" if abs(data[i, j]) > 0.3 else "black"
            ax.text(j, i, f"{raw_delta:+.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Improvement (green=better)")
    ax.set_title("DeepRefusal Impact: Delta Heatmap", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = str(OUTPUT_DIR / "plot_delta_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── Markdown Report ──────────────────────────────────────────────────────────

def write_results_md(all_results, plot_paths):
    """Write RESULTS.md with tables and embedded plots."""
    models = list(all_results.keys())
    lines = []
    lines.append("# DeepRefusal Evaluation Results")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> Evaluation seed: {SEED} | MMLU subset: {MMLU_SUBSET} | GSM8k subset: {GSM8K_SUBSET}")
    lines.append(f"> Decoding: deterministic (do_sample=False, temperature=1.0)")
    lines.append("")

    # ── Table 1: Core ASR ──
    lines.append("## 1. Attack Success Rate (ASR)")
    lines.append("")
    lines.append("Lower ASR = stronger safety. Delta shows DeepRefusal improvement (negative = safer).")
    lines.append("")
    lines.append("| Model | Attack | Baseline ASR | DeepRefusal ASR | Delta |")
    lines.append("|-------|--------|-------------|-----------------|-------|")
    for m in models:
        r = all_results[m]
        for ds in ATTACK_DATASETS:
            b = r[f"baseline_{ds}_asr"]
            d = r[f"deeprefusal_{ds}_asr"]
            delta = d - b
            lines.append(f"| {m} | {ds} | {b:.4f} | {d:.4f} | {delta:+.4f} |")
    lines.append("")

    # Embed ASR plot
    lines.append("### ASR Comparison Chart")
    lines.append("")
    lines.append("![ASR Comparison](artifacts/paper_eval/plot_asr_comparison.png)")
    lines.append("")

    # ── Table 2: Over-Refusal ──
    lines.append("## 2. Over-Refusal Rate")
    lines.append("")
    lines.append("Evaluated on XSTest safe prompts. Lower over-refusal = better.")
    lines.append("")
    lines.append("| Model | Baseline | DeepRefusal | Delta |")
    lines.append("|-------|----------|-------------|-------|")
    for m in models:
        r = all_results[m]
        b = r["baseline_overrefusal"]
        d = r["deeprefusal_overrefusal"]
        lines.append(f"| {m} | {b:.4f} | {d:.4f} | {d - b:+.4f} |")
    lines.append("")

    # 4-class breakdown
    lines.append("### 4-Class Response Breakdown")
    lines.append("")
    lines.append("| Model | Variant | Direct Answer | Indirect Answer | Direct Refusal | Indirect Refusal |")
    lines.append("|-------|---------|---------------|-----------------|----------------|------------------|")
    for m in models:
        r = all_results[m]
        for variant in ["baseline", "deeprefusal"]:
            det = r[f"{variant}_overrefusal_detail"]
            lines.append(
                f"| {m} | {variant} | {det['direct_answer']} | {det['indirect_answer']} "
                f"| {det['direct_refusal']} | {det['indirect_refusal']} |"
            )
    lines.append("")

    lines.append("![Over-Refusal Breakdown](artifacts/paper_eval/plot_overrefusal.png)")
    lines.append("")

    # ── Table 3: Capability ──
    lines.append("## 3. Capability Benchmarks")
    lines.append("")
    lines.append(f"MMLU (n={MMLU_SUBSET}) and GSM8k (n={GSM8K_SUBSET}) subsets.")
    lines.append("")
    lines.append("| Model | Baseline MMLU | DR MMLU | Baseline GSM8k | DR GSM8k |")
    lines.append("|-------|--------------|---------|----------------|----------|")
    for m in models:
        r = all_results[m]
        lines.append(
            f"| {m} | {r['baseline_mmlu']:.4f} | {r['deeprefusal_mmlu']:.4f} "
            f"| {r['baseline_gsm8k']:.4f} | {r['deeprefusal_gsm8k']:.4f} |"
        )
    lines.append("")

    lines.append("![Capability Benchmarks](artifacts/paper_eval/plot_capability.png)")
    lines.append("")

    # ── Table 4: Utility ──
    lines.append("## 4. Utility Rate")
    lines.append("")
    lines.append("Fraction of benign UltraChat prompts answered (not refused).")
    lines.append("")
    lines.append("| Model | Baseline | DeepRefusal | Delta |")
    lines.append("|-------|----------|-------------|-------|")
    for m in models:
        r = all_results[m]
        b = r["baseline_utility"]
        d = r["deeprefusal_utility"]
        lines.append(f"| {m} | {b:.4f} | {d:.4f} | {d - b:+.4f} |")
    lines.append("")

    # ── Radar ──
    lines.append("## 5. Radar Plot: Overall Model Comparison")
    lines.append("")
    lines.append("Axes: Safety (1 - ASR), Utility, MMLU, GSM8k, Low Over-Refusal. Higher = better.")
    lines.append("")
    lines.append("![Radar Plot](artifacts/paper_eval/plot_radar.png)")
    lines.append("")

    # ── Delta Heatmap ──
    lines.append("## 6. DeepRefusal Impact Heatmap")
    lines.append("")
    lines.append("Green = improvement from DeepRefusal, Red = degradation.")
    lines.append("")
    lines.append("![Delta Heatmap](artifacts/paper_eval/plot_delta_heatmap.png)")
    lines.append("")

    # ── Methodology ──
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Models**: Gemma3-1B-IT, Qwen3-1.7B, DeepSeek-R1-1.5B")
    lines.append("- **DeepRefusal checkpoints**: LoRA adapters trained with probabilistic ablation (p=0.05, alpha=0.2)")
    lines.append("- **Attack datasets**: HarmBench (standard jailbreak), AdvBench (harmful prompts)")
    lines.append("- **Over-refusal**: XSTest safe subset, 4-class heuristic classification")
    lines.append("- **Capability**: MMLU and GSM8k via HuggingFace datasets")
    lines.append("- **Refusal detection**: Keyword-based RefusalDetector (22 phrases, first 500 chars)")
    lines.append("- **Decoding**: Deterministic (greedy, temperature=1.0, do_sample=False)")
    lines.append("")

    path = str(ROOT / "RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", path)
    return path


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting full evaluation...")
    all_results = run_all()

    # Save raw results JSON
    raw_path = OUTPUT_DIR / "all_results.json"

    # Convert tuples to lists for JSON serialization
    def serialize(obj):
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Saved raw results to %s", raw_path)

    # Generate CSV tables
    write_tables(all_results)

    # Generate plots
    plot_paths = {}
    plot_paths["asr"] = plot_asr_comparison(all_results)
    plot_paths["overrefusal"] = plot_overrefusal(all_results)
    plot_paths["capability"] = plot_capability(all_results)
    plot_paths["radar"] = plot_radar(all_results)
    plot_paths["heatmap"] = plot_delta_heatmap(all_results)

    # Write markdown report
    write_results_md(all_results, plot_paths)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("Results: %s", OUTPUT_DIR)
    logger.info("Report:  RESULTS.md")
    logger.info("=" * 60)
