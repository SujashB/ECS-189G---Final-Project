#!/usr/bin/env python3
"""Re-evaluate DeepSeek-R1 1.5B on MMLU, GSM8k, and Utility for both baseline and DeepRefusal."""

import gc
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model_utils import load_config, generate_text, cleanup_memory
from src.refusal_detector import RefusalDetector
from src.data_utils import load_jsonl
from evaluation.capability import evaluate_mmlu, evaluate_gsm8k

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "artifacts" / "paper_eval"
RESULTS_PATH = OUTPUT_DIR / "all_results.json"
ULTRACHAT_PATH = str(ROOT / "data" / "processed" / "ultrachat_benign_eval.jsonl")

MMLU_SUBSET = 100
GSM8K_SUBSET = 100
MAX_UTILITY_SAMPLES = 30
SEED = 42


def load_model(model_name, checkpoint=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    models_config = load_config(str(ROOT / "configs" / "models.yaml"))
    model_config = models_config["models"][model_name]
    model_path = model_config["path"]
    dtype = getattr(torch, model_config.get("dtype", "bfloat16"))

    _tokenizer_hub = {
        "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    }

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    except Exception:
        hub_path = _tokenizer_hub.get(model_name)
        if hub_path:
            tokenizer = AutoTokenizer.from_pretrained(hub_path, trust_remote_code=True)

    if tokenizer is None:
        raise RuntimeError(f"Cannot load tokenizer for {model_name}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


def eval_utility(model, tokenizer, detector):
    records = load_jsonl(ULTRACHAT_PATH)[:MAX_UTILITY_SAMPLES]
    prompts = [r["prompt"] for r in records]
    if not prompts:
        return 0.0
    from evaluation.types import EvalConfig
    gen_params = EvalConfig(seed=SEED, deterministic=True, max_new_tokens=128).get_gen_params()
    responses = generate_text(model, tokenizer, prompts, **gen_params)
    refusal_rate = detector.get_refusal_rate(responses)
    return 1.0 - refusal_rate


def free_model(*args):
    for obj in args:
        del obj
    gc.collect()
    cleanup_memory()


def main():
    detector = RefusalDetector()
    checkpoint = str(ROOT / "runs" / "deepseek-r1-1.5b" / "deeprefusal")

    # Load existing results
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)

    # ── Baseline evaluation ──
    logger.info("=" * 60)
    logger.info("Evaluating DeepSeek-R1 1.5B BASELINE on MMLU, GSM8k, Utility")
    logger.info("=" * 60)
    model, tokenizer = load_model("deepseek-r1-1.5b")

    logger.info("Running MMLU (subset=%d)...", MMLU_SUBSET)
    mmlu = evaluate_mmlu(model, tokenizer, subset_size=MMLU_SUBSET, seed=SEED)
    logger.info("Baseline MMLU: %.4f (%d/%d)", mmlu.accuracy, mmlu.correct, mmlu.total)
    all_results["deepseek-r1-1.5b"]["baseline_mmlu"] = mmlu.accuracy

    logger.info("Running GSM8k (subset=%d)...", GSM8K_SUBSET)
    gsm = evaluate_gsm8k(model, tokenizer, subset_size=GSM8K_SUBSET, seed=SEED)
    logger.info("Baseline GSM8k: %.4f (%d/%d)", gsm.accuracy, gsm.correct, gsm.total)
    all_results["deepseek-r1-1.5b"]["baseline_gsm8k"] = gsm.accuracy

    logger.info("Running Utility...")
    utility = eval_utility(model, tokenizer, detector)
    logger.info("Baseline Utility: %.4f", utility)
    all_results["deepseek-r1-1.5b"]["baseline_utility"] = utility

    free_model(model, tokenizer)

    # ── DeepRefusal evaluation ──
    logger.info("=" * 60)
    logger.info("Evaluating DeepSeek-R1 1.5B DEEPREFUSAL on MMLU, GSM8k, Utility")
    logger.info("=" * 60)
    model, tokenizer = load_model("deepseek-r1-1.5b", checkpoint=checkpoint)

    logger.info("Running MMLU (subset=%d)...", MMLU_SUBSET)
    mmlu_dr = evaluate_mmlu(model, tokenizer, subset_size=MMLU_SUBSET, seed=SEED)
    logger.info("DeepRefusal MMLU: %.4f (%d/%d)", mmlu_dr.accuracy, mmlu_dr.correct, mmlu_dr.total)
    all_results["deepseek-r1-1.5b"]["deeprefusal_mmlu"] = mmlu_dr.accuracy

    logger.info("Running GSM8k (subset=%d)...", GSM8K_SUBSET)
    gsm_dr = evaluate_gsm8k(model, tokenizer, subset_size=GSM8K_SUBSET, seed=SEED)
    logger.info("DeepRefusal GSM8k: %.4f (%d/%d)", gsm_dr.accuracy, gsm_dr.correct, gsm_dr.total)
    all_results["deepseek-r1-1.5b"]["deeprefusal_gsm8k"] = gsm_dr.accuracy

    logger.info("Running Utility...")
    utility_dr = eval_utility(model, tokenizer, detector)
    logger.info("DeepRefusal Utility: %.4f", utility_dr)
    all_results["deepseek-r1-1.5b"]["deeprefusal_utility"] = utility_dr

    free_model(model, tokenizer)

    # ── Save updated results ──
    def serialize(obj):
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    logger.info("Updated results saved to %s", RESULTS_PATH)

    # Print summary
    r = all_results["deepseek-r1-1.5b"]
    logger.info("=" * 60)
    logger.info("DEEPSEEK-R1 1.5B UPDATED RESULTS")
    logger.info("  Baseline  MMLU: %.2f%%  GSM8k: %.2f%%  Utility: %.2f%%",
                r["baseline_mmlu"] * 100, r["baseline_gsm8k"] * 100, r["baseline_utility"] * 100)
    logger.info("  DeepRef   MMLU: %.2f%%  GSM8k: %.2f%%  Utility: %.2f%%",
                r["deeprefusal_mmlu"] * 100, r["deeprefusal_gsm8k"] * 100, r["deeprefusal_utility"] * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
