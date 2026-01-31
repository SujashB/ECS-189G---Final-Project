#!/usr/bin/env python3
"""
Evaluation script for DeepRefusal variants.

Evaluates a checkpoint on:
- HarmBench-Standard (jailbreak success rate)
- AdvBench eval split (harmful refusal)
- XSTest (over-refusal)
- Benign utility: UltraChat subset

Usage:
    python scripts/eval.py \
        --model gemma-2b \
        --variant standard_refusal_training \
        --models-config configs/models.yaml \
        --data-config configs/data.yaml \
        --eval-config configs/eval.yaml
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import load_config, load_jsonl
from src.model_utils import (
    load_model_and_tokenizer,
    generate_text,
    log_gpu_memory,
    cleanup_memory,
)
from src.refusal_detector import RefusalDetector, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_prompt_for_model(
    prompt: str,
    model_type: str,
    chat_templates: dict
) -> str:
    """Format a prompt using the model's chat template (instruction only)."""
    template = chat_templates.get(model_type, chat_templates.get("llama"))
    
    if model_type == "gemma":
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif model_type == "qwen2":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:  # llama-style
        return f"[INST] {prompt} [/INST] "


def evaluate_dataset(
    model,
    tokenizer,
    data: List[Dict[str, Any]],
    dataset_type: str,
    model_type: str,
    chat_templates: dict,
    eval_config: dict,
    detector: RefusalDetector,
    max_samples: Optional[int] = None
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        data: Evaluation data
        dataset_type: Type of dataset
        model_type: Model type for formatting
        chat_templates: Chat templates
        eval_config: Evaluation config
        detector: Refusal detector
        max_samples: Maximum samples to evaluate
        
    Returns:
        Tuple of (predictions, metrics)
    """
    eval_params = eval_config.get("evaluation", {})
    
    if max_samples and len(data) > max_samples:
        data = random.sample(data, max_samples)
    
    logger.info(f"Evaluating {len(data)} samples for {dataset_type}...")
    
    predictions = []
    batch_size = eval_params.get("batch_size", 8)
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc=dataset_type):
        batch = data[i:i + batch_size]
        
        # Format prompts
        prompts = [
            format_prompt_for_model(item["prompt"], model_type, chat_templates)
            for item in batch
        ]
        
        # Generate responses
        responses = generate_text(
            model, tokenizer, prompts,
            max_new_tokens=eval_params.get("max_new_tokens", 256),
            temperature=eval_params.get("temperature", 0.7),
            top_p=eval_params.get("top_p", 0.9),
            do_sample=eval_params.get("do_sample", True),
            batch_size=1  # Process one at a time within batch for memory
        )
        
        # Store predictions
        for item, response in zip(batch, responses):
            pred = {
                "prompt": item["prompt"],
                "response": response,
                "is_refusal": detector.is_refusal(response),
            }
            
            # Add label if present (for XSTest)
            if "label" in item:
                pred["label"] = item["label"]
            
            predictions.append(pred)
    
    # Compute metrics
    metrics = compute_metrics(predictions, dataset_type, detector)
    
    return predictions, metrics


def load_checkpoint_model(
    model_name: str,
    variant: str,
    models_config: dict,
    runs_dir: Path
) -> tuple[Any, Any, dict]:
    """
    Load a trained model checkpoint.
    
    Returns:
        Tuple of (model, tokenizer, run_config)
    """
    run_dir = runs_dir / model_name / variant
    
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")
    
    # Load run config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            run_config = json.load(f)
    else:
        run_config = {}
    
    # Check if this is a base model (no training)
    is_base = run_config.get("is_base", variant == "base_quantized")
    
    if is_base:
        # Load base model directly
        logger.info(f"Loading base model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            models_config,
            load_in_4bit=True,
            use_unsloth=True
        )
    else:
        # Load trained checkpoint
        checkpoint_dir = run_dir / "final"
        if not checkpoint_dir.exists():
            # Try to find latest checkpoint
            checkpoints = list(run_dir.glob("checkpoint-*"))
            if checkpoints:
                checkpoint_dir = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            else:
                raise ValueError(f"No checkpoint found in {run_dir}")
        
        logger.info(f"Loading checkpoint from: {checkpoint_dir}")
        
        # Load with Unsloth
        from unsloth import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint_dir),
            max_seq_length=models_config["models"][model_name].get("max_seq_length", 2048),
            dtype=getattr(torch, models_config["models"][model_name].get("dtype", "bfloat16")),
            load_in_4bit=True,
        )
        
        # Set for inference
        FastLanguageModel.for_inference(model)
    
    return model, tokenizer, run_config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepRefusal model variants"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name from models config"
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Training variant to evaluate"
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default="configs/models.yaml",
        help="Path to models configuration"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data.yaml",
        help="Path to data configuration"
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="configs/eval.yaml",
        help="Path to evaluation configuration"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["harmbench", "advbench", "xstest", "ultrachat"],
        choices=["harmbench", "advbench", "xstest", "ultrachat"],
        help="Datasets to evaluate on"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="./runs",
        help="Base directory for run outputs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load configs
    models_config = load_config(args.models_config)
    data_config = load_config(args.data_config)
    eval_config = load_config(args.eval_config)
    
    runs_dir = Path(args.runs_dir)
    
    # Get model config
    model_config = models_config["models"][args.model]
    model_type = model_config.get("type", "llama")
    chat_templates = models_config.get("chat_templates", {})
    
    # Load model
    logger.info(f"Loading model: {args.model}, variant: {args.variant}")
    model, tokenizer, run_config = load_checkpoint_model(
        args.model, args.variant, models_config, runs_dir
    )
    log_gpu_memory("After model load: ")
    
    # Create refusal detector
    detector = RefusalDetector.from_config(eval_config)
    
    # Setup output directory
    output_dir = runs_dir / args.model / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation data
    processed_dir = Path(data_config["processed_data"]["output_dir"])
    
    all_predictions = {}
    all_metrics = {}
    
    # Evaluate on each dataset
    if "harmbench" in args.datasets:
        harmbench_file = processed_dir / data_config["processed_data"]["harmbench_standard_eval"]
        if harmbench_file.exists():
            harmbench_data = load_jsonl(harmbench_file)
            predictions, metrics = evaluate_dataset(
                model, tokenizer, harmbench_data, "harmbench",
                model_type, chat_templates, eval_config, detector,
                max_samples=args.max_samples
            )
            all_predictions["harmbench"] = predictions
            all_metrics["harmbench"] = metrics
            logger.info(f"HarmBench ASR: {metrics.get('attack_success_rate', 'N/A'):.3f}")
        else:
            logger.warning(f"HarmBench eval file not found: {harmbench_file}")
    
    if "advbench" in args.datasets:
        advbench_file = processed_dir / "advbench_harmful_eval.jsonl"
        if advbench_file.exists():
            advbench_data = load_jsonl(advbench_file)
            predictions, metrics = evaluate_dataset(
                model, tokenizer, advbench_data, "advbench",
                model_type, chat_templates, eval_config, detector,
                max_samples=args.max_samples
            )
            all_predictions["advbench"] = predictions
            all_metrics["advbench"] = metrics
            logger.info(f"AdvBench Refusal Rate: {metrics.get('harmful_refusal_rate', 'N/A'):.3f}")
        else:
            logger.warning(f"AdvBench eval file not found: {advbench_file}")
    
    if "xstest" in args.datasets:
        xstest_file = processed_dir / data_config["processed_data"]["xstest_eval"]
        if xstest_file.exists():
            xstest_data = load_jsonl(xstest_file)
            predictions, metrics = evaluate_dataset(
                model, tokenizer, xstest_data, "xstest",
                model_type, chat_templates, eval_config, detector,
                max_samples=args.max_samples
            )
            all_predictions["xstest"] = predictions
            all_metrics["xstest"] = metrics
            logger.info(f"XSTest Over-refusal Rate: {metrics.get('over_refusal_rate', 'N/A'):.3f}")
        else:
            logger.warning(f"XSTest eval file not found: {xstest_file}")
    
    if "ultrachat" in args.datasets:
        ultrachat_file = processed_dir / "ultrachat_benign_eval.jsonl"
        if ultrachat_file.exists():
            ultrachat_data = load_jsonl(ultrachat_file)
            predictions, metrics = evaluate_dataset(
                model, tokenizer, ultrachat_data, "ultrachat",
                model_type, chat_templates, eval_config, detector,
                max_samples=args.max_samples or 200  # Limit for speed
            )
            all_predictions["ultrachat"] = predictions
            all_metrics["ultrachat"] = metrics
            logger.info(f"UltraChat Utility Rate: {metrics.get('utility_rate', 'N/A'):.3f}")
        else:
            logger.warning(f"UltraChat eval file not found: {ultrachat_file}")
    
    # Save results
    if eval_config.get("output", {}).get("save_predictions", True):
        predictions_file = output_dir / "predictions.jsonl"
        with open(predictions_file, "w") as f:
            for dataset, preds in all_predictions.items():
                for pred in preds:
                    pred["dataset"] = dataset
                    f.write(json.dumps(pred) + "\n")
        logger.info(f"Saved predictions to {predictions_file}")
    
    if eval_config.get("output", {}).get("save_metrics", True):
        metrics_file = output_dir / "metrics.json"
        output_metrics = {
            "model": args.model,
            "variant": args.variant,
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "metrics": all_metrics,
        }
        with open(metrics_file, "w") as f:
            json.dump(output_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Variant: {args.variant}")
    logger.info("-" * 60)
    
    for dataset, metrics in all_metrics.items():
        logger.info(f"\n{dataset.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Cleanup
    cleanup_memory()
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
