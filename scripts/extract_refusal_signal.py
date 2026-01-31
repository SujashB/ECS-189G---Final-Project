#!/usr/bin/env python3
"""
Extract refusal direction from model hidden states.

Stage 1 of DeepRefusal: Identify the refusal direction vector
by computing the difference between mean harmful and mean benign
hidden states.

Usage:
    python scripts/extract_refusal_signal.py \
        --model gemma-2b \
        --models-config configs/models.yaml \
        --data-config configs/data.yaml \
        --intervention-config configs/intervention.yaml
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import load_config, load_jsonl
from src.model_utils import (
    load_model_and_tokenizer,
    get_hidden_states,
    log_gpu_memory,
    cleanup_memory,
)
from src.interventions import (
    compute_refusal_direction,
    InferenceInterventionManager,
)
from src.refusal_detector import RefusalDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def sample_prompts(
    data: list[dict],
    n_samples: int,
    seed: int = 42
) -> list[str]:
    """Sample prompts from dataset."""
    random.seed(seed)
    sampled = random.sample(data, min(n_samples, len(data)))
    return [item["prompt"] for item in sampled]


def format_prompt_for_model(
    prompt: str,
    model_type: str,
    chat_templates: dict
) -> str:
    """Format a prompt using the model's chat template (instruction only)."""
    template = chat_templates.get(model_type, chat_templates.get("llama"))
    # Only include instruction part, not response
    # Extract just the user turn
    if "{instruction}" in template:
        # Find where instruction ends (before response placeholder)
        parts = template.split("{response}")
        if len(parts) > 0:
            user_part = parts[0].replace("{instruction}", prompt)
            return user_part.rstrip()
    return prompt


def run_verification_experiment(
    model,
    tokenizer,
    refusal_direction: torch.Tensor,
    harmful_prompts: list[str],
    benign_prompts: list[str],
    alpha_values: list[float],
    model_type: str,
    chat_templates: dict,
    eval_config: dict
) -> dict:
    """
    Run verification experiment to validate refusal direction.
    
    Tests:
    1. Adding +alpha*d to benign prompts should increase refusal
    2. Removing projection from harmful prompts should decrease refusal
    """
    logger.info("Running verification experiment...")
    
    detector = RefusalDetector.from_config(eval_config)
    results = {"strengthen_benign": {}, "weaken_harmful": {}}
    
    # Format prompts
    formatted_benign = [
        format_prompt_for_model(p, model_type, chat_templates)
        for p in benign_prompts
    ]
    formatted_harmful = [
        format_prompt_for_model(p, model_type, chat_templates)
        for p in harmful_prompts
    ]
    
    # Test 1: Strengthen refusal on benign prompts (should increase refusal)
    logger.info("Test 1: Strengthening refusal signal on benign prompts...")
    for alpha in alpha_values:
        if alpha == 0:
            # Baseline: no intervention
            from src.model_utils import generate_text
            responses = generate_text(
                model, tokenizer, formatted_benign,
                max_new_tokens=128, do_sample=False
            )
        else:
            with InferenceInterventionManager(
                model, refusal_direction,
                alpha=alpha, layers=[-1, -2], mode="strengthen"
            ):
                from src.model_utils import generate_text
                responses = generate_text(
                    model, tokenizer, formatted_benign,
                    max_new_tokens=128, do_sample=False
                )
        
        refusal_rate = detector.get_refusal_rate(responses)
        results["strengthen_benign"][str(alpha)] = {
            "refusal_rate": refusal_rate,
            "n_samples": len(responses)
        }
        logger.info(f"  alpha={alpha}: refusal_rate={refusal_rate:.3f}")
    
    # Test 2: Weaken refusal on harmful prompts (should decrease refusal)
    logger.info("Test 2: Weakening refusal signal on harmful prompts...")
    for alpha in alpha_values:
        if alpha == 0:
            from src.model_utils import generate_text
            responses = generate_text(
                model, tokenizer, formatted_harmful,
                max_new_tokens=128, do_sample=False
            )
        else:
            with InferenceInterventionManager(
                model, refusal_direction,
                alpha=alpha, layers=[-1, -2], mode="weaken"
            ):
                from src.model_utils import generate_text
                responses = generate_text(
                    model, tokenizer, formatted_harmful,
                    max_new_tokens=128, do_sample=False
                )
        
        refusal_rate = detector.get_refusal_rate(responses)
        results["weaken_harmful"][str(alpha)] = {
            "refusal_rate": refusal_rate,
            "n_samples": len(responses)
        }
        logger.info(f"  alpha={alpha}: refusal_rate={refusal_rate:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract refusal direction from model hidden states"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name from models config"
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
        "--intervention-config",
        type=str,
        default="configs/intervention.yaml",
        help="Path to intervention configuration"
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="configs/eval.yaml",
        help="Path to eval configuration"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples for direction extraction (overrides config)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index for hidden state extraction (overrides config)"
    )
    parser.add_argument(
        "--run-verification",
        action="store_true",
        help="Run verification experiment after extraction"
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
    intervention_config = load_config(args.intervention_config)
    eval_config = load_config(args.eval_config)
    
    # Get extraction parameters
    refusal_dir_config = intervention_config["refusal_direction"]
    n_samples = args.n_samples or refusal_dir_config.get("n_samples", 200)
    layer_idx = args.layer if args.layer is not None else refusal_dir_config.get("extraction_layer", -1)
    token_aggregation = refusal_dir_config.get("token_aggregation", "last")
    
    logger.info(f"Extraction config: n_samples={n_samples}, layer={layer_idx}, "
                f"token_aggregation={token_aggregation}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        models_config,
        load_in_4bit=True,
        use_unsloth=True
    )
    log_gpu_memory("After model load: ")
    
    # Get model type for chat template
    model_config = models_config["models"][args.model]
    model_type = model_config.get("type", "llama")
    chat_templates = models_config.get("chat_templates", {})
    
    # Load harmful and benign prompts
    processed_dir = Path(data_config["processed_data"]["output_dir"])
    
    harmful_file = processed_dir / data_config["processed_data"]["advbench_harmful_train"]
    benign_file = processed_dir / data_config["processed_data"]["ultrachat_benign_train"]
    
    if not harmful_file.exists() or not benign_file.exists():
        logger.error("Processed data files not found. Run prepare_data.py first.")
        sys.exit(1)
    
    harmful_data = load_jsonl(harmful_file)
    benign_data = load_jsonl(benign_file)
    
    logger.info(f"Loaded {len(harmful_data)} harmful and {len(benign_data)} benign samples")
    
    # Sample prompts
    harmful_prompts = sample_prompts(harmful_data, n_samples, seed=args.seed)
    benign_prompts = sample_prompts(benign_data, n_samples, seed=args.seed + 1)
    
    # Format prompts with chat template
    formatted_harmful = [
        format_prompt_for_model(p, model_type, chat_templates)
        for p in harmful_prompts
    ]
    formatted_benign = [
        format_prompt_for_model(p, model_type, chat_templates)
        for p in benign_prompts
    ]
    
    # Extract hidden states
    logger.info(f"Extracting hidden states from layer {layer_idx}...")
    
    harmful_hidden = get_hidden_states(
        model, tokenizer, formatted_harmful,
        layer_idx=layer_idx,
        token_aggregation=token_aggregation,
        batch_size=8
    )
    logger.info(f"Harmful hidden states shape: {harmful_hidden.shape}")
    
    benign_hidden = get_hidden_states(
        model, tokenizer, formatted_benign,
        layer_idx=layer_idx,
        token_aggregation=token_aggregation,
        batch_size=8
    )
    logger.info(f"Benign hidden states shape: {benign_hidden.shape}")
    
    log_gpu_memory("After hidden state extraction: ")
    
    # Compute refusal direction
    logger.info("Computing refusal direction...")
    refusal_direction = compute_refusal_direction(
        harmful_hidden,
        benign_hidden,
        normalize=refusal_dir_config.get("normalize", True)
    )
    logger.info(f"Refusal direction shape: {refusal_direction.shape}")
    logger.info(f"Refusal direction norm: {refusal_direction.norm().item():.4f}")
    
    # Save artifacts
    artifacts_dir = Path("artifacts") / args.model
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save refusal direction
    refusal_dir_path = artifacts_dir / "refusal_dir.pt"
    torch.save(refusal_direction, refusal_dir_path)
    logger.info(f"Saved refusal direction to {refusal_dir_path}")
    
    # Save metadata
    metadata = {
        "model": args.model,
        "model_path": model_config["path"],
        "layer_index": layer_idx,
        "token_aggregation": token_aggregation,
        "n_harmful_samples": len(harmful_prompts),
        "n_benign_samples": len(benign_prompts),
        "hidden_dim": refusal_direction.shape[0],
        "direction_norm": refusal_direction.norm().item(),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    
    metadata_path = artifacts_dir / "refusal_dir_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Run verification experiment if requested
    if args.run_verification:
        logger.info("=" * 50)
        logger.info("Running verification experiment...")
        
        alpha_values = intervention_config["verification"]["alpha_values"]
        n_verification = intervention_config["verification"]["n_samples"]
        
        # Sample new prompts for verification
        verify_harmful = sample_prompts(harmful_data, n_verification, seed=args.seed + 100)
        verify_benign = sample_prompts(benign_data, n_verification, seed=args.seed + 101)
        
        verification_results = run_verification_experiment(
            model, tokenizer, refusal_direction,
            verify_harmful, verify_benign,
            alpha_values, model_type, chat_templates, eval_config
        )
        
        # Save verification results
        verification_path = artifacts_dir / "verification_results.json"
        with open(verification_path, "w") as f:
            json.dump(verification_results, f, indent=2)
        logger.info(f"Saved verification results to {verification_path}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("Verification Summary:")
        logger.info("Strengthening refusal on benign (should increase refusal rate):")
        for alpha, res in verification_results["strengthen_benign"].items():
            logger.info(f"  alpha={alpha}: {res['refusal_rate']:.3f}")
        
        logger.info("Weakening refusal on harmful (should decrease refusal rate):")
        for alpha, res in verification_results["weaken_harmful"].items():
            logger.info(f"  alpha={alpha}: {res['refusal_rate']:.3f}")
    
    # Cleanup
    cleanup_memory()
    
    logger.info("=" * 50)
    logger.info("Refusal direction extraction complete!")
    logger.info(f"Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
