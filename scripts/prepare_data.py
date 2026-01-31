#!/usr/bin/env python3
"""
Data preparation script.
Loads and processes datasets for training and evaluation.

Usage:
    python scripts/prepare_data.py --data-config configs/data.yaml
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    load_config,
    load_advbench,
    load_harmbench,
    load_xstest,
    load_ultrachat,
    format_refusal_training_data,
    save_jsonl,
    create_train_eval_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment or cache."""
    # Check environment variables
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    if token:
        return token
    
    # Try to get from huggingface_hub cache
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
    except Exception:
        pass
    
    return token


def prepare_advbench(
    config: dict,
    output_dir: Path
) -> None:
    """Prepare AdvBench harmful prompts for refusal training."""
    logger.info("Loading AdvBench dataset...")
    
    try:
        data = load_advbench(config)
        logger.info(f"Loaded {len(data)} harmful prompts from AdvBench")
    except Exception as e:
        logger.error(f"Failed to load AdvBench: {e}")
        raise
    
    # Get refusal templates
    refusal_templates = config.get("refusal_templates", [
        "I cannot help with that request. It appears to involve potentially harmful content."
    ])
    
    # Format as refusal training pairs
    formatted_data = format_refusal_training_data(
        data,
        refusal_templates,
        seed=config.get("splits", {}).get("seed", 42)
    )
    
    # Split into train/eval
    train_ratio = config.get("splits", {}).get("train_ratio", 0.9)
    train_data, eval_data = create_train_eval_split(
        formatted_data,
        train_ratio=train_ratio,
        seed=config.get("splits", {}).get("seed", 42)
    )
    
    # Save training data
    output_file = output_dir / config["processed_data"]["advbench_harmful_train"]
    save_jsonl(train_data, output_file)
    logger.info(f"Saved {len(train_data)} training samples to {output_file}")
    
    # Save eval data with different name
    eval_file = output_dir / "advbench_harmful_eval.jsonl"
    save_jsonl(eval_data, eval_file)
    logger.info(f"Saved {len(eval_data)} eval samples to {eval_file}")


def prepare_ultrachat(
    config: dict,
    output_dir: Path
) -> None:
    """Prepare UltraChat benign instruction-response pairs."""
    logger.info("Loading UltraChat dataset...")
    
    max_samples = config["datasets"]["ultrachat"].get("max_samples", 10000)
    
    try:
        data = load_ultrachat(config, max_samples=max_samples)
        logger.info(f"Loaded {len(data)} benign pairs from UltraChat")
    except Exception as e:
        logger.error(f"Failed to load UltraChat: {e}")
        raise
    
    # Add type marker
    for item in data:
        item["type"] = "benign_utility"
    
    # Split into train/eval
    train_ratio = config.get("splits", {}).get("train_ratio", 0.9)
    train_data, eval_data = create_train_eval_split(
        data,
        train_ratio=train_ratio,
        seed=config.get("splits", {}).get("seed", 42)
    )
    
    # Save training data
    output_file = output_dir / config["processed_data"]["ultrachat_benign_train"]
    save_jsonl(train_data, output_file)
    logger.info(f"Saved {len(train_data)} training samples to {output_file}")
    
    # Save eval data
    eval_file = output_dir / "ultrachat_benign_eval.jsonl"
    save_jsonl(eval_data, eval_file)
    logger.info(f"Saved {len(eval_data)} eval samples to {eval_file}")


def prepare_xstest(
    config: dict,
    output_dir: Path,
    token: str | None = None
) -> None:
    """Prepare XSTest for over-refusal evaluation."""
    logger.info("Loading XSTest dataset...")
    
    try:
        data = load_xstest(config, token=token)
        logger.info(f"Loaded {len(data)} samples from XSTest")
    except Exception as e:
        logger.error(f"Failed to load XSTest: {e}")
        raise
    
    # Add type marker
    for item in data:
        item["type"] = "over_refusal_eval"
    
    # Count labels
    safe_count = sum(1 for item in data if item.get("label") == "safe")
    unsafe_count = sum(1 for item in data if item.get("label") == "unsafe")
    logger.info(f"XSTest: {safe_count} safe, {unsafe_count} unsafe samples")
    
    # Save
    output_file = output_dir / config["processed_data"]["xstest_eval"]
    save_jsonl(data, output_file)
    logger.info(f"Saved {len(data)} samples to {output_file}")


def prepare_harmbench(
    config: dict,
    output_dir: Path,
    token: str | None = None
) -> None:
    """Prepare HarmBench for jailbreak evaluation."""
    logger.info("Loading HarmBench dataset...")
    
    try:
        data = load_harmbench(config, token=token)
        logger.info(f"Loaded {len(data)} samples from HarmBench")
    except Exception as e:
        logger.error(f"Failed to load HarmBench: {e}")
        logger.warning("HarmBench is a gated dataset. You may need to:")
        logger.warning("  1. Accept the license at https://huggingface.co/datasets/walledai/HarmBench")
        logger.warning("  2. Run: huggingface-cli login")
        raise
    
    # Add type marker
    for item in data:
        item["type"] = "jailbreak_eval"
    
    # Save
    output_file = output_dir / config["processed_data"]["harmbench_standard_eval"]
    save_jsonl(data, output_file)
    logger.info(f"Saved {len(data)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for DeepRefusal reproduction"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["advbench", "ultrachat", "xstest", "harmbench"],
        choices=["advbench", "ultrachat", "xstest", "harmbench"],
        help="Which datasets to prepare"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    
    # Load config
    logger.info(f"Loading config from {args.data_config}")
    config = load_config(args.data_config)
    
    # Override seed if specified
    if "splits" not in config:
        config["splits"] = {}
    config["splits"]["seed"] = args.seed
    
    # Create output directory
    output_dir = Path(config["processed_data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Get HF token for gated datasets
    token = get_hf_token()
    if token:
        logger.info("HuggingFace token found")
    else:
        logger.warning("No HuggingFace token found. Gated datasets may fail.")
    
    # Prepare each dataset
    errors = []
    
    if "advbench" in args.datasets:
        try:
            prepare_advbench(config, output_dir)
        except Exception as e:
            errors.append(("advbench", str(e)))
    
    if "ultrachat" in args.datasets:
        try:
            prepare_ultrachat(config, output_dir)
        except Exception as e:
            errors.append(("ultrachat", str(e)))
    
    if "xstest" in args.datasets:
        try:
            prepare_xstest(config, output_dir, token=token)
        except Exception as e:
            errors.append(("xstest", str(e)))
    
    if "harmbench" in args.datasets:
        try:
            prepare_harmbench(config, output_dir, token=token)
        except Exception as e:
            errors.append(("harmbench", str(e)))
    
    # Summary
    logger.info("=" * 50)
    logger.info("Data preparation complete!")
    logger.info(f"Output directory: {output_dir}")
    
    if errors:
        logger.warning(f"Errors occurred for {len(errors)} dataset(s):")
        for dataset, error in errors:
            logger.warning(f"  - {dataset}: {error}")
    else:
        logger.info("All datasets prepared successfully!")
    
    # List output files
    logger.info("Output files:")
    for f in sorted(output_dir.glob("*.jsonl")):
        size = f.stat().st_size / 1024
        logger.info(f"  - {f.name} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
