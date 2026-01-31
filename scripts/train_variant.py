#!/usr/bin/env python3
"""
Train model variants for DeepRefusal reproduction.

Supports 4 variants:
1. base_quantized - No training, just evaluate base model
2. standard_refusal_training - Harmful refusal + benign ultrachat  
3. refusal_plus_prefill_aug - Variant 2 + harmful-prefix augmentation
4. deeprefusal_quantized - Variant 3 + random refusal weakening intervention

Usage:
    python scripts/train_variant.py \
        --model gemma-2b \
        --variant standard_refusal_training \
        --models-config configs/models.yaml \
        --train-config configs/train.yaml \
        --data-config configs/data.yaml \
        --intervention-config configs/intervention.yaml
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import load_config, load_jsonl, prepare_sft_dataset
from src.model_utils import (
    load_model_and_tokenizer,
    setup_lora,
    save_model,
    log_gpu_memory,
    cleanup_memory,
)
from src.interventions import RandomRefusalWeakeningCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


VARIANTS = [
    "base_quantized",
    "standard_refusal_training",
    "refusal_plus_prefill_aug",
    "deeprefusal_quantized"
]


def load_training_data(
    data_config: Dict[str, Any],
    variant: str,
    train_config: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Load training data based on variant.
    
    Args:
        data_config: Data configuration
        variant: Training variant
        train_config: Training configuration
        
    Returns:
        Combined training data
    """
    processed_dir = Path(data_config["processed_data"]["output_dir"])
    
    # Load harmful refusal data
    harmful_file = processed_dir / data_config["processed_data"]["advbench_harmful_train"]
    harmful_data = load_jsonl(harmful_file)
    logger.info(f"Loaded {len(harmful_data)} harmful refusal samples")
    
    # Load benign utility data
    benign_file = processed_dir / data_config["processed_data"]["ultrachat_benign_train"]
    benign_data = load_jsonl(benign_file)
    logger.info(f"Loaded {len(benign_data)} benign utility samples")
    
    # Get mixing ratios
    mixing = train_config.get("data_mixing", {})
    harmful_ratio = mixing.get("harmful_refusal_ratio", 0.5)
    benign_ratio = mixing.get("benign_utility_ratio", 0.5)
    
    # Balance datasets
    total_samples = min(
        len(harmful_data) / harmful_ratio,
        len(benign_data) / benign_ratio
    )
    n_harmful = int(total_samples * harmful_ratio)
    n_benign = int(total_samples * benign_ratio)
    
    # Sample if needed
    if len(harmful_data) > n_harmful:
        harmful_data = random.sample(harmful_data, n_harmful)
    if len(benign_data) > n_benign:
        benign_data = random.sample(benign_data, n_benign)
    
    combined_data = harmful_data + benign_data
    
    # Add harmful prefix data for variants 3 and 4
    if variant in ["refusal_plus_prefill_aug", "deeprefusal_quantized"]:
        prefix_file = processed_dir / data_config["processed_data"]["harmful_prefix_train"]
        if prefix_file.exists():
            prefix_data = load_jsonl(prefix_file)
            
            # Get prefix ratio from config
            prefix_config = train_config.get("prefix_augmentation", {})
            prefix_ratio = prefix_config.get("prefix_ratio", 0.3)
            n_prefix = int(len(harmful_data) * prefix_ratio)
            
            if len(prefix_data) > n_prefix:
                prefix_data = random.sample(prefix_data, n_prefix)
            
            combined_data.extend(prefix_data)
            logger.info(f"Added {len(prefix_data)} harmful prefix samples")
        else:
            logger.warning(f"Harmful prefix file not found: {prefix_file}")
            logger.warning("Run make_harmful_prefix_data.py first for variants 3-4")
    
    random.shuffle(combined_data)
    logger.info(f"Total training samples: {len(combined_data)}")
    
    return combined_data


class DeepRefusalTrainerCallback:
    """
    Custom callback for applying DeepRefusal interventions during training.
    Wraps the RandomRefusalWeakeningCallback for TRL/Unsloth trainer.
    """
    
    def __init__(
        self,
        model,
        refusal_direction: torch.Tensor,
        intervention_config: Dict[str, Any]
    ):
        self.intervention = RandomRefusalWeakeningCallback(
            model,
            refusal_direction,
            intervention_config
        )
        self.step_count = 0
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the start of each training step."""
        self.intervention.on_step_begin()
        self.step_count += 1
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        self.intervention.on_step_end()


def train_with_unsloth(
    model,
    tokenizer,
    train_dataset: Dataset,
    train_config: Dict[str, Any],
    output_dir: Path,
    variant: str,
    refusal_direction: Optional[torch.Tensor] = None,
    intervention_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Train model using Unsloth + TRL SFTTrainer.
    
    Args:
        model: The model with LoRA adapters
        tokenizer: Tokenizer
        train_dataset: Training dataset
        train_config: Training configuration
        output_dir: Output directory
        variant: Training variant
        refusal_direction: Refusal direction for DeepRefusal
        intervention_config: Intervention config for DeepRefusal
    """
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    training_params = train_config["training"]
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_params.get("num_train_epochs", 3),
        max_steps=training_params.get("max_steps", -1),
        per_device_train_batch_size=training_params.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training_params.get("gradient_accumulation_steps", 4),
        learning_rate=training_params.get("learning_rate", 2e-4),
        weight_decay=training_params.get("weight_decay", 0.01),
        warmup_ratio=training_params.get("warmup_ratio", 0.03),
        lr_scheduler_type=training_params.get("lr_scheduler_type", "cosine"),
        optim=training_params.get("optim", "adamw_8bit"),
        logging_steps=training_params.get("logging_steps", 10),
        save_steps=training_params.get("save_steps", 100),
        save_total_limit=train_config.get("output", {}).get("save_total_limit", 2),
        bf16=training_params.get("bf16", True),
        fp16=training_params.get("fp16", False),
        gradient_checkpointing=training_params.get("gradient_checkpointing", True),
        seed=training_params.get("seed", 42),
        report_to="none",  # Disable wandb by default
    )
    
    # Setup callbacks for DeepRefusal variant
    callbacks = []
    if variant == "deeprefusal_quantized" and refusal_direction is not None:
        logger.info("Setting up DeepRefusal intervention callback...")
        callback = DeepRefusalTrainerCallback(
            model, refusal_direction, intervention_config
        )
        callbacks.append(callback)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=training_params.get("max_seq_length", 1024),
        packing=False,
    )
    
    # Add custom callback for DeepRefusal
    if callbacks:
        for cb in callbacks:
            # Manually handle callbacks since SFTTrainer may not support custom ones directly
            original_training_step = trainer.training_step
            
            def wrapped_training_step(model, inputs, cb=cb):
                cb.on_step_begin(None, None, None)
                result = original_training_step(model, inputs)
                cb.on_step_end(None, None, None)
                return result
            
            trainer.training_step = wrapped_training_step
    
    # Train
    logger.info("Starting training...")
    log_gpu_memory("Before training: ")
    
    trainer.train()
    
    log_gpu_memory("After training: ")
    
    # Save final model
    final_output = output_dir / "final"
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    logger.info(f"Saved final model to {final_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Train model variants for DeepRefusal reproduction"
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
        choices=VARIANTS,
        help="Training variant"
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default="configs/models.yaml",
        help="Path to models configuration"
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train.yaml",
        help="Path to training configuration"
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: runs/<model>/<variant>)"
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configs
    models_config = load_config(args.models_config)
    train_config = load_config(args.train_config)
    data_config = load_config(args.data_config)
    intervention_config = load_config(args.intervention_config)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_output = Path(train_config.get("output", {}).get("base_dir", "./runs"))
        output_dir = base_output / args.model / args.variant
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Get model config
    model_config = models_config["models"][args.model]
    model_type = model_config.get("type", "llama")
    chat_templates = models_config.get("chat_templates", {})
    
    # Handle base_quantized variant (no training needed)
    if args.variant == "base_quantized":
        logger.info("Variant: base_quantized - No training needed")
        logger.info("Loading model for evaluation only...")
        
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            models_config,
            load_in_4bit=True,
            use_unsloth=True
        )
        
        # Save model path info for evaluation
        config_out = {
            "model": args.model,
            "variant": args.variant,
            "model_path": model_config["path"],
            "is_base": True,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_out, f, indent=2)
        
        logger.info("Base model ready for evaluation")
        cleanup_memory()
        return
    
    # Load model for training
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        models_config,
        load_in_4bit=True,
        use_unsloth=True
    )
    log_gpu_memory("After model load: ")
    
    # Setup LoRA
    logger.info("Setting up LoRA adapters...")
    model = setup_lora(model, train_config, use_unsloth=True)
    log_gpu_memory("After LoRA setup: ")
    
    # Load training data
    logger.info(f"Loading training data for variant: {args.variant}")
    training_data = load_training_data(data_config, args.variant, train_config)
    
    # Prepare dataset
    train_dataset = prepare_sft_dataset(
        training_data,
        tokenizer,
        model_type,
        chat_templates,
        max_length=train_config["training"].get("max_seq_length", 1024)
    )
    logger.info(f"Prepared {len(train_dataset)} training examples")
    
    # Load refusal direction for DeepRefusal variant
    refusal_direction = None
    if args.variant == "deeprefusal_quantized":
        artifacts_dir = Path("artifacts") / args.model
        refusal_dir_path = artifacts_dir / "refusal_dir.pt"
        
        if not refusal_dir_path.exists():
            logger.error(f"Refusal direction not found: {refusal_dir_path}")
            logger.error("Run extract_refusal_signal.py first for DeepRefusal variant")
            sys.exit(1)
        
        refusal_direction = torch.load(refusal_dir_path)
        logger.info(f"Loaded refusal direction from {refusal_dir_path}")
    
    # Train
    train_with_unsloth(
        model,
        tokenizer,
        train_dataset,
        train_config,
        output_dir,
        args.variant,
        refusal_direction=refusal_direction,
        intervention_config=intervention_config
    )
    
    # Save training config
    config_out = {
        "model": args.model,
        "variant": args.variant,
        "model_path": model_config["path"],
        "train_samples": len(training_data),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "train_config": train_config["training"],
        "qlora_config": train_config["qlora"],
    }
    
    if args.variant == "deeprefusal_quantized":
        config_out["intervention_config"] = intervention_config["intervention"]
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)
    
    # Cleanup
    cleanup_memory()
    
    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
