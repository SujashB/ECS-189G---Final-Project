#!/usr/bin/env python3
"""
Generate harmful prefix augmentation data.

For each harmful prompt, generate a short harmful prefix (20-60 tokens)
using the BASE model (no safety intervention), then create training examples
where input includes prompt + harmful prefix, target is refusal.

Usage:
    python scripts/make_harmful_prefix_data.py \
        --model gemma-2b \
        --models-config configs/models.yaml \
        --data-config configs/data.yaml
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import load_config, load_jsonl, save_jsonl
from src.model_utils import (
    load_model_and_tokenizer,
    log_gpu_memory,
    cleanup_memory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_prompt_for_generation(
    prompt: str,
    model_type: str,
    chat_templates: dict,
    add_harmful_start: bool = True
) -> str:
    """
    Format prompt to elicit harmful prefix generation.
    
    The model should continue as if it's going to comply with the harmful request.
    """
    template = chat_templates.get(model_type, chat_templates.get("llama"))
    
    # Format: instruction + start of assistant response
    if model_type == "gemma":
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        if add_harmful_start:
            # Add a small start to encourage harmful continuation
            formatted += "Sure, here's"
    elif model_type == "qwen2":
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        if add_harmful_start:
            formatted += "Certainly,"
    else:  # llama-style
        formatted = f"[INST] {prompt} [/INST] "
        if add_harmful_start:
            formatted += "Of course,"
    
    return formatted


def generate_harmful_prefix(
    model,
    tokenizer,
    prompt: str,
    model_type: str,
    chat_templates: dict,
    min_tokens: int = 20,
    max_tokens: int = 60,
    temperature: float = 1.0,
    top_p: float = 0.9
) -> Optional[str]:
    """
    Generate a harmful prefix for a prompt.
    
    Returns the generated prefix or None if generation failed.
    """
    formatted = format_prompt_for_generation(prompt, model_type, chat_templates)
    
    device = next(model.parameters()).device
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    
    try:
        with torch.no_grad():
            # Generate with sampling to get diverse prefixes
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Extract generated tokens
        generated_ids = outputs[0, input_length:]
        prefix = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up prefix
        prefix = prefix.strip()
        
        # Check if prefix is long enough
        if len(prefix.split()) < 5:
            return None
        
        return prefix
    
    except Exception as e:
        logger.warning(f"Generation failed: {e}")
        return None


def is_prefix_harmful(prefix: str) -> bool:
    """
    Basic check if prefix seems harmful/compliant (not a refusal).
    
    Returns True if the prefix doesn't contain obvious refusal patterns.
    """
    refusal_patterns = [
        "i cannot", "i can't", "i'm sorry", "i apologize",
        "i am not able", "i'm not able", "i won't", "i will not",
        "not appropriate", "cannot help", "cannot assist",
        "against my", "harmful", "unethical", "illegal"
    ]
    
    prefix_lower = prefix.lower()
    for pattern in refusal_patterns:
        if pattern in prefix_lower:
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate harmful prefix augmentation data"
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
        "--min-tokens",
        type=int,
        default=20,
        help="Minimum tokens for prefix"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Maximum tokens for prefix"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process"
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
    
    # Get model config
    model_config = models_config["models"][args.model]
    model_type = model_config.get("type", "llama")
    chat_templates = models_config.get("chat_templates", {})
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        models_config,
        load_in_4bit=True,
        use_unsloth=True
    )
    log_gpu_memory("After model load: ")
    
    # Load harmful prompts
    processed_dir = Path(data_config["processed_data"]["output_dir"])
    harmful_file = processed_dir / data_config["processed_data"]["advbench_harmful_train"]
    
    if not harmful_file.exists():
        logger.error(f"Harmful data file not found: {harmful_file}")
        logger.error("Run prepare_data.py first.")
        sys.exit(1)
    
    harmful_data = load_jsonl(harmful_file)
    logger.info(f"Loaded {len(harmful_data)} harmful prompts")
    
    if args.max_samples:
        harmful_data = harmful_data[:args.max_samples]
        logger.info(f"Using {len(harmful_data)} samples")
    
    # Get refusal templates for training targets
    refusal_templates = data_config.get("refusal_templates", [
        "I cannot help with that request. It appears to involve potentially harmful content."
    ])
    
    # Generate prefixes
    logger.info("Generating harmful prefixes...")
    prefix_data = []
    
    for item in tqdm(harmful_data, desc="Generating prefixes"):
        prompt = item["prompt"]
        
        # Generate prefix
        prefix = generate_harmful_prefix(
            model, tokenizer, prompt,
            model_type, chat_templates,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        if prefix and is_prefix_harmful(prefix):
            # Create training example
            # Input: original prompt + harmful prefix
            # Target: refusal
            combined_prompt = f"{prompt}\n\n[Assistant started to respond:] {prefix}"
            refusal_response = random.choice(refusal_templates)
            
            prefix_data.append({
                "prompt": combined_prompt,
                "response": refusal_response,
                "original_prompt": prompt,
                "harmful_prefix": prefix,
                "type": "harmful_prefix_refusal"
            })
    
    logger.info(f"Generated {len(prefix_data)} harmful prefix examples")
    logger.info(f"Success rate: {len(prefix_data) / len(harmful_data) * 100:.1f}%")
    
    # Save
    output_file = processed_dir / data_config["processed_data"]["harmful_prefix_train"]
    save_jsonl(prefix_data, output_file)
    logger.info(f"Saved to {output_file}")
    
    # Save metadata
    metadata = {
        "model": args.model,
        "n_input_prompts": len(harmful_data),
        "n_generated_prefixes": len(prefix_data),
        "success_rate": len(prefix_data) / len(harmful_data),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    
    metadata_file = processed_dir / "harmful_prefix_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")
    
    # Cleanup
    cleanup_memory()
    
    logger.info("Harmful prefix generation complete!")


if __name__ == "__main__":
    main()
