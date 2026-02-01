"""
Model loading and management utilities.
Supports Unsloth for efficient QLoRA training.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_config(
    model_name: str,
    models_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model in config
        models_config: Full models configuration
        
    Returns:
        Model configuration dictionary
    """
    if model_name not in models_config["models"]:
        available = list(models_config["models"].keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    return models_config["models"][model_name]


def load_model_and_tokenizer(
    model_name: str,
    models_config: Dict[str, Any],
    load_in_4bit: bool = True,
    use_unsloth: bool = True,
    max_seq_length: Optional[int] = None
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer using Unsloth for efficient inference/training.
    
    Args:
        model_name: Name of model in config
        models_config: Models configuration dictionary
        load_in_4bit: Whether to use 4-bit quantization
        use_unsloth: Whether to use Unsloth for loading
        max_seq_length: Override max sequence length
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = get_model_config(model_name, models_config)
    model_path = model_config["path"]
    dtype = getattr(torch, model_config.get("dtype", "bfloat16"))
    seq_length = max_seq_length or model_config.get("max_seq_length", 2048)
    
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"4-bit: {load_in_4bit}, dtype: {dtype}, max_seq_length: {seq_length}")
    
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            
            # Map local paths to HuggingFace hub for Unsloth (tokenizer bug workaround)
            # Unsloth works better with HF hub paths for newer models like Gemma 3
            hf_model_map = {
                "./models/gemma3-1b-it": "google/gemma-3-1b-it",
                "./models/qwen3-1.7b": "Qwen/Qwen3-1.7B",
                "./models/deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            }
            
            # Use HF hub path if available, otherwise use local path
            unsloth_model_path = hf_model_map.get(model_path, model_path)
            logger.info(f"Unsloth loading from: {unsloth_model_path}")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=unsloth_model_path,
                max_seq_length=seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
        except (ImportError, NotImplementedError, RuntimeError, TypeError) as e:
            logger.warning(f"Unsloth failed ({e}), falling back to standard loading")
            use_unsloth = False
    
    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from peft import prepare_model_for_kbit_training
        
        # Load config first
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = getattr(config, 'model_type', 'unknown')
            logger.info(f"Detected model type: {model_type}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            config = None
            model_type = 'unknown'
        
        # Load tokenizer - transformers 4.57 has a bug with local Gemma 3 tokenizer
        # Workaround: load from HuggingFace Hub based on model type
        tokenizer = None
        
        # Map local model types to HF hub paths for tokenizer
        hf_tokenizer_map = {
            'gemma3_text': 'google/gemma-3-1b-it',
            'gemma3': 'google/gemma-3-1b-it',
            'gemma2': 'google/gemma-2-2b-it',
            'gemma': 'google/gemma-2b-it',
            'qwen2': 'Qwen/Qwen2.5-1.5B-Instruct',
            'llama': None,  # Usually works locally
        }
        
        # Try local first
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
            )
            logger.info("Loaded tokenizer from local path")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Local tokenizer failed ({e})")
            
            # Try HF hub fallback based on model type
            hf_path = hf_tokenizer_map.get(model_type)
            if hf_path:
                logger.info(f"Trying tokenizer from HuggingFace: {hf_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        hf_path,
                        trust_remote_code=True,
                    )
                    logger.info(f"Loaded tokenizer from {hf_path}")
                except Exception as e2:
                    logger.error(f"HF tokenizer also failed: {e2}")
        
        if tokenizer is None:
            raise RuntimeError(f"Could not load tokenizer for {model_path}. "
                             "Try: pip install --upgrade transformers")
        
        # Check if CUDA is available for quantization
        cuda_available = torch.cuda.is_available()
        
        if load_in_4bit and cuda_available:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
        elif cuda_available:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        else:
            # CPU fallback (for testing without GPU)
            logger.warning("CUDA not available, loading model on CPU (slow)")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float32,  # CPU doesn't support bfloat16 well
                trust_remote_code=True,
            )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def setup_lora(
    model,
    train_config: Dict[str, Any],
    use_unsloth: bool = True
):
    """
    Setup LoRA adapters for the model.
    
    Args:
        model: The base model
        train_config: Training configuration with LoRA params
        use_unsloth: Whether to use Unsloth
        
    Returns:
        Model with LoRA adapters
    """
    qlora_config = train_config["qlora"]
    
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            
            model = FastLanguageModel.get_peft_model(
                model,
                r=qlora_config["r"],
                lora_alpha=qlora_config["lora_alpha"],
                lora_dropout=qlora_config["lora_dropout"],
                target_modules=qlora_config["target_modules"],
                bias=qlora_config["bias"],
                use_gradient_checkpointing="unsloth",
                random_state=train_config["training"].get("seed", 42),
            )
            return model
        except (ImportError, NotImplementedError, RuntimeError) as e:
            logger.warning(f"Unsloth LoRA setup failed ({e}), using PEFT")
    
    # Fallback to standard PEFT
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        r=qlora_config["r"],
        lora_alpha=qlora_config["lora_alpha"],
        lora_dropout=qlora_config["lora_dropout"],
        target_modules=qlora_config["target_modules"],
        bias=qlora_config["bias"],
        task_type=qlora_config["task_type"],
    )
    model = get_peft_model(model, lora_config)
    
    return model


def get_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int = -1,
    token_aggregation: str = "last",
    batch_size: int = 8,
    max_length: int = 512
) -> torch.Tensor:
    """
    Extract hidden states from a specific layer for given prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of text prompts
        layer_idx: Layer index to extract from (-1 = last)
        token_aggregation: How to aggregate tokens ("last" or "mean")
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        Tensor of hidden states [num_prompts, hidden_dim]
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_hidden_states = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get hidden states from specified layer
        hidden_states = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
        
        # Aggregate across tokens
        if token_aggregation == "last":
            # Get last non-padding token for each sequence
            attention_mask = inputs["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_hidden = torch.stack([
                hidden_states[j, seq_lengths[j], :]
                for j in range(hidden_states.size(0))
            ])
        elif token_aggregation == "mean":
            # Mean over all non-padding tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            batch_hidden = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
        else:
            raise ValueError(f"Unknown token_aggregation: {token_aggregation}")
        
        all_hidden_states.append(batch_hidden.cpu())
    
    return torch.cat(all_hidden_states, dim=0)


def generate_text(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    batch_size: int = 1
) -> list[str]:
    """
    Generate text responses for given prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample
        batch_size: Batch size for generation
        
    Returns:
        List of generated responses
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_length:]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_responses.extend(responses)
    
    return all_responses


def save_model(
    model,
    tokenizer,
    output_path: str,
    save_full: bool = False
) -> None:
    """
    Save model and tokenizer.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer
        output_path: Path to save to
        save_full: Whether to save full model or just adapters
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_path)
    
    tokenizer.save_pretrained(output_path)
    logger.info(f"Model saved to {output_path}")


def cleanup_memory() -> None:
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_gpu_memory(prefix: str = "") -> Dict[str, float]:
    """
    Log current GPU memory usage.
    
    Args:
        prefix: Prefix for log message
        
    Returns:
        Dictionary with memory stats
    """
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    stats = {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated
    }
    
    logger.info(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
    
    return stats
