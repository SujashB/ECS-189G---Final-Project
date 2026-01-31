"""
Data loading and processing utilities.
Supports both HuggingFace datasets and local files.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from datasets import Dataset, load_dataset


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_advbench(
    config: Dict[str, Any],
    split: str = "train"
) -> List[Dict[str, str]]:
    """
    Load AdvBench dataset for harmful prompts.
    
    Args:
        config: Data configuration dictionary
        split: Which split to load
        
    Returns:
        List of dictionaries with 'prompt' key
    """
    advbench_config = config["datasets"]["advbench"]
    
    if advbench_config.get("use_local") and advbench_config.get("local_path"):
        # Load from local file
        local_path = Path(advbench_config["local_path"])
        if local_path.suffix == ".jsonl":
            data = []
            with open(local_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        elif local_path.suffix == ".json":
            with open(local_path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {local_path.suffix}")
    else:
        # Load from HuggingFace
        ds = load_dataset(
            advbench_config["hf_path"],
            split=advbench_config.get("hf_split", split)
        )
        prompt_col = advbench_config.get("prompt_column", "goal")
        data = [{"prompt": row[prompt_col]} for row in ds]
    
    return data


def load_harmbench(
    config: Dict[str, Any],
    token: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Load HarmBench dataset for jailbreak evaluation.
    
    Args:
        config: Data configuration dictionary
        token: HuggingFace token for gated datasets
        
    Returns:
        List of dictionaries with 'prompt' key
    """
    harmbench_config = config["datasets"]["harmbench"]
    
    if harmbench_config.get("use_local") and harmbench_config.get("local_path"):
        local_path = Path(harmbench_config["local_path"])
        if local_path.suffix == ".jsonl":
            data = []
            with open(local_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            with open(local_path, "r") as f:
                data = json.load(f)
    else:
        ds = load_dataset(
            harmbench_config["hf_path"],
            harmbench_config.get("hf_config", "standard"),
            split=harmbench_config.get("hf_split", "test"),
            token=token
        )
        prompt_col = harmbench_config.get("prompt_column", "prompt")
        data = [{"prompt": row[prompt_col]} for row in ds]
    
    return data


def load_xstest(
    config: Dict[str, Any],
    token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load XSTest dataset for over-refusal evaluation.
    
    Args:
        config: Data configuration dictionary
        token: HuggingFace token for gated datasets
        
    Returns:
        List of dictionaries with 'prompt' and 'label' keys
    """
    xstest_config = config["datasets"]["xstest"]
    
    if xstest_config.get("use_local") and xstest_config.get("local_path"):
        local_path = Path(xstest_config["local_path"])
        if local_path.suffix == ".jsonl":
            data = []
            with open(local_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            with open(local_path, "r") as f:
                data = json.load(f)
    else:
        ds = load_dataset(
            xstest_config["hf_path"],
            split=xstest_config.get("hf_split", "test"),
            token=token
        )
        prompt_col = xstest_config.get("prompt_column", "prompt")
        label_col = xstest_config.get("label_column", "label")
        data = [
            {"prompt": row[prompt_col], "label": row.get(label_col, "unknown")}
            for row in ds
        ]
    
    return data


def load_ultrachat(
    config: Dict[str, Any],
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load UltraChat dataset for benign instruction-response pairs.
    
    Args:
        config: Data configuration dictionary
        max_samples: Maximum number of samples to load
        
    Returns:
        List of dictionaries with 'prompt' and 'response' keys
    """
    ultrachat_config = config["datasets"]["ultrachat"]
    max_samples = max_samples or ultrachat_config.get("max_samples", 10000)
    
    if ultrachat_config.get("use_local") and ultrachat_config.get("local_path"):
        local_path = Path(ultrachat_config["local_path"])
        if local_path.suffix == ".jsonl":
            data = []
            with open(local_path, "r") as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    data.append(json.loads(line))
        else:
            with open(local_path, "r") as f:
                data = json.load(f)
                if max_samples:
                    data = data[:max_samples]
    else:
        ds = load_dataset(
            ultrachat_config["hf_path"],
            split=ultrachat_config.get("hf_split", "train_sft")
        )
        
        # UltraChat has 'messages' format: list of {role, content}
        data = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            messages = row.get("messages", [])
            if len(messages) >= 2:
                # Extract first user message and assistant response
                user_msg = None
                assistant_msg = None
                for msg in messages:
                    if msg.get("role") == "user" and user_msg is None:
                        user_msg = msg.get("content", "")
                    elif msg.get("role") == "assistant" and assistant_msg is None:
                        assistant_msg = msg.get("content", "")
                    if user_msg and assistant_msg:
                        break
                if user_msg and assistant_msg:
                    data.append({"prompt": user_msg, "response": assistant_msg})
    
    return data


def format_refusal_training_data(
    harmful_prompts: List[Dict[str, str]],
    refusal_templates: List[str],
    seed: int = 42
) -> List[Dict[str, str]]:
    """
    Format harmful prompts into refusal training pairs.
    
    Args:
        harmful_prompts: List of harmful prompts
        refusal_templates: List of refusal response templates
        seed: Random seed for template selection
        
    Returns:
        List of dictionaries with 'prompt' and 'response' keys
    """
    random.seed(seed)
    
    formatted_data = []
    for item in harmful_prompts:
        prompt = item["prompt"]
        # Randomly select a refusal template
        response = random.choice(refusal_templates)
        formatted_data.append({
            "prompt": prompt,
            "response": response,
            "type": "harmful_refusal"
        })
    
    return formatted_data


def save_jsonl(data: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Save data to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(input_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_train_eval_split(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.9,
    seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into training and evaluation sets.
    
    Args:
        data: List of data items
        train_ratio: Ratio of data for training
        seed: Random seed
        
    Returns:
        Tuple of (train_data, eval_data)
    """
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def prepare_sft_dataset(
    data: List[Dict[str, str]],
    tokenizer,
    model_type: str,
    chat_templates: Dict[str, str],
    max_length: int = 1024
) -> Dataset:
    """
    Prepare data for SFT training with proper chat formatting.
    
    Args:
        data: List of prompt-response pairs
        tokenizer: HuggingFace tokenizer
        model_type: Type of model (gemma, qwen2, llama)
        chat_templates: Dictionary of chat templates per model type
        max_length: Maximum sequence length
        
    Returns:
        HuggingFace Dataset ready for training
    """
    template = chat_templates.get(model_type, chat_templates.get("llama"))
    
    formatted_texts = []
    for item in data:
        text = template.format(
            instruction=item["prompt"],
            response=item["response"]
        )
        formatted_texts.append({"text": text})
    
    return Dataset.from_list(formatted_texts)
