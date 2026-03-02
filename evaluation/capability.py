"""Capability benchmarks: MMLU and GSM8k."""

import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.model_utils import generate_text
from evaluation.types import CapabilityResult

logger = logging.getLogger(__name__)


def _extract_answer_letter(text: str) -> Optional[str]:
    """Extract the first A/B/C/D answer letter from model output."""
    # Try patterns like "The answer is A", "Answer: B", just "C", etc.
    patterns = [
        r"(?:answer|option)\s*(?:is|:)\s*\(?([A-Da-d])\)?",
        r"^\s*\(?([A-Da-d])\)?[\s\.\)]",
        r"\b([A-Da-d])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _extract_numeric_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from GSM8k-style reasoning."""
    # Look for #### <number> pattern (GSM8k ground truth format)
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    # Look for "the answer is <number>"
    m = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*\$?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # Fall back to last number in the text
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def evaluate_mmlu(
    model,
    tokenizer,
    subset_size: Optional[int] = None,
    seed: int = 42,
    batch_size: int = 1,
) -> CapabilityResult:
    """
    Evaluate model on MMLU benchmark.

    Loads from HuggingFace (cais/mmlu or lukaemon/mmlu), formats as
    multiple-choice, extracts model answer letter.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        subset_size: If set, only evaluate on this many examples.
        seed: Random seed for subset sampling.
        batch_size: Batch size for generation.

    Returns:
        CapabilityResult with accuracy metrics.
    """
    from datasets import load_dataset

    # Try loading MMLU
    ds = None
    for dataset_name in ["cais/mmlu", "lukaemon/mmlu", "tasksource/mmlu"]:
        try:
            ds = load_dataset(dataset_name, "all", split="test", trust_remote_code=True)
            logger.info("Loaded MMLU from %s (%d examples)", dataset_name, len(ds))
            break
        except Exception as e:
            logger.debug("Could not load %s: %s", dataset_name, e)
            continue

    if ds is None:
        logger.error("Could not load MMLU from any source.")
        return CapabilityResult(benchmark_name="mmlu", accuracy=0.0, total=0, correct=0)

    # Subsample if requested
    indices = list(range(len(ds)))
    if subset_size and subset_size < len(ds):
        rng = random.Random(seed)
        indices = rng.sample(indices, subset_size)

    correct = 0
    total = len(indices)

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        prompts = []
        gold_answers = []

        for idx in batch_indices:
            row = ds[idx]
            question = row.get("question", row.get("input", ""))
            choices = row.get("choices", [])
            answer = row.get("answer", "")

            # Build multiple-choice prompt
            if choices:
                options = "\n".join(
                    f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
                )
                prompt = (
                    f"Question: {question}\n\n{options}\n\n"
                    "Answer with just the letter (A, B, C, or D):"
                )
            else:
                prompt = f"Question: {question}\nAnswer with just the letter (A, B, C, or D):"

            prompts.append(prompt)

            # Normalize gold answer
            if isinstance(answer, int):
                gold_answers.append(chr(65 + answer))
            else:
                gold_answers.append(str(answer).strip().upper())

        responses = generate_text(
            model, tokenizer, prompts,
            max_new_tokens=32, do_sample=False, batch_size=batch_size,
        )

        for resp, gold in zip(responses, gold_answers):
            pred = _extract_answer_letter(resp)
            if pred == gold:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("MMLU: %d/%d correct (%.3f)", correct, total, accuracy)
    return CapabilityResult(
        benchmark_name="mmlu", accuracy=accuracy, total=total, correct=correct,
    )


def evaluate_gsm8k(
    model,
    tokenizer,
    subset_size: Optional[int] = None,
    seed: int = 42,
    batch_size: int = 1,
) -> CapabilityResult:
    """
    Evaluate model on GSM8k math reasoning benchmark.

    Loads from HuggingFace (openai/gsm8k), generates response,
    extracts final numeric answer via regex, exact-matches ground truth.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        subset_size: If set, only evaluate on this many examples.
        seed: Random seed for subset sampling.
        batch_size: Batch size for generation.

    Returns:
        CapabilityResult with accuracy metrics.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        logger.info("Loaded GSM8k (%d examples)", len(ds))
    except Exception as e:
        logger.error("Could not load GSM8k: %s", e)
        return CapabilityResult(benchmark_name="gsm8k", accuracy=0.0, total=0, correct=0)

    indices = list(range(len(ds)))
    if subset_size and subset_size < len(ds):
        rng = random.Random(seed)
        indices = rng.sample(indices, subset_size)

    correct = 0
    total = len(indices)

    for batch_start in range(0, total, batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        prompts = []
        gold_answers = []

        for idx in batch_indices:
            row = ds[idx]
            question = row["question"]
            answer_text = row["answer"]

            prompt = (
                f"Solve the following math problem step by step.\n\n"
                f"Question: {question}\n\n"
                f"Show your work and give the final answer as a number."
            )
            prompts.append(prompt)

            # Extract gold numeric answer from "#### <number>"
            gold = _extract_numeric_answer(answer_text)
            gold_answers.append(gold)

        responses = generate_text(
            model, tokenizer, prompts,
            max_new_tokens=256, do_sample=False, batch_size=batch_size,
        )

        for resp, gold in zip(responses, gold_answers):
            if gold is None:
                total -= 1
                continue
            pred = _extract_numeric_answer(resp)
            if pred is not None and pred == gold:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("GSM8k: %d/%d correct (%.3f)", correct, total, accuracy)
    return CapabilityResult(
        benchmark_name="gsm8k", accuracy=accuracy, total=total, correct=correct,
    )
