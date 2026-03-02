"""Attack execution: load prompts, run generation, classify responses."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path so we can import src modules
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.model_utils import generate_text
from src.refusal_detector import RefusalDetector
from src.data_utils import load_jsonl
from evaluation.types import AttackResult

logger = logging.getLogger(__name__)

# Mapping from paper attack names to data directories
ATTACK_DATA_DIRS: Dict[str, str] = {
    "manual": "data/manual/",
    "codeattack": "data/codeattack/",
    "gcg": "data/gcg/",
    "refusal": "data/refusal/",
    "refusal_transfer": "data/refusal_transfer/",
    "prefill": "data/prefill/",
}

# Default fallback dataset
_FALLBACK_DATASET = "data/processed/harmbench_standard_eval.jsonl"


def load_attack_prompts(
    attack_type: str,
    project_root: Optional[str] = None,
) -> List[str]:
    """
    Load attack prompts for a given attack type.

    Looks in the corresponding data/<attack>/ directory for JSONL files.
    Falls back to harmbench_standard_eval.jsonl if the directory is missing
    or empty.

    Args:
        attack_type: One of the keys in ATTACK_DATA_DIRS.
        project_root: Root of the project (defaults to parent of evaluation/).

    Returns:
        List of prompt strings.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    attack_dir = root / ATTACK_DATA_DIRS.get(attack_type, f"data/{attack_type}/")

    prompts: List[str] = []
    if attack_dir.is_dir():
        jsonl_files = sorted(attack_dir.glob("*.jsonl"))
        for fp in jsonl_files:
            for record in load_jsonl(str(fp)):
                if "prompt" in record:
                    prompts.append(record["prompt"])

    if not prompts:
        fallback = root / _FALLBACK_DATASET
        if fallback.exists():
            logger.warning(
                "No prompts found for attack '%s' in %s; "
                "falling back to %s",
                attack_type, attack_dir, fallback,
            )
            for record in load_jsonl(str(fallback)):
                if "prompt" in record:
                    prompts.append(record["prompt"])
        else:
            logger.error("Fallback dataset %s not found either.", fallback)

    return prompts


def run_attack(
    model,
    tokenizer,
    prompts: List[str],
    attack_type: str,
    gen_params: Optional[Dict[str, Any]] = None,
    detector: Optional[RefusalDetector] = None,
    batch_size: int = 1,
) -> AttackResult:
    """
    Run an attack: generate responses and classify as refusal/non-refusal.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompts: List of attack prompts.
        attack_type: Name of the attack (for labelling).
        gen_params: Generation parameters for generate_text.
        detector: RefusalDetector instance (created with defaults if None).
        batch_size: Batch size for generation.

    Returns:
        AttackResult with counts and outputs.
    """
    if detector is None:
        detector = RefusalDetector()

    params = gen_params or {}
    logger.info("Running attack '%s' on %d prompts", attack_type, len(prompts))

    responses = generate_text(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        **params,
    )

    refusals = detector.detect_batch(responses)
    successes = sum(1 for r in refusals if not r)  # non-refusal = attack success
    failures = sum(1 for r in refusals if r)

    result = AttackResult(
        attack_name=attack_type,
        total=len(prompts),
        successes=successes,
        failures=failures,
        outputs=responses,
    )
    logger.info(
        "Attack '%s': ASR=%.3f (%d/%d)",
        attack_type, result.asr, successes, len(prompts),
    )
    return result


def run_all_paper_attacks(
    model,
    tokenizer,
    gen_params: Optional[Dict[str, Any]] = None,
    detector: Optional[RefusalDetector] = None,
    project_root: Optional[str] = None,
    batch_size: int = 1,
) -> List[AttackResult]:
    """
    Run all 6 paper attack types and return results.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        gen_params: Generation parameters.
        detector: Shared RefusalDetector.
        project_root: Project root directory.
        batch_size: Batch size for generation.

    Returns:
        List of AttackResult, one per attack type.
    """
    if detector is None:
        detector = RefusalDetector()

    results: List[AttackResult] = []
    for attack_type in ATTACK_DATA_DIRS:
        prompts = load_attack_prompts(attack_type, project_root=project_root)
        if not prompts:
            logger.warning("Skipping attack '%s': no prompts available.", attack_type)
            continue
        result = run_attack(
            model, tokenizer, prompts, attack_type,
            gen_params=gen_params, detector=detector, batch_size=batch_size,
        )
        results.append(result)

    return results
