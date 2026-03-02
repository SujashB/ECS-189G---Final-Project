"""Structured CSV reporting for evaluation results."""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.types import AttackResult, CapabilityResult, OverRefusalResult

logger = logging.getLogger(__name__)


def _ensure_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_core_asr_table(
    results: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Write table_core_asr.csv.

    Each row in `results` should have keys:
        model, attack, baseline_asr, deeprefusal_asr, delta_asr

    Returns:
        Path to the written CSV file.
    """
    out = _ensure_dir(output_dir) / "table_core_asr.csv"
    fieldnames = ["model", "attack", "baseline_asr", "deeprefusal_asr", "delta_asr"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    logger.info("Wrote %s (%d rows)", out, len(results))
    return str(out)


def generate_overrefusal_table(
    results: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Write table_overrefusal.csv.

    Each row: model, baseline_overrefusal, deeprefusal_overrefusal, delta
    """
    out = _ensure_dir(output_dir) / "table_overrefusal.csv"
    fieldnames = ["model", "baseline_overrefusal", "deeprefusal_overrefusal", "delta"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    logger.info("Wrote %s (%d rows)", out, len(results))
    return str(out)


def generate_capability_table(
    results: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Write table_capability.csv.

    Each row: model, baseline_mmlu, deeprefusal_mmlu, baseline_gsm8k, deeprefusal_gsm8k
    """
    out = _ensure_dir(output_dir) / "table_capability.csv"
    fieldnames = [
        "model", "baseline_mmlu", "deeprefusal_mmlu",
        "baseline_gsm8k", "deeprefusal_gsm8k",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    logger.info("Wrote %s (%d rows)", out, len(results))
    return str(out)


def generate_kv_extension_table(
    results: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Write table_kv_extension.csv.

    Flexible schema — writes whatever keys are present in results.
    """
    out = _ensure_dir(output_dir) / "table_kv_extension.csv"
    if not results:
        logger.warning("No KV extension results to write.")
        return str(out)
    fieldnames = list(results[0].keys())
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    logger.info("Wrote %s (%d rows)", out, len(results))
    return str(out)


def generate_p_sweep_table(
    results: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Write table_p_sweep.csv.

    Flexible schema — writes whatever keys are present in results.
    """
    out = _ensure_dir(output_dir) / "table_p_sweep.csv"
    if not results:
        logger.warning("No p-sweep results to write.")
        return str(out)
    fieldnames = list(results[0].keys())
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    logger.info("Wrote %s (%d rows)", out, len(results))
    return str(out)


def generate_all_tables(
    core_asr_results: Optional[List[Dict[str, Any]]] = None,
    overrefusal_results: Optional[List[Dict[str, Any]]] = None,
    capability_results: Optional[List[Dict[str, Any]]] = None,
    kv_extension_results: Optional[List[Dict[str, Any]]] = None,
    p_sweep_results: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = "artifacts/eval_output",
) -> Dict[str, str]:
    """
    Convenience function: generate all available tables.

    Returns:
        Dict mapping table name to file path.
    """
    paths: Dict[str, str] = {}
    if core_asr_results:
        paths["core_asr"] = generate_core_asr_table(core_asr_results, output_dir)
    if overrefusal_results:
        paths["overrefusal"] = generate_overrefusal_table(overrefusal_results, output_dir)
    if capability_results:
        paths["capability"] = generate_capability_table(capability_results, output_dir)
    if kv_extension_results:
        paths["kv_extension"] = generate_kv_extension_table(kv_extension_results, output_dir)
    if p_sweep_results:
        paths["p_sweep"] = generate_p_sweep_table(p_sweep_results, output_dir)
    return paths
