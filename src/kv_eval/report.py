"""Artifact writing for KV eval."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from .types import AggregateMetrics, EvalRecord, RunConfig


def make_run_dir(output_root: str) -> Path:
    run_dir = Path(output_root) / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_raw_jsonl(path: Path, records: Iterable[EvalRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=True) + "\n")


def write_summary_json(path: Path, config: RunConfig, summaries: Iterable[AggregateMetrics], notes: Dict[str, str]) -> None:
    payload = {
        "config": config.to_dict(),
        "notes": notes,
        "summaries": [s.to_dict() for s in summaries],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_tables_csv(path: Path, summaries: Iterable[AggregateMetrics]) -> None:
    rows = [s.to_dict() for s in summaries]
    fieldnames: List[str] = list(rows[0].keys()) if rows else []

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
