"""Types for KV-cache robustness evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RunConfig:
    models: List[str]
    track: str = "hybrid"  # ollama, hf, hybrid
    attack_modes: Optional[List[str]] = None
    harmful_samples: int = 100
    benign_samples: int = 100
    output_root: str = "artifacts/kv_cache_eval"
    seed: int = 42
    defense_params: Optional[List[Dict[str, float]]] = None
    decode_params: Optional[Dict[str, Any]] = None
    cache_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalRecord:
    model: str
    track: str
    attack_mode: str
    defense_on: bool
    defense_name: str
    prompt_id: str
    prompt_text: str
    is_harmful: bool
    refused: bool
    response: str
    latency_ms: float
    tokens_out: int
    session_reused: bool = False
    kv_len_prefill: Optional[int] = None
    kv_len_decode: Optional[int] = None
    cache_reset: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    model: str
    track: str
    attack_mode: str
    defense_name: str
    n_total: int
    n_harmful: int
    n_benign: int
    refusal_rate: float
    asr: float
    over_refusal_rate: float
    utility_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    toks_per_sec: float
    asr_ci_low: float
    asr_ci_high: float
    over_refusal_ci_low: float
    over_refusal_ci_high: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
