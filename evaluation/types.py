"""Core data types for the evaluation framework."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AttackResult:
    """Result of running a single attack type."""
    attack_name: str
    total: int
    successes: int
    failures: int
    outputs: List[str] = field(default_factory=list)

    @property
    def asr(self) -> float:
        """Attack Success Rate = successes / total."""
        if self.total == 0:
            return 0.0
        return self.successes / self.total


@dataclass
class OverRefusalResult:
    """Result of 4-class over-refusal evaluation."""
    direct_answer: int = 0
    indirect_answer: int = 0
    direct_refusal: int = 0
    indirect_refusal: int = 0

    @property
    def total(self) -> int:
        return (self.direct_answer + self.indirect_answer
                + self.direct_refusal + self.indirect_refusal)

    @property
    def over_refusal_rate(self) -> float:
        """Fraction of responses that are any kind of refusal on benign prompts."""
        if self.total == 0:
            return 0.0
        return (self.direct_refusal + self.indirect_refusal) / self.total


@dataclass
class CapabilityResult:
    """Result of a capability benchmark (MMLU or GSM8k)."""
    benchmark_name: str
    accuracy: float
    total: int
    correct: int


@dataclass
class EvalConfig:
    """Configuration for deterministic/reproducible evaluation."""
    seed: int = 42
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    max_new_tokens: int = 256
    deterministic: bool = False

    def get_gen_params(self) -> dict:
        """Return generation parameters dict for model_utils.generate_text."""
        if self.deterministic:
            return {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 1.0,
                "top_p": 1.0,
                "do_sample": False,
            }
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }
