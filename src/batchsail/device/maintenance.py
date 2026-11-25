"""Idle-device cache maintenance contracts."""

from dataclasses import dataclass
from typing import Any

from .types import Residency


@dataclass(frozen=True, slots=True)
class TrimPlan:
    device: str
    epoch: int
    victims: tuple[Residency, ...]
    keep: tuple[Residency, ...]


@dataclass(frozen=True, slots=True)
class TrimResult:
    device: str
    outcome: str
    removed: tuple[Residency, ...] = ()
    remaining: tuple[Residency, ...] = ()
    detail: str | None = None

    @staticmethod
    def _resident(entry: Residency) -> dict[str, Any]:
        return {
            **entry.key.to_json(),
            "weight_bytes": entry.weight_bytes,
            "generation": entry.generation,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "outcome": self.outcome,
            "removed_bytes": sum(item.weight_bytes for item in self.removed),
            "removed": [self._resident(item) for item in self.removed],
            "remaining": [self._resident(item) for item in self.remaining],
            "detail": self.detail,
        }
