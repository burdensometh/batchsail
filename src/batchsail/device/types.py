"""Memory planning and address contracts between scheduling and execution."""

from dataclasses import dataclass

from ..registry.model import ModelKey


@dataclass(frozen=True, slots=True)
class MemoryDemand:
    key: ModelKey
    weight_bytes: int
    workspace_bytes: int


@dataclass(frozen=True, slots=True)
class Residency:
    key: ModelKey
    generation: int
    weight_bytes: int


@dataclass(frozen=True, slots=True)
class Placement:
    device: str
    epoch: int
    demand: MemoryDemand
    victims: tuple[Residency, ...]
    cached: bool
    bytes_after: int

    @property
    def eviction_bytes(self) -> int:
        return sum(victim.weight_bytes for victim in self.victims)


@dataclass(frozen=True, slots=True)
class DeviceLease:
    device: str
    token: int
    residency: Residency
    workspace_bytes: int
    cache: tuple[Residency, ...]

    @property
    def key(self) -> ModelKey:
        return self.residency.key


@dataclass(frozen=True, slots=True)
class DeviceRejection:
    device: str
    reason: str
    missing_bytes: int
    locked_revisions: tuple[ModelKey, ...] = ()
