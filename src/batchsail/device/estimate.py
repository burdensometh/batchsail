"""Report whether a shape could be scheduled without mutating device state."""

from dataclasses import dataclass
from typing import Any

from .types import DeviceRejection, MemoryDemand, Placement


@dataclass(frozen=True, slots=True)
class CapacityEstimate:
    demand: MemoryDemand
    rows: int
    padded_rows: int
    slots: int
    decisions: tuple[Placement | DeviceRejection, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            **self.demand.key.to_json(),
            "rows": self.rows,
            "padded_rows": self.padded_rows,
            "slots": self.slots,
            "weight_bytes": self.demand.weight_bytes,
            "workspace_bytes": self.demand.workspace_bytes,
            "minimum_device_bytes": self.demand.weight_bytes + self.demand.workspace_bytes,
            "eligible": any(isinstance(item, Placement) for item in self.decisions),
            "devices": [self._decision(item) for item in self.decisions],
        }

    @staticmethod
    def _decision(item: Placement | DeviceRejection) -> dict[str, Any]:
        if isinstance(item, Placement):
            return {
                "name": item.device,
                "eligible": True,
                "cache_hit": item.cached,
                "bytes_after": item.bytes_after,
                "evictions": [victim.key.to_json() for victim in item.victims],
            }
        return {
            "name": item.device,
            "eligible": False,
            "reason": item.reason,
            "missing_bytes": item.missing_bytes,
            "locked_revisions": [key.to_json() for key in item.locked_revisions],
        }
