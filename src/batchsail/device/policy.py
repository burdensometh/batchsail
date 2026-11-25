"""V1 DMA-style page accounting for resident weights and packed workspaces."""

from dataclasses import dataclass
from typing import Any

from ..codec import fields, identifier, integer
from ..errors import Invalid
from ..registry.model import ModelRevision
from .types import MemoryDemand


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    name: str
    capacity_bytes: int

    def __post_init__(self) -> None:
        identifier(self.name, "device")
        integer(self.capacity_bytes, "device capacity", minimum=1, maximum=2**48)

    @classmethod
    def from_config(cls, value: Any) -> "DeviceSpec":
        spec = fields(value, ("name", "capacity_bytes"), "device")
        return cls(spec["name"], spec["capacity_bytes"])


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    page_bytes: int = 4096
    workspace_alignment: int = 64
    slot_bytes: int = 48

    def __post_init__(self) -> None:
        for name, value in (
            ("page bytes", self.page_bytes),
            ("workspace alignment", self.workspace_alignment),
        ):
            integer(value, name, minimum=1, maximum=1_048_576)
            if value & (value - 1):
                raise Invalid(f"{name} must be a power of two")
        integer(self.slot_bytes, "slot bytes", minimum=0, maximum=65_536)

    def align_workspace(self, nbytes: int) -> int:
        alignment = self.workspace_alignment
        return (nbytes + alignment - 1) // alignment * alignment

    def weight_bytes(self, revision: ModelRevision) -> int:
        reserved = revision.pages * self.page_bytes
        if reserved < revision.weight_nbytes:
            raise Invalid(f"{revision.key.label} declares fewer pages than its float32 weights require")
        return reserved

    @classmethod
    def from_config(cls, value: Any) -> "MemoryPolicy":
        data = fields(value, ("page_bytes", "workspace_alignment", "slot_bytes"), "memory policy")
        return cls(data["page_bytes"], data["workspace_alignment"], data["slot_bytes"])


def memory_demand(
    policy: MemoryPolicy, revision: ModelRevision, *, padded_rows: int, slots: int,
) -> MemoryDemand:
    if not 1 <= slots or not 1 <= padded_rows <= revision.input_spec.max_rows:
        raise Invalid("workspace requires at least one slot and a valid padded row count")
    packed_rows = padded_rows * slots
    workspace = packed_rows * revision.row_workspace_bytes + slots * policy.slot_bytes
    return MemoryDemand(revision.key, policy.weight_bytes(revision), policy.align_workspace(workspace))
