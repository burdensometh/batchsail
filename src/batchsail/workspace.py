"""Version-one metadata shared by writable runtime and offline inspection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codec import atomic_write, decode, encode, fields, integer, read_bytes
from .errors import Conflict

WORKSPACE_FILENAME = "workspace.v1.json"


@dataclass(frozen=True, slots=True)
class WorkspaceFormat:
    page_bytes: int

    def __post_init__(self) -> None:
        integer(self.page_bytes, "page_bytes", minimum=1, maximum=1_048_576)
        if self.page_bytes & (self.page_bytes - 1):
            raise Conflict("v1 workspace pages must be a power of two")

    def to_v1(self) -> dict[str, Any]:
        return {"version": 1, "model_weight_unit": "pages", "page_bytes": self.page_bytes}

    @classmethod
    def read(cls, root: Path) -> "WorkspaceFormat | None":
        marker = root / WORKSPACE_FILENAME
        if not marker.exists():
            return None
        document = fields(
            decode(read_bytes(marker, maximum=4096)),
            ("version", "model_weight_unit", "page_bytes"),
            "workspace format",
        )
        integer(document["version"], "workspace version", minimum=1, maximum=1)
        if document["model_weight_unit"] != "pages":
            raise Conflict("v1 workspace must store model weight allocations in pages")
        return cls(document["page_bytes"])

    @classmethod
    def ensure(cls, root: Path, configured_page_bytes: int) -> "WorkspaceFormat":
        requested = cls(configured_page_bytes)
        stored = cls.read(root)
        if stored is None:
            atomic_write(root / WORKSPACE_FILENAME, encode(requested.to_v1()) + b"\n")
            return requested
        if stored != requested:
            raise Conflict(
                f"workspace uses {stored.page_bytes}-byte pages, configured {requested.page_bytes}",
            )
        return stored
