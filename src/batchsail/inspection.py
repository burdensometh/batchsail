"""Offline model, workspace-format and historical receipt inspection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RuntimeConfig, WorkspaceGuard
from .device.policy import memory_demand
from .errors import SailError
from .registry.model import InstalledModel, ModelKey
from .registry.store import ModelStore
from .results.inspection import ReceiptAudit, inspect_receipts
from .workspace import WorkspaceFormat


@dataclass(slots=True)
class WorkspaceAudit:
    root: Path
    generation: int | None = None
    format: WorkspaceFormat | None = None
    active: dict[str, ModelKey] = field(default_factory=dict)
    installed: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)
    receipts: ReceiptAudit | None = None

    @property
    def clean(self) -> bool:
        return not self.issues and (self.receipts is None or self.receipts.issue_count == 0)

    def issue(self, kind: str, message: str) -> None:
        self.issues.append({"code": kind, "message": message})

    def to_json(self) -> dict[str, Any]:
        return {
            "workspace": str(self.root),
            "clean": self.clean,
            "generation": self.generation,
            "format": None if self.format is None else self.format.to_v1(),
            "active": [item.to_json() for _, item in sorted(self.active.items())],
            "models": self.installed,
            "issues": self.issues,
            "receipts": None if self.receipts is None else self.receipts.to_json(),
        }


def _inventory(
    audit: WorkspaceAudit, config: RuntimeConfig, models: dict[ModelKey, InstalledModel],
) -> None:
    for key, installed in sorted(models.items()):
        revision = installed.revision
        item: dict[str, Any] = {
            "manifest": revision.to_v1(),
            "active": audit.active.get(key.name) == key,
            "fits_minimum_input": [],
            "fits_maximum_input": [],
        }
        try:
            smallest = config.batching.padded_rows(1, revision.input_spec.max_rows)
            claims = tuple(
                memory_demand(config.memory, revision, padded_rows=rows, slots=1)
                for rows in (smallest, revision.input_spec.max_rows)
            )
            for label, demand in zip(("fits_minimum_input", "fits_maximum_input"), claims, strict=True):
                required = demand.weight_bytes + demand.workspace_bytes
                item[label] = [device.name for device in config.devices if device.capacity_bytes >= required]
            if item["active"] and not item["fits_minimum_input"]:
                audit.issue("active_revision_unrunnable", f"{key.label} cannot fit one slot on the configured devices")
        except SailError as exc:
            audit.issue("invalid_revision_memory", f"{key.label}: {exc}")
        audit.installed.append(item)


def inspect_workspace(root: Path, config: RuntimeConfig) -> WorkspaceAudit:
    with WorkspaceGuard(root) as workspace:
        audit = WorkspaceAudit(workspace.root)
        store = ModelStore(workspace.root / "models")
        try:
            audit.format = WorkspaceFormat.read(workspace.root)
        except SailError as exc:
            audit.issue("invalid_workspace_format", str(exc))
        if audit.format is not None and audit.format.page_bytes != config.memory.page_bytes:
            audit.issue(
                "page_size_mismatch",
                f"workspace uses {audit.format.page_bytes} bytes per page; config uses {config.memory.page_bytes}",
            )
        try:
            audit.generation, audit.active = store.read_catalog()
        except SailError as exc:
            audit.issue("invalid_catalog", str(exc))
        try:
            models = store.discover()
        except SailError as exc:
            audit.issue("invalid_model_bundle", str(exc))
            models = {}
        else:
            for key in audit.active.values():
                if key not in models:
                    audit.issue("active_revision_missing", f"{key.label} is active but its bundle is missing")
            _inventory(audit, config, models)
        audit.receipts = inspect_receipts(workspace.root / "receipts.sqlite", models)
        return audit
