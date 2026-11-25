"""Filesystem-backed immutable models and atomic active revision selection."""

from pathlib import Path
import shutil
import tempfile
from typing import Any

from ..codec import atomic_write, decode, encode, fields, identifier, integer, read_bytes, sync_directory
from ..errors import Conflict, Invalid
from .model import (
    InstalledModel, MANIFEST_FILENAME, MAX_MODEL_FILE, ModelKey, ModelRevision,
    WEIGHT_FILENAME, WeightSet,
)

CATALOG_FILENAME = "catalog.v1.json"


class ModelStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.revisions = self.root / "revisions"

    def _entry(self, key: ModelKey) -> Path:
        return self.revisions / key.name / key.revision

    @staticmethod
    def read_bundle(bundle: Path) -> tuple[ModelRevision, bytes]:
        manifest = read_bytes(bundle / MANIFEST_FILENAME, maximum=65_536)
        revision = ModelRevision.from_v1(decode(manifest))
        raw = read_bytes(bundle / WEIGHT_FILENAME, maximum=MAX_MODEL_FILE)
        WeightSet.from_v1(raw, revision)
        return revision, raw

    @staticmethod
    def write_bundle(bundle: Path, revision: ModelRevision, weights: WeightSet) -> None:
        raw = weights.encode_v1()
        WeightSet.from_v1(raw, revision)
        if bundle.exists() and any(bundle.iterdir()):
            raise Conflict(f"bundle directory is not empty: {bundle}")
        ModelStore._write_new_bundle(bundle, revision, raw)

    def install(self, bundle: Path) -> InstalledModel:
        revision, weights = self.read_bundle(bundle)
        return self.install_checked(revision, weights)

    def install_checked(self, revision: ModelRevision, weights: bytes) -> InstalledModel:
        target = self._entry(revision.key)
        if target.exists():
            existing, raw = self.read_bundle(target)
            if existing != revision or raw != weights:
                raise Conflict(f"revision {revision.key.label} already has different content")
            return InstalledModel(revision, target / WEIGHT_FILENAME)
        self._write_new_bundle(target, revision, weights)
        sync_directory(self.revisions)
        sync_directory(self.root)
        return InstalledModel(revision, target / WEIGHT_FILENAME)

    @staticmethod
    def _write_new_bundle(target: Path, revision: ModelRevision, weights: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".install-", dir=target.parent))
        try:
            atomic_write(temporary / WEIGHT_FILENAME, weights)
            atomic_write(temporary / MANIFEST_FILENAME, encode(revision.to_v1()) + b"\n")
            temporary.rename(target)
            sync_directory(target.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def discover(self) -> dict[ModelKey, InstalledModel]:
        installed: dict[ModelKey, InstalledModel] = {}
        if not self.revisions.exists():
            return installed
        for model in sorted(self.revisions.iterdir()):
            if model.name.startswith(".") or not model.is_dir():
                continue
            for directory in sorted(model.iterdir()):
                if directory.name.startswith(".") or not directory.is_dir():
                    continue
                revision, _ = self.read_bundle(directory)
                expected = ModelKey(model.name, directory.name)
                if revision.key != expected:
                    raise Invalid(f"bundle at {directory} declares revision {revision.key.label}")
                installed[expected] = InstalledModel(revision, directory / WEIGHT_FILENAME)
        return installed

    def read_catalog(self) -> tuple[int, dict[str, ModelKey]]:
        path = self.root / CATALOG_FILENAME
        if not path.exists():
            return 0, {}
        value = fields(
            decode(read_bytes(path, maximum=1_048_576)),
            ("version", "generation", "active"),
            "model catalog",
        )
        integer(value["version"], "catalog version", minimum=1, maximum=1)
        generation = integer(value["generation"], "catalog generation")
        active = value["active"]
        if not isinstance(active, dict):
            raise Invalid("catalog active must map model names to revisions")
        return generation, {name: ModelKey(name, revision) for name, revision in active.items()}

    def write_catalog(self, generation: int, active: dict[str, ModelKey]) -> None:
        integer(generation, "catalog generation", minimum=1)
        payload: dict[str, Any] = {
            "version": 1, "generation": generation,
            "active": {name: key.revision for name, key in active.items()},
        }
        for name, key in active.items():
            if identifier(name, "active model") != key.name:
                raise Invalid("active catalog keys must agree with the model name")
        atomic_write(self.root / CATALOG_FILENAME, encode(payload) + b"\n")
