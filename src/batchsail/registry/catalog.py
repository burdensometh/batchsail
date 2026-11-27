"""Catalog generation and explicit pins across model activation."""

from collections import Counter
from dataclasses import dataclass
from threading import RLock
from typing import Any

from ..errors import Conflict, ErrorCode, Invalid, SailError
from .model import InstalledModel, ModelKey, ModelRevision
from .store import ModelStore


@dataclass(frozen=True, slots=True)
class CatalogSwitch:
    generation: int
    active: dict[str, ModelKey]

    def to_json(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "active": {name: key.to_json() for name, key in sorted(self.active.items())},
        }


@dataclass(frozen=True, slots=True)
class RevisionView:
    model: InstalledModel
    active: bool
    pins: int

    @property
    def draining(self) -> bool:
        return not self.active and self.pins > 0


@dataclass(slots=True)
class RevisionPin:
    model: InstalledModel
    token: int


class Catalog:
    def __init__(self, store: ModelStore):
        self._lock = RLock()
        self._store = store
        self._models = store.discover()
        self._generation, self._active = store.read_catalog()
        for key in self._active.values():
            if key not in self._models:
                raise Invalid(f"catalog refers to missing revision {key.label}")
        self._holders: dict[int, ModelKey] = {}
        self._pins: Counter[ModelKey] = Counter()
        self._sequence = 0

    def install(self, revision: ModelRevision, weights: bytes) -> InstalledModel:
        with self._lock:
            model = self._store.install_checked(revision, weights)
            self._models[model.key] = model
            return model

    def get(self, key: ModelKey) -> InstalledModel:
        with self._lock:
            try:
                return self._models[key]
            except KeyError as exc:
                raise SailError(ErrorCode.UNKNOWN_MODEL, f"unknown revision {key.label}") from exc

    def activate(self, key: ModelKey, *, expected_generation: int | None = None) -> int:
        with self._lock:
            self.get(key)
            if expected_generation is not None and self._generation != expected_generation:
                raise Conflict(f"catalog generation is {self._generation}, expected {expected_generation}")
            if self._active.get(key.name) == key:
                return self._generation
            updated = {**self._active, key.name: key}
            generation = self._generation + 1
            self._store.write_catalog(generation, updated)
            self._active = updated
            self._generation = generation
            return generation

    def switch_many(
        self,
        targets: tuple[ModelKey, ...],
        *,
        expected_generation: int | None = None,
    ) -> CatalogSwitch:
        """Atomically move every named model to its target revision in one generation.

        Verification of every target and the expected generation completes before any
        catalog state is touched, so a failed switch leaves all active revisions and
        the generation untouched. A successful switch advances the generation by at
        most one even when several models change.
        """
        if not targets:
            raise Invalid("a catalog switch requires at least one target revision")
        names = [key.name for key in targets]
        if len(set(names)) != len(names):
            raise Invalid("a catalog switch can name each model at most once")
        with self._lock:
            if expected_generation is not None and self._generation != expected_generation:
                raise Conflict(
                    f"catalog generation is {self._generation}, expected {expected_generation}"
                )
            updated = dict(self._active)
            for key in targets:
                self.get(key)
                updated[key.name] = key
            if updated == self._active:
                return CatalogSwitch(self._generation, dict(self._active))
            generation = self._generation + 1
            self._store.write_catalog(generation, updated)
            self._active = updated
            self._generation = generation
            return CatalogSwitch(generation, dict(updated))

    def active(self, name: str) -> InstalledModel:
        with self._lock:
            key = self._active.get(name)
            if key is None:
                raise SailError(ErrorCode.UNKNOWN_MODEL, f"model {name} has no active revision")
            return self._models[key]

    def pin(self, name: str) -> RevisionPin:
        with self._lock:
            model = self.active(name)
            key = model.key
            self._sequence += 1
            self._holders[self._sequence] = key
            self._pins[key] += 1
            return RevisionPin(model, self._sequence)

    def release(self, pin: RevisionPin) -> None:
        with self._lock:
            key = self._holders.get(pin.token)
            if key != pin.model.key:
                raise RuntimeError("revision pin has already been released or belongs to another model")
            del self._holders[pin.token]
            self._pins[key] -= 1
            if self._pins[key] == 0:
                del self._pins[key]

    def draining(self) -> frozenset[ModelKey]:
        with self._lock:
            return frozenset(key for key in self._pins if self._active.get(key.name) != key)

    def protected_residencies(self) -> frozenset[ModelKey]:
        with self._lock:
            return frozenset(self._active.values()).union(self._pins)

    def snapshot(self) -> tuple[int, tuple[RevisionView, ...]]:
        with self._lock:
            models = tuple(
                RevisionView(model, self._active.get(key.name) == key, self._pins[key])
                for key, model in sorted(self._models.items())
            )
            return self._generation, models
