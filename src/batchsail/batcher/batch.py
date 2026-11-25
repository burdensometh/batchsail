"""Batch shape, attempt identities, and dense packed tensor layout."""

from array import array
from dataclasses import dataclass
from typing import Iterable

from ..errors import Invalid
from ..registry.model import InstalledModel, ModelKey
from ..request.lifecycle import Ticket
from ..request.model import RequestKey
from ..tensor import F32_BYTES, MAX_ELEMENTS, Tensor


@dataclass(frozen=True, slots=True, order=True)
class ShapeKey:
    revision: ModelKey
    rows: int
    columns: int


@dataclass(frozen=True, slots=True)
class BatchSlot:
    index: int
    key: RequestKey
    sequence: int
    attempt: int
    rows: int
    row_offset: int


@dataclass(frozen=True, slots=True)
class Batch:
    id: str
    model: InstalledModel
    shape: ShapeKey
    slots: tuple[BatchSlot, ...]
    inputs: tuple[Tensor, ...]

    @classmethod
    def from_tickets(cls, batch_id: str, shape: ShapeKey, tickets: Iterable[Ticket]) -> "Batch":
        items = tuple(tickets)
        if not items or len({ticket.key for ticket in items}) != len(items):
            raise Invalid("batch requires one or more distinct requests")
        model = items[0].pin.model
        if model.key != shape.revision:
            raise Invalid("batch shape does not refer to its installed revision")
        inputs: list[Tensor] = []
        slots: list[BatchSlot] = []
        for index, ticket in enumerate(items):
            tensor = ticket.request.tensor
            if ticket.pin.model != model or tensor.columns != shape.columns or tensor.rows > shape.rows:
                raise Invalid("attempt does not match batch revision or shape")
            slots.append(BatchSlot(
                index, ticket.key, ticket.sequence, ticket.attempt + 1, tensor.rows, index * shape.rows,
            ))
            inputs.append(tensor)
        return cls(batch_id, model, shape, tuple(slots), tuple(inputs))

    @property
    def packed_rows(self) -> int:
        return len(self.slots) * self.shape.rows


@dataclass(slots=True)
class PackedBatch:
    rows_per_slot: int
    input_columns: int
    row_counts: tuple[int, ...]
    values: array

    @classmethod
    def pack(cls, batch: Batch) -> "PackedBatch":
        columns = batch.shape.columns
        elements = batch.packed_rows * columns
        if elements > MAX_ELEMENTS * 64:
            raise Invalid("packed input exceeds the representable limit")
        values = array("f", [0.0]) * elements
        if values.itemsize != F32_BYTES:
            raise RuntimeError("CPU backend requires a 32-bit float array")
        for slot, tensor in zip(batch.slots, batch.inputs, strict=True):
            start = slot.row_offset * columns
            values[start : start + len(tensor.values)] = array("f", tensor.values)
        return cls(batch.shape.rows, columns, tuple(slot.rows for slot in batch.slots), values)

    def check(self, columns: int, max_rows: int) -> None:
        if self.input_columns != columns or not 0 < self.rows_per_slot <= max_rows:
            raise Invalid("packed batch shape does not match the model")
        if not self.row_counts or any(not 0 < count <= self.rows_per_slot for count in self.row_counts):
            raise Invalid("packed batch has invalid slot rows")
        if len(self.values) != len(self.row_counts) * self.rows_per_slot * columns:
            raise Invalid("packed data does not match declared rows")
