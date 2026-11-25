"""Immutable dense row-major float32 tensors and model shape contracts."""

from dataclasses import dataclass
from math import isfinite, prod
from struct import Struct
from typing import Any

from .codec import fields, finite, integer
from .errors import Invalid

F32_BYTES = 4
MAX_ELEMENTS = 1_048_576
_FLOAT = Struct("<f")


def float32(value: Any) -> float:
    number = finite(value, "tensor value")
    try:
        return _FLOAT.unpack(_FLOAT.pack(number))[0]
    except OverflowError as exc:
        raise Invalid("tensor value does not fit float32") from exc


@dataclass(frozen=True, slots=True)
class Tensor:
    shape: tuple[int, int]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise Invalid("tensor must have two dimensions")
        for dimension in self.shape:
            integer(dimension, "dimension", minimum=1, maximum=MAX_ELEMENTS)
        size = prod(self.shape)
        if size > MAX_ELEMENTS or size != len(self.values):
            raise Invalid("tensor shape does not match its data or exceeds the element limit")
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "values", tuple(float32(value) for value in self.values))

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def columns(self) -> int:
        return self.shape[1]

    @property
    def nbytes(self) -> int:
        return len(self.values) * F32_BYTES

    def row(self, index: int) -> tuple[float, ...]:
        if not 0 <= index < self.rows:
            raise IndexError(index)
        start = index * self.columns
        return self.values[start : start + self.columns]

    def to_json(self) -> dict[str, Any]:
        return {"dtype": "float32", "shape": list(self.shape), "values": list(self.values)}

    @classmethod
    def from_json(cls, value: Any) -> "Tensor":
        tensor = fields(value, ("dtype", "shape", "values"), "tensor")
        if tensor["dtype"] != "float32":
            raise Invalid("v1 tensors use float32")
        if not isinstance(tensor["shape"], list) or not isinstance(tensor["values"], list):
            raise Invalid("tensor shape and values must be arrays")
        return cls(tuple(tensor["shape"]), tuple(tensor["values"]))


@dataclass(frozen=True, slots=True)
class TensorSpec:
    columns: int
    max_rows: int

    def __post_init__(self) -> None:
        integer(self.columns, "tensor width", minimum=1, maximum=4096)
        integer(self.max_rows, "maximum rows", minimum=1, maximum=65536)
        if self.columns * self.max_rows > MAX_ELEMENTS:
            raise Invalid("tensor specification exceeds the element limit")

    def check(self, tensor: Tensor, label: str) -> None:
        if tensor.columns != self.columns or tensor.rows > self.max_rows:
            raise Invalid(f"{label} expects 1 to {self.max_rows} rows of width {self.columns}")

    def to_v1(self) -> dict[str, int]:
        return {"columns": self.columns, "max_rows": self.max_rows}

    @classmethod
    def from_v1(cls, value: Any) -> "TensorSpec":
        spec = fields(value, ("columns", "max_rows"), "tensor specification")
        return cls(spec["columns"], spec["max_rows"])


def equivalent(first: Tensor, second: Tensor, *, tolerance: float = 1e-6) -> bool:
    if first.shape != second.shape:
        return False
    return all(
        isfinite(left) and isfinite(right) and abs(left - right) <= tolerance * max(1, abs(right))
        for left, right in zip(first.values, second.values, strict=True)
    )
