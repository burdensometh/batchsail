"""Reference CPU kernels over packed, padded float32 arrays."""

from array import array
from math import isfinite
from typing import Protocol

from ..batcher.batch import PackedBatch
from ..errors import ErrorCode, SailError
from ..registry.model import ModelRevision, WeightSet
from ..tensor import Tensor

F32_LIMIT = 3.4028234663852886e38


class Kernel(Protocol):
    def run(self, packed: PackedBatch) -> tuple[Tensor, ...]: ...


def _store(array_values: array, index: int, value: float) -> None:
    if not isfinite(value) or abs(value) > F32_LIMIT:
        raise SailError(ErrorCode.EXECUTION, "kernel output overflows float32")
    array_values[index] = value


def _unpack(output: array, counts: tuple[int, ...], stride: int, width: int) -> tuple[Tensor, ...]:
    tensors = []
    for index, rows in enumerate(counts):
        start = index * stride * width
        tensors.append(Tensor((rows, width), tuple(output[start : start + rows * width])))
    return tuple(tensors)


class Affine:
    """Computes y[row, channel] = bias[channel] + dot(x[row], weights[channel])."""

    def __init__(self, revision: ModelRevision, weights: WeightSet):
        self.input_width = revision.input_spec.columns
        self.output_width = revision.output_spec.columns
        self.max_rows = revision.input_spec.max_rows
        self.weights = array("f", weights.coefficients)
        self.bias = array("f", weights.bias)

    def run(self, packed: PackedBatch) -> tuple[Tensor, ...]:
        packed.check(self.input_width, self.max_rows)
        input_width = self.input_width
        output_width = self.output_width
        stride = packed.rows_per_slot
        weights = self.weights
        values = packed.values
        output = array("f", [0.0]) * (len(packed.row_counts) * stride * output_width)
        for slot, rows in enumerate(packed.row_counts):
            slot_row = slot * stride
            for row_index in range(slot_row, slot_row + rows):
                begin = row_index * input_width
                end = row_index * output_width
                for channel in range(output_width):
                    coefficient = channel * input_width
                    value = self.bias[channel]
                    for column in range(input_width):
                        value += values[begin + column] * weights[coefficient + column]
                    _store(output, end + channel, value)
        return _unpack(output, packed.row_counts, stride, output_width)


class CausalFilter:
    """Finite impulse response with independent zero history at each slot boundary."""

    def __init__(self, revision: ModelRevision, weights: WeightSet):
        self.input_width = revision.input_spec.columns
        self.output_width = revision.output_spec.columns
        self.max_rows = revision.input_spec.max_rows
        self.taps = revision.taps
        self.weights = array("f", weights.coefficients)
        self.bias = array("f", weights.bias)

    def run(self, packed: PackedBatch) -> tuple[Tensor, ...]:
        packed.check(self.input_width, self.max_rows)
        width = self.input_width
        channels = self.output_width
        stride = packed.rows_per_slot
        weights = self.weights
        values = packed.values
        output = array("f", [0.0]) * (len(packed.row_counts) * stride * channels)
        for slot, rows in enumerate(packed.row_counts):
            base_row = slot * stride
            for time in range(rows):
                absolute_row = base_row + time
                for channel in range(channels):
                    value = self.bias[channel]
                    for tap in range(min(self.taps, time + 1)):
                        source = (absolute_row - tap) * width
                        coefficient = (tap * channels + channel) * width
                        for column in range(width):
                            value += values[source + column] * weights[coefficient + column]
                    _store(output, absolute_row * channels + channel, value)
        return _unpack(output, packed.row_counts, stride, channels)


def compile_kernel(revision: ModelRevision, weights: WeightSet) -> Kernel:
    if revision.operator == "affine":
        return Affine(revision, weights)
    if revision.operator == "causal_filter":
        return CausalFilter(revision, weights)
    raise SailError(ErrorCode.INVALID, f"no CPU kernel for {revision.operator}")
