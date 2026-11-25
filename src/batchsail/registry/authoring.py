"""Convert a numeric model definition into a reproducible v1 bundle."""

from pathlib import Path
from typing import Any

from ..codec import decode, digest, fields, integer, read_bytes
from ..errors import Invalid
from ..tensor import TensorSpec, float32
from .model import MAX_MODEL_FILE, ModelKey, ModelRevision, WeightSet
from .store import ModelStore


def parse_definition(document: Any) -> tuple[ModelRevision, WeightSet]:
    definition = fields(
        document,
        (
            "model", "revision", "operator", "input", "output", "weight_pages",
            "taps", "coefficients", "bias",
        ),
        "model definition",
    )
    input_spec = TensorSpec.from_v1(definition["input"])
    output_spec = TensorSpec.from_v1(definition["output"])
    taps = integer(definition["taps"], "taps", minimum=1, maximum=32)
    count = taps * input_spec.columns * output_spec.columns
    coefficients = definition["coefficients"]
    bias = definition["bias"]
    if not isinstance(coefficients, list) or len(coefficients) != count:
        raise Invalid(f"expected {count} model coefficients")
    if not isinstance(bias, list) or len(bias) != output_spec.columns:
        raise Invalid(f"expected {output_spec.columns} model biases")
    weights = WeightSet(tuple(float32(item) for item in coefficients), tuple(float32(item) for item in bias))
    revision = ModelRevision(
        ModelKey(definition["model"], definition["revision"]),
        definition["operator"], input_spec, output_spec,
        digest(weights.encode_v1()), definition["weight_pages"], taps,
    )
    return revision, weights


def pack_definition(source: Path, bundle: Path) -> ModelRevision:
    revision, weights = parse_definition(decode(read_bytes(source, maximum=MAX_MODEL_FILE)))
    ModelStore.write_bundle(bundle, revision, weights)
    return revision
