"""Version-one model descriptors, weight files, and revision identity."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..codec import decode, digest, digest_text, encode, fields, identifier, integer
from ..errors import Invalid
from ..tensor import F32_BYTES, TensorSpec, float32

MAX_MODEL_FILE = 8 * 1024 * 1024
WEIGHT_FILENAME = "weights.v1.json"
MANIFEST_FILENAME = "manifest.v1.json"


@dataclass(frozen=True, slots=True, order=True)
class ModelKey:
    name: str
    revision: str

    def __post_init__(self) -> None:
        identifier(self.name, "model")
        identifier(self.revision, "revision")

    @property
    def label(self) -> str:
        return f"{self.name}@{self.revision}"

    def to_json(self) -> dict[str, str]:
        return {"model": self.name, "revision": self.revision}

    @classmethod
    def from_json(cls, value: Any) -> "ModelKey":
        key = fields(value, ("model", "revision"), "model key")
        return cls(key["model"], key["revision"])


@dataclass(frozen=True, slots=True)
class ModelRevision:
    key: ModelKey
    operator: str
    input_spec: TensorSpec
    output_spec: TensorSpec
    weight_sha256: str
    pages: int
    taps: int

    def __post_init__(self) -> None:
        if self.operator not in ("affine", "causal_filter"):
            raise Invalid(f"unsupported v1 model operator {self.operator!r}")
        if self.input_spec.max_rows != self.output_spec.max_rows:
            raise Invalid("input and output must share maximum sequence length")
        if self.operator == "affine":
            integer(self.taps, "affine taps", minimum=1, maximum=1)
        else:
            integer(self.taps, "causal filter taps", minimum=2, maximum=32)
        digest_text(self.weight_sha256, "weight digest")
        integer(self.pages, "weight allocation pages", minimum=1, maximum=1_048_576)
        if self.expected_coefficients > 1_048_576:
            raise Invalid("model has too many coefficients")

    @property
    def expected_coefficients(self) -> int:
        return self.taps * self.input_spec.columns * self.output_spec.columns

    @property
    def weight_nbytes(self) -> int:
        return F32_BYTES * (self.expected_coefficients + self.output_spec.columns)

    @property
    def row_workspace_bytes(self) -> int:
        return F32_BYTES * (self.input_spec.columns + self.output_spec.columns)

    @classmethod
    def from_v1(cls, value: Any) -> "ModelRevision":
        manifest = fields(
            value,
            (
                "version", "model", "revision", "operator", "input", "output",
                "weight_file", "weights_sha256", "weight_pages", "taps",
            ),
            "model manifest",
        )
        integer(manifest["version"], "manifest version", minimum=1, maximum=1)
        if manifest["weight_file"] != WEIGHT_FILENAME:
            raise Invalid(f"v1 weight_file must be {WEIGHT_FILENAME}")
        return cls(
            ModelKey(manifest["model"], manifest["revision"]),
            manifest["operator"],
            TensorSpec.from_v1(manifest["input"]),
            TensorSpec.from_v1(manifest["output"]),
            manifest["weights_sha256"],
            manifest["weight_pages"],
            manifest["taps"],
        )

    def to_v1(self) -> dict[str, Any]:
        return {
            "version": 1,
            **self.key.to_json(),
            "operator": self.operator,
            "input": self.input_spec.to_v1(),
            "output": self.output_spec.to_v1(),
            "weight_file": WEIGHT_FILENAME,
            "weights_sha256": self.weight_sha256,
            "weight_pages": self.pages,
            "taps": self.taps,
        }


@dataclass(frozen=True, slots=True)
class WeightSet:
    coefficients: tuple[float, ...]
    bias: tuple[float, ...]

    @classmethod
    def from_v1(cls, raw: bytes, revision: ModelRevision) -> "WeightSet":
        if digest(raw) != revision.weight_sha256:
            raise Invalid(f"weights for {revision.key.label} do not match their SHA-256")
        value = fields(decode(raw), ("version", "coefficients", "bias"), "weights")
        integer(value["version"], "weights version", minimum=1, maximum=1)
        coefficients = value["coefficients"]
        bias = value["bias"]
        if not isinstance(coefficients, list) or len(coefficients) != revision.expected_coefficients:
            raise Invalid(f"bad coefficient count for {revision.key.label}")
        if not isinstance(bias, list) or len(bias) != revision.output_spec.columns:
            raise Invalid(f"bad bias count for {revision.key.label}")
        return cls(tuple(float32(item) for item in coefficients), tuple(float32(item) for item in bias))

    def encode_v1(self) -> bytes:
        return encode({"version": 1, "coefficients": self.coefficients, "bias": self.bias}) + b"\n"


@dataclass(frozen=True, slots=True)
class InstalledModel:
    revision: ModelRevision
    weight_path: Path

    @property
    def key(self) -> ModelKey:
        return self.revision.key
