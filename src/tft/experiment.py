"""Typed experiment orchestration for the TFT numerical laboratory.

The implemented experiment is deliberately narrow: it compares a real,
symmetric rank-2 matrix with the same matrix expressed in a seeded
proper-orthogonal basis, then maps the preserved eigenvalue spectrum to an
illustrative stereo sonification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .resonance import (
    invariants,
    map_to_audio,
    rng,
    rotate_rank2,
    rotation_matrix,
    spd_matrix,
)


ALGORITHM_ID = "tft.matrix-invariance.v1"
RECIPE_SCHEMA = "tft.run.recipe.v1"
OBSERVATIONS_SCHEMA = "tft.run.observations.v1"
MAX_SAMPLE_COUNT = 6_000_000


def _require_int(name: str, value: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if not minimum <= converted <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return converted


def _require_float(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = False,
) -> float:
    if isinstance(value, (bool, complex, np.complexfloating)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise TypeError(f"{name} must be a real number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        valid = converted >= minimum if minimum_inclusive else converted > minimum
        if not valid:
            relation = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be no greater than {maximum}")
    return converted


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Identity-bearing inputs for one TFT matrix experiment."""

    seed: int = 1337
    dimension: int = 3
    sample_rate: int = 48_000
    duration_s: float = 2.0
    fmin_hz: float = 220.0
    fmax_hz: float = 880.0
    target_peak: float = 0.92
    fade_s: float = 0.01
    regularization: float = 1.0e-6
    tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        seed = _require_int("seed", self.seed, 0, (1 << 64) - 1)
        dimension = _require_int("dimension", self.dimension, 2, 32)
        sample_rate = _require_int("sample_rate", self.sample_rate, 8_000, 192_000)
        duration_s = _require_float("duration_s", self.duration_s, minimum=0.0, maximum=60.0)
        fmin_hz = _require_float("fmin_hz", self.fmin_hz, minimum=0.0)
        fmax_hz = _require_float("fmax_hz", self.fmax_hz, minimum=fmin_hz)
        target_peak = _require_float("target_peak", self.target_peak, minimum=0.0, maximum=1.0)
        fade_s = _require_float(
            "fade_s", self.fade_s, minimum=0.0, maximum=duration_s / 2.0, minimum_inclusive=True
        )
        regularization = _require_float("regularization", self.regularization, minimum=0.0)
        tolerance = _require_float("tolerance", self.tolerance, minimum=0.0, maximum=1.0e-3)

        if fmax_hz >= sample_rate / 2.0:
            raise ValueError("fmax_hz must be below the Nyquist frequency")
        sample_count = int(round(duration_s * sample_rate))
        if sample_count < 2:
            raise ValueError("duration_s and sample_rate must produce at least two samples")
        if sample_count > MAX_SAMPLE_COUNT:
            raise ValueError("duration_s and sample_rate produce too many samples")
        if sample_count == 2 and int(round(fade_s * sample_rate)) > 0:
            raise ValueError("a two-sample render cannot contain a fade envelope")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "fmin_hz", fmin_hz)
        object.__setattr__(self, "fmax_hz", fmax_hz)
        object.__setattr__(self, "target_peak", target_peak)
        object.__setattr__(self, "fade_s", fade_s)
        object.__setattr__(self, "regularization", regularization)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def sample_count(self) -> int:
        return int(round(self.duration_s * self.sample_rate))

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _residual(reference: float, candidate: float) -> dict[str, float]:
    absolute = abs(candidate - reference)
    scale = max(abs(reference), abs(candidate), np.finfo(float).tiny)
    return {"absolute": float(absolute), "relative": float(absolute / scale)}


def _vector_residual(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    absolute = float(np.max(np.abs(delta), initial=0.0))
    scale = max(
        float(np.max(np.abs(reference), initial=0.0)),
        float(np.max(np.abs(candidate), initial=0.0)),
        np.finfo(float).tiny,
    )
    return {"max_absolute": absolute, "max_relative": float(absolute / scale)}


def _within_combined_tolerance(
    absolute_residual: float,
    reference_scale: float,
    tolerance: float,
) -> bool:
    """Use one recorded value as both absolute and relative tolerance."""

    return absolute_residual <= tolerance + tolerance * max(reference_scale, 0.0)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Numerical arrays and observations produced by :func:`run_experiment`."""

    config: ExperimentConfig
    tensor: np.ndarray
    rotation: np.ndarray
    rotated_tensor: np.ndarray
    initial_invariants: Mapping[str, Any]
    rotated_invariants: Mapping[str, Any]
    frequencies_hz: np.ndarray
    stereo: np.ndarray

    def observations(self) -> dict[str, Any]:
        initial = self.initial_invariants
        rotated = self.rotated_invariants
        fro_residual = _residual(float(initial["fro"]), float(rotated["fro"]))
        trace_residual = _residual(float(initial["trace"]), float(rotated["trace"]))
        determinant_residual = _residual(
            float(initial["determinant"]), float(rotated["determinant"])
        )
        eigenvalue_residual = _vector_residual(initial["eigvals"], rotated["eigvals"])

        identity = np.eye(self.config.dimension, dtype=float)
        orthogonality_residual = float(np.linalg.norm(self.rotation.T @ self.rotation - identity, "fro"))
        rotation_determinant = float(np.linalg.det(self.rotation))
        tensor_reconstruction_residual = float(
            np.max(
                np.abs(self.rotated_tensor - self.rotation @ self.tensor @ self.rotation.T),
                initial=0.0,
            )
        )

        left = self.stereo[:, 0]
        right = self.stereo[:, 1]
        left_centered = left - float(np.mean(left))
        right_centered = right - float(np.mean(right))
        denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        correlation = (
            float(np.dot(left_centered, right_centered) / denominator) if denominator > 0.0 else 0.0
        )

        tolerance = self.config.tolerance
        eigenvalue_scale = max(
            float(np.max(np.abs(initial["eigvals"]), initial=0.0)),
            float(np.max(np.abs(rotated["eigvals"]), initial=0.0)),
        )
        tensor_scale = max(float(np.max(np.abs(self.rotated_tensor), initial=0.0)), 1.0)
        passed = bool(
            _within_combined_tolerance(
                fro_residual["absolute"],
                max(abs(float(initial["fro"])), abs(float(rotated["fro"]))),
                tolerance,
            )
            and _within_combined_tolerance(
                trace_residual["absolute"],
                max(abs(float(initial["trace"])), abs(float(rotated["trace"]))),
                tolerance,
            )
            and _within_combined_tolerance(
                eigenvalue_residual["max_absolute"],
                eigenvalue_scale,
                tolerance,
            )
            and orthogonality_residual <= tolerance * max(1.0, self.config.dimension)
            and abs(rotation_determinant - 1.0) <= tolerance * max(1.0, self.config.dimension)
            and tensor_reconstruction_residual <= tolerance + tolerance * tensor_scale
        )

        return {
            "schema": OBSERVATIONS_SCHEMA,
            "algorithm": ALGORITHM_ID,
            "status": "pass" if passed else "fail",
            "tolerance": tolerance,
            "tolerance_rule": "absolute_residual <= tolerance + tolerance * reference_scale",
            "matrix": {
                "dimension": self.config.dimension,
                "initial": {
                    "frobenius_norm": float(initial["fro"]),
                    "trace": float(initial["trace"]),
                    "determinant": float(initial["determinant"]),
                    "eigenvalues": np.asarray(initial["eigvals"], dtype=float).tolist(),
                },
                "rotated": {
                    "frobenius_norm": float(rotated["fro"]),
                    "trace": float(rotated["trace"]),
                    "determinant": float(rotated["determinant"]),
                    "eigenvalues": np.asarray(rotated["eigvals"], dtype=float).tolist(),
                },
                "residuals": {
                    "frobenius_norm": fro_residual,
                    "trace": trace_residual,
                    "determinant": determinant_residual,
                    "eigenvalues": eigenvalue_residual,
                    "orthogonality_frobenius": orthogonality_residual,
                    "rotation_determinant_from_one": abs(rotation_determinant - 1.0),
                    "rotated_tensor_max_absolute": tensor_reconstruction_residual,
                },
                "rotation_determinant": rotation_determinant,
            },
            "sonification": {
                "mapping": "linear min-max eigenvalue mapping",
                "frequencies_hz": self.frequencies_hz.astype(float).tolist(),
                "sample_rate": self.config.sample_rate,
                "sample_count": int(self.stereo.shape[0]),
                "channels": 2,
                "peak": float(np.max(np.abs(self.stereo), initial=0.0)),
                "rms": [
                    float(np.sqrt(np.mean(self.stereo[:, channel] ** 2))) for channel in range(2)
                ],
                "stereo_correlation": correlation,
                "phase_convention": (
                    "right-channel sine partner for each left-channel cosine component, "
                    "followed by a shared fade envelope"
                ),
            },
            "claim_boundary": (
                "This run demonstrates numerical invariants of a finite real symmetric matrix "
                "under a proper-orthogonal basis change and an illustrative pitch mapping. "
                "It does not solve the proposed TFT field equation or validate a physical theory."
            ),
        }


def synthesize_quadrature(
    frequencies_hz: np.ndarray,
    *,
    sample_rate: int,
    sample_count: int,
    seed: int,
    target_peak: float,
    fade_s: float,
) -> np.ndarray:
    """Synthesize deterministic cosine/sine partners as finite stereo audio."""

    validated_sample_rate = _require_int("sample_rate", sample_rate, 8_000, 192_000)
    validated_sample_count = _require_int("sample_count", sample_count, 2, MAX_SAMPLE_COUNT)
    validated_peak = _require_float("target_peak", target_peak, minimum=0.0, maximum=1.0)
    max_fade = validated_sample_count / (2.0 * validated_sample_rate)
    validated_fade = _require_float(
        "fade_s", fade_s, minimum=0.0, maximum=max_fade, minimum_inclusive=True
    )
    if validated_sample_count == 2 and int(round(validated_fade * validated_sample_rate)) > 0:
        raise ValueError("a two-sample render cannot contain a fade envelope")
    raw_frequencies = np.asarray(frequencies_hz)
    if np.iscomplexobj(raw_frequencies) or raw_frequencies.dtype.kind not in "iuf":
        raise TypeError("frequencies_hz must contain real numeric values")
    frequencies = np.asarray(raw_frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain positive finite values")
    if np.any(frequencies >= validated_sample_rate / 2.0):
        raise ValueError("frequencies_hz must be below the Nyquist frequency")
    phase_rng = rng(seed)
    phases = phase_rng.uniform(0.0, 2.0 * np.pi, size=frequencies.size)
    stereo = np.empty((validated_sample_count, 2), dtype=np.float64)
    fade_frames = min(
        int(round(validated_fade * validated_sample_rate)), validated_sample_count // 2
    )
    block_size = 262_144
    for start in range(0, validated_sample_count, block_size):
        stop = min(start + block_size, validated_sample_count)
        indices = np.arange(start, stop, dtype=np.float64)
        time = indices / float(validated_sample_rate)
        left = np.zeros(stop - start, dtype=np.float64)
        right = np.zeros(stop - start, dtype=np.float64)
        for frequency, phase in zip(frequencies, phases, strict=True):
            angle = 2.0 * np.pi * frequency * time + phase
            left += np.cos(angle)
            right += np.sin(angle)
        left /= frequencies.size
        right /= frequencies.size

        if fade_frames > 0:
            sample_indices = np.arange(start, stop, dtype=np.int64)
            envelope = np.ones(stop - start, dtype=np.float64)
            fade_in = sample_indices < fade_frames
            fade_out = sample_indices >= validated_sample_count - fade_frames
            if fade_frames == 1:
                envelope[fade_in | fade_out] = 0.0
            else:
                envelope[fade_in] = np.sin(
                    (sample_indices[fade_in] / (fade_frames - 1)) * (np.pi / 2.0)
                ) ** 2
                distance_from_end = validated_sample_count - 1 - sample_indices[fade_out]
                envelope[fade_out] = np.sin(
                    (distance_from_end / (fade_frames - 1)) * (np.pi / 2.0)
                ) ** 2
            left *= envelope
            right *= envelope
        stereo[start:stop, 0] = left
        stereo[start:stop, 1] = right

    peak = float(np.max(np.abs(stereo), initial=0.0))
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("synthesis produced an invalid zero-energy signal")
    stereo *= validated_peak / peak
    return stereo


def run_experiment(
    config: ExperimentConfig,
    *,
    tensor: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
) -> ExperimentResult:
    """Run a generated or caller-supplied finite matrix experiment."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")

    if tensor is None:
        tensor_array = spd_matrix(
            dim=config.dimension,
            seed=config.seed,
            regularization=config.regularization,
        )
    else:
        tensor_array = np.asarray(tensor)
        # Validation is intentionally delegated to invariants so the public
        # input contract has one authoritative implementation.
        invariants(tensor_array)
        tensor_array = np.array(tensor_array, dtype=np.float64, copy=True, order="C")
        if tensor_array.shape != (config.dimension, config.dimension):
            raise ValueError("tensor shape does not match config.dimension")

    if rotation is None:
        rotation_seed = (config.seed + 1) & ((1 << 64) - 1)
        rotation_array = rotation_matrix(dim=config.dimension, seed=rotation_seed)
    else:
        # Preserve the caller's dtype until rotate_rank2 validates it; casting
        # first would silently discard the imaginary part of a complex input.
        rotation_array = np.asarray(rotation)

    rotated_tensor = rotate_rank2(tensor_array, rotation_array)
    initial_invariants = invariants(tensor_array)
    rotated_invariants = invariants(rotated_tensor)
    frequencies_hz = map_to_audio(
        initial_invariants["eigvals"],
        fmin=config.fmin_hz,
        fmax=config.fmax_hz,
    )
    phase_seed = config.seed ^ 0x9E3779B97F4A7C15
    stereo = synthesize_quadrature(
        frequencies_hz,
        sample_rate=config.sample_rate,
        sample_count=config.sample_count,
        seed=phase_seed,
        target_peak=config.target_peak,
        fade_s=config.fade_s,
    )

    def readonly_array(value: np.ndarray) -> np.ndarray:
        frozen = np.array(value, dtype=np.float64, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    initial_values = dict(initial_invariants)
    initial_values["eigvals"] = readonly_array(initial_invariants["eigvals"])
    rotated_values = dict(rotated_invariants)
    rotated_values["eigvals"] = readonly_array(rotated_invariants["eigvals"])
    frozen_initial = MappingProxyType(initial_values)
    frozen_rotated = MappingProxyType(rotated_values)

    return ExperimentResult(
        config=config,
        tensor=readonly_array(tensor_array),
        rotation=readonly_array(rotation_array),
        rotated_tensor=readonly_array(rotated_tensor),
        initial_invariants=frozen_initial,
        rotated_invariants=frozen_rotated,
        frequencies_hz=readonly_array(frequencies_hz),
        stereo=readonly_array(stereo),
    )
