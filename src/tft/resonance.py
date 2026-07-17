"""Validated numerical primitives for TFT Lab.

These functions operate on finite real matrices and signals. They do not
implement a tensor field, covariant derivative, or field-equation solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np


UINT64_MAX = (1 << 64) - 1
DEFAULT_SYMMETRY_TOLERANCE = 1.0e-12
DEFAULT_ORTHOGONALITY_TOLERANCE = 1.0e-10


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    converted = int(seed)
    if not 0 <= converted <= UINT64_MAX:
        raise ValueError(f"seed must be between 0 and {UINT64_MAX}")
    return converted


def _validated_dimension(dim: int) -> int:
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
        raise TypeError("dim must be an integer")
    converted = int(dim)
    if not 2 <= converted <= 32:
        raise ValueError("dim must be between 2 and 32")
    return converted


def _finite_real_array(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _square_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = _finite_real_array(value, name=name)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    return matrix


def _symmetric_matrix(
    value: Any,
    *,
    name: str,
    symmetry_tolerance: float,
) -> np.ndarray:
    matrix = _square_matrix(value, name=name)
    if not np.isfinite(symmetry_tolerance) or symmetry_tolerance < 0.0:
        raise ValueError("symmetry_tolerance must be a finite non-negative number")
    scale = float(np.max(np.abs(matrix), initial=0.0))
    if scale == 0.0:
        relative_asymmetry = 0.0
    else:
        scaled = matrix / scale
        relative_asymmetry = float(np.max(np.abs(scaled - scaled.T), initial=0.0))
    if relative_asymmetry > symmetry_tolerance:
        raise ValueError(
            f"{name} must be symmetric within tolerance "
            f"(relative asymmetry {relative_asymmetry:.3e})"
        )
    # Remove only round-off-sized antisymmetry after the explicit check.
    return matrix / 2.0 + matrix.T / 2.0


def _normalized_axis(axis: int, ndim: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    converted = int(axis)
    if converted < 0:
        converted += ndim
    if not 0 <= converted < ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array with {ndim} dimensions")
    return converted


def _stable_rms(value: np.ndarray, axis: int, *, keepdims: bool) -> np.ndarray:
    scale = np.max(np.abs(value), axis=axis, keepdims=True)
    divisor = np.where(scale > 0.0, scale, 1.0)
    normalized = value / divisor
    rms = scale * np.sqrt(np.mean(normalized * normalized, axis=axis, keepdims=True))
    if not np.all(np.isfinite(rms)):
        raise ValueError("signal RMS exceeds the finite float64 domain")
    return rms if keepdims else np.squeeze(rms, axis=axis)


def rng(seed: int = 1337) -> np.random.Generator:
    """Create the package's seeded NumPy generator.

    ``None`` is intentionally rejected: TFT run recipes must not silently
    switch from seeded to entropy-backed generation.
    """

    return np.random.default_rng(_validated_seed(seed))


def rotation_matrix(dim: int = 3, seed: int = 1337) -> np.ndarray:
    """Draw a seeded proper-orthogonal matrix using sign-corrected QR.

    Correcting each QR column by the sign of the corresponding diagonal of
    ``R`` removes the orientation bias in the naive QR construction. A final
    column flip conditions the draw on determinant +1, yielding an SO(n)
    matrix (up to floating-point round-off).
    """

    dimension = _validated_dimension(dim)
    generator = rng(seed)
    candidate = generator.normal(size=(dimension, dimension))
    orthogonal, upper = np.linalg.qr(candidate)
    diagonal = np.diag(upper)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    orthogonal = orthogonal * signs
    if float(np.linalg.det(orthogonal)) < 0.0:
        orthogonal[:, -1] *= -1.0
    return np.asarray(orthogonal, dtype=np.float64)


def spd_matrix(
    dim: int = 3,
    seed: int = 1337,
    *,
    regularization: float = 1.0e-6,
) -> np.ndarray:
    """Construct a seeded strictly positive-definite symmetric matrix."""

    dimension = _validated_dimension(dim)
    if isinstance(regularization, (bool, complex, np.complexfloating)) or not isinstance(
        regularization, (int, float, np.number)
    ):
        raise TypeError("regularization must be a real number")
    floor = float(regularization)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("regularization must be finite and greater than zero")
    generator = rng(seed)
    candidate = generator.normal(size=(dimension, dimension))
    matrix = candidate @ candidate.T
    matrix += floor * np.eye(dimension, dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def rotate_rank2(
    tensor: np.ndarray,
    rotation: np.ndarray,
    *,
    orthogonality_tolerance: float = DEFAULT_ORTHOGONALITY_TOLERANCE,
) -> np.ndarray:
    """Apply ``R T R.T`` after validating a proper-orthogonal basis change."""

    matrix = _square_matrix(tensor, name="tensor")
    basis = _square_matrix(rotation, name="rotation")
    if matrix.shape != basis.shape:
        raise ValueError("tensor and rotation must have the same shape")
    if not np.isfinite(orthogonality_tolerance) or orthogonality_tolerance <= 0.0:
        raise ValueError("orthogonality_tolerance must be finite and greater than zero")
    identity = np.eye(basis.shape[0], dtype=np.float64)
    residual = float(np.linalg.norm(basis.T @ basis - identity, ord="fro"))
    limit = orthogonality_tolerance * max(1.0, basis.shape[0])
    if residual > limit:
        raise ValueError(
            f"rotation must be orthogonal within tolerance (Frobenius residual {residual:.3e})"
        )
    determinant = float(np.linalg.det(basis))
    if determinant <= 0.0 or abs(determinant - 1.0) > limit:
        raise ValueError("rotation must be proper orthogonal with determinant +1")
    with np.errstate(over="ignore", invalid="ignore"):
        rotated = np.asarray(basis @ matrix @ basis.T, dtype=np.float64)
    if not np.all(np.isfinite(rotated)):
        raise ValueError("rotated tensor exceeds the finite float64 domain")
    return rotated


def invariants(
    tensor: np.ndarray,
    *,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> dict[str, float | np.ndarray]:
    """Measure finite-matrix invariants after explicit symmetry validation."""

    matrix = _symmetric_matrix(
        tensor,
        name="tensor",
        symmetry_tolerance=symmetry_tolerance,
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    magnitude = float(np.max(np.abs(matrix), initial=0.0))
    if magnitude == 0.0:
        frobenius = 0.0
    else:
        frobenius = float(magnitude * np.sqrt(np.sum((matrix / magnitude) ** 2)))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        trace = float(np.trace(matrix))
        determinant = float(np.linalg.det(matrix))
    sorted_eigenvalues = np.asarray(np.sort(eigenvalues), dtype=np.float64)
    if not (
        np.isfinite(frobenius)
        and np.isfinite(trace)
        and np.isfinite(determinant)
        and np.all(np.isfinite(sorted_eigenvalues))
    ):
        raise ValueError("tensor invariants exceed the finite float64 domain")
    return {
        "fro": frobenius,
        "trace": trace,
        "determinant": determinant,
        "eigvals": sorted_eigenvalues,
    }


def invariant_residuals(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> dict[str, float]:
    """Return absolute residuals between two symmetric-matrix observations."""

    initial = invariants(reference, symmetry_tolerance=symmetry_tolerance)
    transformed = invariants(candidate, symmetry_tolerance=symmetry_tolerance)
    initial_eigenvalues = np.asarray(initial["eigvals"], dtype=float)
    transformed_eigenvalues = np.asarray(transformed["eigvals"], dtype=float)
    if initial_eigenvalues.shape != transformed_eigenvalues.shape:
        raise ValueError("reference and candidate must have the same dimension")
    return {
        "fro": abs(float(transformed["fro"]) - float(initial["fro"])),
        "trace": abs(float(transformed["trace"]) - float(initial["trace"])),
        "determinant": abs(
            float(transformed["determinant"]) - float(initial["determinant"])
        ),
        "eigvals_max": float(
            np.max(np.abs(transformed_eigenvalues - initial_eigenvalues), initial=0.0)
        ),
    }


def analytic_signal(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the canonical NumPy-FFT analytic signal.

    A single backend is used deliberately so merely installing SciPy cannot
    change a TFT run's numeric path.
    """

    signal = _finite_real_array(x, name="x")
    if signal.ndim == 0:
        raise ValueError("x must have at least one dimension")
    normalized_axis = _normalized_axis(axis, signal.ndim)
    length = signal.shape[normalized_axis]
    if length < 2:
        raise ValueError("x must contain at least two samples along axis")

    scale = np.max(np.abs(signal), axis=normalized_axis, keepdims=True)
    scaled_signal = signal / np.where(scale > 0.0, scale, 1.0)
    spectrum = np.fft.fft(scaled_signal, axis=normalized_axis)
    multiplier = np.zeros(length, dtype=np.float64)
    multiplier[0] = 1.0
    if length % 2 == 0:
        multiplier[length // 2] = 1.0
        multiplier[1 : length // 2] = 2.0
    else:
        multiplier[1 : (length + 1) // 2] = 2.0
    shape = [1] * signal.ndim
    shape[normalized_axis] = length
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.fft.ifft(spectrum * multiplier.reshape(shape), axis=normalized_axis) * scale
    if not np.all(np.isfinite(result)):
        raise ValueError("analytic signal exceeds the finite float64 domain")
    return result


def phi_lock_pair(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Return per-trace real/quadrature partners with matched RMS energy.

    ``phi`` here means a pi/2 phase offset, not the golden ratio. A nonzero
    signal containing only DC and/or the even-length Nyquist component has no
    Hilbert quadrature and is rejected. An all-zero signal returns two zeros.
    """

    analytic = analytic_signal(x, axis=axis)
    normalized_axis = _normalized_axis(axis, analytic.ndim)
    real = np.real(analytic).astype(np.float64, copy=False)
    quadrature = np.imag(analytic).astype(np.float64, copy=False)
    real_rms = _stable_rms(real, normalized_axis, keepdims=True)
    quadrature_rms = _stable_rms(quadrature, normalized_axis, keepdims=True)

    nonzero_real = real_rms > 0.0
    missing_quadrature = nonzero_real & (quadrature_rms == 0.0)
    if np.any(missing_quadrature):
        raise ValueError("quadrature is undefined for a nonzero DC/Nyquist-only trace")
    scale = np.ones_like(real_rms)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(real_rms, quadrature_rms, out=scale, where=quadrature_rms > 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        matched_quadrature = quadrature * scale
    if not np.all(np.isfinite(matched_quadrature)):
        raise ValueError("matched quadrature exceeds the finite float64 domain")
    return real.copy(), matched_quadrature


def quadrature_metrics(
    real: np.ndarray,
    quadrature: np.ndarray,
    *,
    axis: int = -1,
) -> dict[str, float | np.ndarray]:
    """Measure per-trace RMS values and mean-centered correlation."""

    first = _finite_real_array(real, name="real")
    second = _finite_real_array(quadrature, name="quadrature")
    if first.shape != second.shape or first.ndim == 0:
        raise ValueError("real and quadrature must have the same non-scalar shape")
    normalized_axis = _normalized_axis(axis, first.ndim)
    if first.shape[normalized_axis] < 2:
        raise ValueError("signals must contain at least two samples along axis")
    first_rms = _stable_rms(first, normalized_axis, keepdims=False)
    second_rms = _stable_rms(second, normalized_axis, keepdims=False)
    first_scale = np.max(np.abs(first), axis=normalized_axis, keepdims=True)
    second_scale = np.max(np.abs(second), axis=normalized_axis, keepdims=True)
    first_normalized = first / np.where(first_scale > 0.0, first_scale, 1.0)
    second_normalized = second / np.where(second_scale > 0.0, second_scale, 1.0)
    first_centered = first_normalized - np.mean(
        first_normalized, axis=normalized_axis, keepdims=True
    )
    second_centered = second_normalized - np.mean(
        second_normalized, axis=normalized_axis, keepdims=True
    )
    numerator = np.sum(first_centered * second_centered, axis=normalized_axis)
    denominator = np.linalg.norm(first_centered, axis=normalized_axis) * np.linalg.norm(
        second_centered, axis=normalized_axis
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )

    def scalar_or_array(value: np.ndarray) -> float | np.ndarray:
        return float(value) if np.ndim(value) == 0 else np.asarray(value, dtype=np.float64)

    return {
        "real_rms": scalar_or_array(first_rms),
        "quadrature_rms": scalar_or_array(second_rms),
        "correlation": scalar_or_array(correlation),
    }


def map_to_audio(
    eigvals: np.ndarray,
    fmin: float = 220.0,
    fmax: float = 880.0,
) -> np.ndarray:
    """Linearly map a finite one-dimensional spectrum into a pitch interval."""

    values = _finite_real_array(eigvals, name="eigvals")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigvals must be a non-empty one-dimensional array")
    if isinstance(fmin, (bool, complex, np.complexfloating)) or isinstance(
        fmax, (bool, complex, np.complexfloating)
    ):
        raise TypeError("fmin and fmax must be real numbers")
    lower = float(fmin)
    upper = float(fmax)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("fmin and fmax must be finite")
    if lower <= 0.0 or upper <= lower:
        raise ValueError("frequency bounds must satisfy 0 < fmin < fmax")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum == minimum:
        midpoint = lower / 2.0 + upper / 2.0
        if not np.isfinite(midpoint):
            raise ValueError("frequency midpoint exceeds the finite float64 domain")
        return np.full(values.shape, midpoint, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        span = maximum - minimum
    if np.isfinite(span):
        normalized = (values - minimum) / span
    else:
        # Avoid overflow for spectra spanning almost the entire float64
        # domain, e.g. [-1e308, +1e308].
        center = minimum / 2.0 + maximum / 2.0
        half_span = maximum / 2.0 - minimum / 2.0
        normalized = 0.5 + 0.5 * ((values - center) / half_span)
    mapped = lower + normalized * (upper - lower)
    if not np.all(np.isfinite(mapped)):
        raise ValueError("frequency mapping exceeds the finite float64 domain")
    return np.asarray(mapped, dtype=np.float64)
