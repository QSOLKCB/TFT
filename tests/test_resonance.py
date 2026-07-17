import numpy as np
import pytest

from tft.resonance import (
    analytic_signal,
    invariant_residuals,
    invariants,
    map_to_audio,
    phi_lock_pair,
    quadrature_metrics,
    rng,
    rotate_rank2,
    rotation_matrix,
    spd_matrix,
)


@pytest.mark.parametrize("dimension", [2, 3, 7])
@pytest.mark.parametrize("seed", [0, 123, 9_999])
def test_rotation_is_seeded_proper_orthogonal(dimension, seed):
    first = rotation_matrix(dimension, seed)
    second = rotation_matrix(dimension, seed)
    assert np.array_equal(first, second)
    assert np.allclose(first.T @ first, np.eye(dimension), rtol=1e-12, atol=1e-12)
    assert np.isclose(np.linalg.det(first), 1.0, rtol=1e-12, atol=1e-12)


def test_rotation_sampler_no_longer_has_naive_qr_half_plane_bias():
    cosines = np.array([rotation_matrix(2, seed)[0, 0] for seed in range(400)])
    assert np.any(cosines < -0.5)
    assert abs(float(np.mean(cosines))) < 0.12


def test_spd_matrix_has_declared_positive_floor():
    matrix = spd_matrix(5, 22, regularization=1.0e-5)
    assert np.array_equal(matrix, spd_matrix(5, 22, regularization=1.0e-5))
    assert np.allclose(matrix, matrix.T)
    assert float(np.min(np.linalg.eigvalsh(matrix))) >= 1.0e-5 - 1.0e-12


def test_rotation_preserves_invariants():
    tensor = spd_matrix(dim=4, seed=123)
    rotation = rotation_matrix(dim=4, seed=456)
    rotated = rotate_rank2(tensor, rotation)
    initial = invariants(tensor)
    candidate = invariants(rotated)
    assert np.isclose(initial["fro"], candidate["fro"], rtol=1e-10, atol=1e-12)
    assert np.isclose(initial["trace"], candidate["trace"], rtol=1e-10, atol=1e-12)
    assert np.allclose(initial["eigvals"], candidate["eigvals"], rtol=1e-10, atol=1e-12)


def test_invariants_reject_bad_matrices_instead_of_silently_projecting():
    with pytest.raises(ValueError, match="symmetric"):
        invariants(np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        invariants(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(TypeError, match="real"):
        invariants(np.eye(2, dtype=complex) * (1.0 + 1.0j))
    with pytest.raises(ValueError, match="finite float64 domain"):
        invariants(np.eye(2) * 1.0e308)
    with pytest.raises(ValueError, match="symmetric"):
        invariants(np.array([[0.0, 1.0e-13], [0.0, 0.0]]))


def test_rotate_rejects_reflection_and_nonorthogonal_matrix():
    tensor = np.eye(2)
    with pytest.raises(ValueError, match="proper orthogonal"):
        rotate_rank2(tensor, np.diag([-1.0, 1.0]))
    with pytest.raises(ValueError, match="orthogonal"):
        rotate_rank2(tensor, np.ones((2, 2)))


@pytest.mark.parametrize("length", [255, 256])
def test_analytic_signal_and_phi_pair_for_bin_centered_tone(length):
    index = np.arange(length)
    signal = np.cos(2.0 * np.pi * 5.0 * index / length)
    analytic = analytic_signal(signal)
    assert np.allclose(np.real(analytic), signal, atol=1e-12)
    first, second = phi_lock_pair(signal)
    metrics = quadrature_metrics(first, second)
    assert np.isclose(metrics["real_rms"], metrics["quadrature_rms"], rtol=1e-12)
    assert abs(metrics["correlation"]) < 1e-12


def test_phi_pair_normalizes_each_trace_along_axis():
    index = np.arange(256)
    rows = np.stack(
        [
            np.cos(2.0 * np.pi * 4.0 * index / 256) + 10.0,
            10.0 * np.cos(2.0 * np.pi * 7.0 * index / 256),
        ]
    )
    first, second = phi_lock_pair(rows, axis=1)
    metrics = quadrature_metrics(first, second, axis=1)
    assert np.allclose(metrics["real_rms"], metrics["quadrature_rms"], rtol=1e-12)


def test_phi_pair_defines_zero_and_rejects_dc_only():
    first, second = phi_lock_pair(np.zeros(16))
    assert np.array_equal(first, np.zeros(16))
    assert np.array_equal(second, np.zeros(16))
    with pytest.raises(ValueError, match="undefined"):
        phi_lock_pair(np.ones(16))


def test_signal_helpers_handle_large_finite_dynamic_range():
    analytic = analytic_signal(np.full(16, 1.0e308))
    assert np.all(np.isfinite(analytic))
    metrics = quadrature_metrics(
        np.array([1.0e308, -1.0e308]),
        np.array([-1.0e308, 1.0e308]),
    )
    assert np.isfinite(metrics["real_rms"])
    assert np.isfinite(metrics["quadrature_rms"])
    assert np.isfinite(metrics["correlation"])


def test_analytic_signal_scales_each_trace_independently():
    index = np.arange(64)
    tone = np.cos(2.0 * np.pi * 3.0 * index / 64)
    traces = np.stack((1.0e308 * tone, 1.0e-300 * tone))
    analytic = analytic_signal(traces, axis=1)
    assert np.all(np.isfinite(analytic))
    assert np.allclose(np.real(analytic[0]) / 1.0e308, tone, atol=1.0e-12)
    assert np.allclose(np.real(analytic[1]) / 1.0e-300, tone, atol=1.0e-12)
    assert np.max(np.abs(np.imag(analytic[1]))) > 0.0


def test_subnormal_nonzero_dc_trace_is_rejected():
    with pytest.raises(ValueError, match="undefined"):
        phi_lock_pair(np.full(16, np.nextafter(0.0, 1.0)))


def test_invariant_residuals_has_known_zero_oracle():
    tensor = np.diag([1.0, 2.0, 3.0])
    residuals = invariant_residuals(tensor, tensor.copy())
    assert residuals == {"fro": 0.0, "trace": 0.0, "determinant": 0.0, "eigvals_max": 0.0}


def test_map_to_audio_preserves_nonzero_large_and_small_spans():
    large_offset = map_to_audio(np.array([1.0e12, 1.0e12 + 1000.0]), 100.0, 900.0)
    tiny_span = map_to_audio(np.array([0.0, 1.0e-9]), 100.0, 900.0)
    assert np.array_equal(large_offset, np.array([100.0, 900.0]))
    assert np.allclose(tiny_span, np.array([100.0, 900.0]))
    assert np.array_equal(map_to_audio(np.ones(3), 100.0, 900.0), np.full(3, 500.0))
    full_domain = map_to_audio(np.array([-1.0e308, 0.0, 1.0e308]), 100.0, 900.0)
    assert np.allclose(full_domain, np.array([100.0, 500.0, 900.0]))
    huge_bounds = map_to_audio(np.ones(2), 1.0e308, 1.7e308)
    assert np.all(np.isfinite(huge_bounds))
    assert np.all(huge_bounds == 1.35e308)


@pytest.mark.parametrize(
    "call",
    [
        lambda: rng(None),
        lambda: rotation_matrix(0),
        lambda: spd_matrix(3, regularization=0.0),
        lambda: analytic_signal(np.array([])),
        lambda: analytic_signal(np.array([1.0, np.inf])),
        lambda: map_to_audio(np.array([])),
        lambda: map_to_audio(np.array([1.0, np.nan])),
        lambda: map_to_audio(np.array([1.0 + 1.0j])),
        lambda: map_to_audio(np.array([1.0, 2.0]), 900.0, 100.0),
        lambda: spd_matrix(3, regularization=np.complex128(1.0 + 2.0j)),
        lambda: map_to_audio(np.array([True, False])),
        lambda: map_to_audio(np.array(["1", "2"])),
    ],
)
def test_invalid_inputs_are_rejected(call):
    with pytest.raises((TypeError, ValueError)):
        call()
