import numpy as np
import pytest

from tft.experiment import (
    MAX_SAMPLE_COUNT,
    ExperimentConfig,
    run_experiment,
    synthesize_quadrature,
)
from tft.resonance import rng


def test_experiment_is_repeatable_in_one_runtime():
    config = ExperimentConfig(seed=2026, dimension=4, duration_s=0.05, sample_rate=8_000, fmax_hz=1_500)
    first = run_experiment(config)
    second = run_experiment(config)

    assert np.array_equal(first.tensor, second.tensor)
    assert np.array_equal(first.rotation, second.rotation)
    assert np.array_equal(first.stereo, second.stereo)
    assert first.observations()["status"] == "pass"


def test_requested_sample_rate_controls_sample_count():
    config = ExperimentConfig(
        seed=3,
        dimension=3,
        sample_rate=8_000,
        duration_s=0.125,
        fmin_hz=100,
        fmax_hz=1_200,
    )
    result = run_experiment(config)

    assert result.stereo.shape == (1_000, 2)
    assert np.max(np.abs(result.stereo)) <= config.target_peak + 1.0e-12
    assert result.observations()["sonification"]["sample_rate"] == 8_000


def test_custom_tensor_is_preserved_and_rotated():
    tensor = np.diag([1.0, 2.0, 4.0])
    config = ExperimentConfig(dimension=3, sample_rate=8_000, duration_s=0.02, fmax_hz=1_000)
    result = run_experiment(config, tensor=tensor)

    assert np.array_equal(result.tensor, tensor)
    assert np.allclose(result.rotated_tensor, result.rotation @ tensor @ result.rotation.T)
    assert result.observations()["status"] == "pass"


def test_complex_rotation_is_rejected_instead_of_silently_cast():
    config = ExperimentConfig(dimension=2, sample_rate=8_000, duration_s=0.02, fmax_hz=1_000)
    rotation = np.eye(2, dtype=complex)
    rotation[0, 0] = 1.0 + 1.0j
    with pytest.raises(TypeError, match="real"):
        run_experiment(config, tensor=np.eye(2), rotation=rotation)


def test_traceless_tensor_uses_combined_absolute_relative_tolerance():
    config = ExperimentConfig(dimension=2, sample_rate=8_000, duration_s=0.02, fmax_hz=1_000)
    result = run_experiment(config, tensor=np.diag([-0.5, 0.5]))
    assert result.observations()["status"] == "pass"


def test_result_does_not_alias_caller_arrays_and_is_read_only():
    tensor = np.diag([1.0, 2.0])
    rotation = np.eye(2)
    config = ExperimentConfig(dimension=2, sample_rate=8_000, duration_s=0.02, fmax_hz=1_000)
    result = run_experiment(config, tensor=tensor, rotation=rotation)
    tensor[0, 0] = 99.0
    rotation[0, 0] = -1.0
    assert result.tensor[0, 0] == 1.0
    assert result.rotation[0, 0] == 1.0
    with pytest.raises(ValueError):
        result.tensor[0, 0] = 5.0
    with pytest.raises(TypeError):
        result.initial_invariants["fro"] = 5.0


def test_maximum_seed_wraps_rotation_seed_safely():
    config = ExperimentConfig(
        seed=(1 << 64) - 1,
        dimension=2,
        sample_rate=8_000,
        duration_s=0.02,
        fmax_hz=1_000,
    )
    assert run_experiment(config).observations()["status"] == "pass"


@pytest.mark.parametrize(
    "updates",
    [
        {"seed": -1},
        {"dimension": 1},
        {"sample_rate": 7_999},
        {"duration_s": 0.0},
        {"fmin_hz": 0.0},
        {"fmin_hz": 500.0, "fmax_hz": 500.0},
        {"sample_rate": 8_000, "fmax_hz": 4_000.0},
        {"target_peak": 1.1},
        {"fade_s": -0.01},
        {"regularization": 0.0},
        {"tolerance": float("nan")},
    ],
)
def test_invalid_config_is_rejected(updates):
    with pytest.raises((TypeError, ValueError)):
        ExperimentConfig(**updates)


def test_config_rejects_complex_scalars_and_oversized_render():
    with pytest.raises(TypeError):
        ExperimentConfig(duration_s=np.complex128(1.0 + 2.0j))
    with pytest.raises(ValueError, match="too many samples"):
        ExperimentConfig(sample_rate=192_000, duration_s=(MAX_SAMPLE_COUNT + 1) / 192_000)
    with pytest.raises(ValueError, match="two-sample"):
        ExperimentConfig(
            sample_rate=8_000,
            duration_s=2 / 8_000,
            fade_s=1 / 8_000,
            fmax_hz=1_000,
        )


def test_quadrature_synthesis_rejects_aliasing():
    with pytest.raises(ValueError, match="Nyquist"):
        synthesize_quadrature(
            np.array([4_000.0]),
            sample_rate=8_000,
            sample_count=100,
            seed=1,
            target_peak=0.9,
            fade_s=0.0,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"sample_count": 10.5},
        {"target_peak": 2.0},
        {"fade_s": -1.0},
        {"frequencies_hz": np.array([100.0 + 1.0j])},
    ],
)
def test_direct_synthesis_validates_public_arguments(updates):
    arguments = {
        "frequencies_hz": np.array([100.0]),
        "sample_rate": 8_000,
        "sample_count": 100,
        "seed": 1,
        "target_peak": 0.9,
        "fade_s": 0.0,
    }
    arguments.update(updates)
    with pytest.raises((TypeError, ValueError)):
        synthesize_quadrature(**arguments)


def test_multiblock_synthesis_matches_direct_oracle():
    frequencies = np.array([220.0, 440.0])
    sample_rate = 8_000
    sample_count = 262_145
    seed = 5
    target_peak = 0.8
    fade_s = 0.01
    actual = synthesize_quadrature(
        frequencies,
        sample_rate=sample_rate,
        sample_count=sample_count,
        seed=seed,
        target_peak=target_peak,
        fade_s=fade_s,
    )

    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    phases = rng(seed).uniform(0.0, 2.0 * np.pi, size=frequencies.size)
    angles = 2.0 * np.pi * frequencies[:, None] * time[None, :] + phases[:, None]
    expected = np.column_stack((np.mean(np.cos(angles), axis=0), np.mean(np.sin(angles), axis=0)))
    fade_frames = int(round(fade_s * sample_rate))
    ramp = np.sin(np.arange(fade_frames) / (fade_frames - 1) * (np.pi / 2.0)) ** 2
    envelope = np.ones(sample_count)
    envelope[:fade_frames] = ramp
    envelope[-fade_frames:] = ramp[::-1]
    expected *= envelope[:, None]
    expected *= target_peak / np.max(np.abs(expected))

    assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-15)
