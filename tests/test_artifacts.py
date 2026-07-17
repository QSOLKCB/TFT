import json
import hashlib
from pathlib import Path
import wave

import numpy as np
import pytest

from tft.artifacts import ArtifactError, canonical_json_bytes, verify_bundle, write_bundle
from tft.experiment import ExperimentConfig, run_experiment


def _short_result():
    config = ExperimentConfig(
        seed=9,
        dimension=3,
        sample_rate=8_000,
        duration_s=0.02,
        fmin_hz=100,
        fmax_hz=1_000,
    )
    return run_experiment(config)


def _rehash_artifact(manifest_path: Path, artifact_name: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = (manifest_path.parent / artifact_name).read_bytes()
    entry = next(item for item in manifest["artifacts"] if item["path"] == artifact_name)
    entry["bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def test_bundle_round_trip_and_wav_contract(tmp_path: Path):
    result = _short_result()
    manifest = write_bundle(
        tmp_path / "run",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )

    report = verify_bundle(manifest)
    assert report.passed, report.errors
    assert "numerical:observations" in report.checks
    with wave.open(str(tmp_path / "run" / "sonification.wav"), "rb") as reader:
        assert reader.getframerate() == 8_000
        assert reader.getnframes() == 160
        assert reader.getnchannels() == 2
        assert reader.getsampwidth() == 2


def test_tamper_fails_integrity_but_still_reports_numerical_replay(tmp_path: Path):
    result = _short_result()
    manifest = write_bundle(
        tmp_path / "run",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )
    wav_path = tmp_path / "run" / "sonification.wav"
    wav_path.write_bytes(wav_path.read_bytes() + b"tamper")

    report = verify_bundle(manifest)
    assert not report.passed
    assert any("sonification.wav" in error for error in report.errors)
    assert "numerical:observations" in report.checks


def test_existing_nonempty_directory_requires_force(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not empty"):
        write_bundle(
            output,
            _short_result(),
            tensor_origin="generated:spd(seed=9)",
            rotation_origin="generated:so(seed=10)",
        )


def test_verifier_rejects_path_traversal(tmp_path: Path):
    result = _short_result()
    manifest_path = write_bundle(
        tmp_path / "run",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../escape"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    report = verify_bundle(manifest_path)
    assert not report.passed
    assert any("unsafe artifact path" in error for error in report.errors)


def test_canonical_json_is_sorted_finite_and_newline_terminated():
    assert canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n'
    with pytest.raises(ArtifactError):
        canonical_json_bytes({"bad": float("nan")})


def test_verifier_rejects_nonfinite_json_constant(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema":NaN}\n', encoding="utf-8")
    report = verify_bundle(manifest_path)
    assert not report.passed
    assert any("non-finite JSON constant" in error for error in report.errors)


def test_verifier_cannot_pass_a_failed_numerical_result(tmp_path: Path):
    config = ExperimentConfig(
        dimension=3,
        sample_rate=8_000,
        duration_s=0.02,
        fmax_hz=1_000,
        tolerance=5.0e-324,
    )
    result = run_experiment(config)
    assert result.observations()["status"] == "fail"
    manifest = write_bundle(
        tmp_path / "failed",
        result,
        tensor_origin="generated:spd(seed=1337)",
        rotation_origin="generated:so(seed=1338)",
    )
    report = verify_bundle(manifest)
    assert not report.passed
    assert any("failed invariant checks" in error for error in report.errors)


def test_verifier_returns_fail_for_truncated_wav_instead_of_crashing(tmp_path: Path):
    result = _short_result()
    manifest = write_bundle(
        tmp_path / "run",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )
    wav_path = manifest.parent / "sonification.wav"
    wav_path.write_bytes(wav_path.read_bytes()[:44])
    _rehash_artifact(manifest, "sonification.wav")
    report = verify_bundle(manifest)
    assert not report.passed
    assert any("WAV" in error for error in report.errors)


def test_verifier_rejects_unrelated_pcm_with_valid_header(tmp_path: Path):
    result = _short_result()
    manifest = write_bundle(
        tmp_path / "run",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )
    wav_path = manifest.parent / "sonification.wav"
    wav_data = bytearray(wav_path.read_bytes())
    wav_data[44:] = b"\x00" * (len(wav_data) - 44)
    wav_path.write_bytes(wav_data)
    _rehash_artifact(manifest, "sonification.wav")
    report = verify_bundle(manifest)
    assert not report.passed
    assert any("samples do not numerically replay" in error for error in report.errors)


def test_verifier_returns_fail_for_malformed_npy_instead_of_crashing(tmp_path: Path):
    result = _short_result()
    manifest = write_bundle(
        tmp_path / "run",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )
    (manifest.parent / "tensor.npy").write_bytes(b"")
    _rehash_artifact(manifest, "tensor.npy")
    report = verify_bundle(manifest)
    assert not report.passed
    assert any("replay failed" in error for error in report.errors)


def test_verifier_enforces_npy_dtype_and_full_observations(tmp_path: Path):
    result = _short_result()
    manifest = write_bundle(
        tmp_path / "dtype",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )
    np.save(manifest.parent / "tensor.npy", result.tensor.astype(np.float32), allow_pickle=False)
    _rehash_artifact(manifest, "tensor.npy")
    dtype_report = verify_bundle(manifest)
    assert not dtype_report.passed
    assert any("float64" in error for error in dtype_report.errors)

    second_manifest = write_bundle(
        tmp_path / "observations",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )
    observations_path = second_manifest.parent / "observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    observations["matrix"]["rotation_determinant"] = 0.0
    observations_path.write_bytes(canonical_json_bytes(observations))
    _rehash_artifact(second_manifest, "observations.json")
    observation_report = verify_bundle(second_manifest)
    assert not observation_report.passed
    assert any("rotation_determinant" in error for error in observation_report.errors)


def test_verifier_requires_complete_recipe_config(tmp_path: Path):
    result = _short_result()
    manifest = write_bundle(
        tmp_path / "run",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )
    recipe_path = manifest.parent / "recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    del recipe["config"]["seed"]
    recipe_path.write_bytes(canonical_json_bytes(recipe))
    _rehash_artifact(manifest, "recipe.json")
    report = verify_bundle(manifest)
    assert not report.passed
    assert any("recipe config" in error for error in report.errors)


def test_verifier_rejects_huge_json_number_without_crashing(tmp_path: Path):
    result = _short_result()
    manifest = write_bundle(
        tmp_path / "run",
        result,
        tensor_origin="generated:spd(seed=9)",
        rotation_origin="generated:so(seed=10)",
    )
    observations_path = manifest.parent / "observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    observations["matrix"]["initial"]["frobenius_norm"] = 10**1000
    observations_path.write_bytes(canonical_json_bytes(observations))
    _rehash_artifact(manifest, "observations.json")
    report = verify_bundle(manifest)
    assert not report.passed
    assert any("replay failed" in error for error in report.errors)
