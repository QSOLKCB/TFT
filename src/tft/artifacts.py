"""Artifact writing and replay verification for TFT Lab runs."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable
import wave

import numpy as np

from ._version import __version__
from .experiment import (
    ALGORITHM_ID,
    MAX_SAMPLE_COUNT,
    OBSERVATIONS_SCHEMA,
    RECIPE_SCHEMA,
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)
from .resonance import DEFAULT_ORTHOGONALITY_TOLERANCE, DEFAULT_SYMMETRY_TOLERANCE


MANIFEST_SCHEMA = "tft.run.manifest.v1"
REPLAY_CONTRACT = {
    "numerical": (
        "Stored matrices are re-evaluated against the recorded tolerance. This is the "
        "portable replay criterion."
    ),
    "byte_identity": (
        "SHA-256 values are integrity receipts for this bundle, not a promise that a "
        "different NumPy, BLAS, libm, Python, or Matplotlib stack will regenerate every byte."
    ),
}
VALIDATION_CONTRACT = {
    "symmetry_relative_tolerance": DEFAULT_SYMMETRY_TOLERANCE,
    "orthogonality_frobenius_tolerance": DEFAULT_ORTHOGONALITY_TOLERANCE,
}
MAX_MANIFEST_BYTES = 2_000_000
MAX_ARTIFACT_BYTES = {
    "recipe.json": 1_000_000,
    "observations.json": 2_000_000,
    "tensor.npy": 65_536,
    "rotation.npy": 65_536,
    "rotated_tensor.npy": 65_536,
    "sonification.wav": MAX_SAMPLE_COUNT * 4 + 1_000_000,
    "comparison.png": 50_000_000,
}


class ArtifactError(RuntimeError):
    """Raised when a run bundle cannot be written or verified safely."""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    passed: bool
    manifest_path: Path
    checks: tuple[str, ...]
    errors: tuple[str, ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite-only canonical JSON with a terminating newline."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"value cannot be encoded as canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    canonical = np.ascontiguousarray(array, dtype="<f8")
    np.save(buffer, canonical, allow_pickle=False)
    return buffer.getvalue()


def pcm16_wave_bytes(stereo: np.ndarray, sample_rate: int) -> tuple[bytes, np.ndarray]:
    """Return a standard little-endian PCM16 WAV and its quantized samples."""

    samples = np.asarray(stereo, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] < 1:
        raise ArtifactError("stereo audio must have shape (sample_count, 2)")
    if not np.all(np.isfinite(samples)):
        raise ArtifactError("stereo audio must be finite")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise ArtifactError("sample_rate must be an integer")
    if not 8_000 <= int(sample_rate) <= 192_000:
        raise ArtifactError("sample_rate must be between 8000 and 192000")

    clipped = np.clip(samples, -1.0, 1.0)
    scaled = clipped * 32767.0
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    pcm = rounded.astype("<i2", copy=False)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(int(sample_rate))
        writer.setnframes(int(pcm.shape[0]))
        writer.writeframes(pcm.tobytes(order="C"))
    return buffer.getvalue(), pcm


def _artifact_entry(path: str, data: bytes, media_type: str, **metadata: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path,
        "media_type": media_type,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    entry.update(metadata)
    return entry


def _runtime_metadata() -> dict[str, str]:
    return {
        "tft": __version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def write_bundle(
    output_dir: str | Path,
    result: ExperimentResult,
    *,
    tensor_origin: str,
    rotation_origin: str,
    figure_png: bytes | None = None,
    force: bool = False,
) -> Path:
    """Write a self-describing run bundle and return its manifest path."""

    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be an ExperimentResult")
    expected_tensor_origin = f"generated:spd(seed={result.config.seed})"
    rotation_seed = (result.config.seed + 1) & ((1 << 64) - 1)
    expected_rotation_origin = f"generated:so(seed={rotation_seed})"
    _validate_origin(tensor_origin, expected_tensor_origin)
    _validate_origin(rotation_origin, expected_rotation_origin)
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise ArtifactError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not force:
        raise ArtifactError(f"output directory is not empty (use --force to overwrite known files): {output}")
    output.mkdir(parents=True, exist_ok=True)

    recipe = {
        "schema": RECIPE_SCHEMA,
        "algorithm": ALGORITHM_ID,
        "software": {"name": "qsol-tft", "version": __version__},
        "validation": VALIDATION_CONTRACT,
        "config": result.config.as_dict(),
        "inputs": {
            "tensor": {"origin": tensor_origin, "path": "tensor.npy"},
            "rotation": {"origin": rotation_origin, "path": "rotation.npy"},
        },
        "outputs": {
            "rotated_tensor": "rotated_tensor.npy",
            "observations": "observations.json",
            "sonification": "sonification.wav",
        },
        "replay_contract": REPLAY_CONTRACT,
    }
    observations = result.observations()
    wav_data, pcm = pcm16_wave_bytes(result.stereo, result.config.sample_rate)

    payloads: list[tuple[str, bytes, str, dict[str, Any]]] = [
        ("recipe.json", canonical_json_bytes(recipe), "application/json", {}),
        ("observations.json", canonical_json_bytes(observations), "application/json", {}),
        ("tensor.npy", _npy_bytes(result.tensor), "application/x-npy", {"dtype": "float64"}),
        ("rotation.npy", _npy_bytes(result.rotation), "application/x-npy", {"dtype": "float64"}),
        (
            "rotated_tensor.npy",
            _npy_bytes(result.rotated_tensor),
            "application/x-npy",
            {"dtype": "float64"},
        ),
        (
            "sonification.wav",
            wav_data,
            "audio/wav",
            {
                "encoding": "PCM16LE",
                "channels": 2,
                "sample_rate": result.config.sample_rate,
                "sample_count": int(pcm.shape[0]),
            },
        ),
    ]
    if figure_png is not None:
        if not isinstance(figure_png, bytes) or not figure_png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ArtifactError("figure_png must contain PNG bytes")
        payloads.append(("comparison.png", figure_png, "image/png", {"canonical": False}))

    entries: list[dict[str, Any]] = []
    for relative_path, data, media_type, metadata in payloads:
        _atomic_write(output / relative_path, data)
        entries.append(_artifact_entry(relative_path, data, media_type, **metadata))

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "algorithm": ALGORITHM_ID,
        "artifacts": sorted(entries, key=lambda item: item["path"]),
        "runtime": _runtime_metadata(),
        "result": observations["status"],
        "claim_boundary": observations["claim_boundary"],
    }
    manifest_path = output / "manifest.json"
    _atomic_write(manifest_path, canonical_json_bytes(manifest))
    return manifest_path


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    def reject_nonfinite_constant(value: str) -> None:
        raise ArtifactError(f"non-finite JSON constant is not allowed: {value}")

    def parse_finite_float(value: str) -> float:
        converted = float(value)
        if not np.isfinite(converted):
            raise ArtifactError(f"non-finite JSON number is not allowed: {value}")
        return converted

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=reject_nonfinite_constant,
                parse_float=parse_finite_float,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON root must be an object: {path}")
    return value


def _safe_artifact_path(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ArtifactError("artifact path must be a non-empty POSIX relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ArtifactError(f"unsafe artifact path: {relative!r}")
    resolved_base = base.resolve()
    resolved = (base / candidate).resolve()
    if not resolved.is_relative_to(resolved_base):
        raise ArtifactError(f"artifact escapes bundle directory: {relative!r}")
    return resolved


def _compare_recorded_observations(
    recorded: dict[str, Any], recomputed: dict[str, Any], tolerance: float
) -> list[str]:
    errors: list[str] = []

    def compare(left: Any, right: Any, location: str) -> None:
        if isinstance(right, dict):
            if not isinstance(left, dict):
                errors.append(f"{location} must be an object")
                return
            if set(left) != set(right):
                errors.append(f"{location} has unexpected or missing fields")
                return
            for key in sorted(right):
                compare(left[key], right[key], f"{location}.{key}")
            return
        if isinstance(right, list):
            if not isinstance(left, list) or len(left) != len(right):
                errors.append(f"{location} must be a list of length {len(right)}")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                compare(left_item, right_item, f"{location}[{index}]")
            return
        if isinstance(right, bool) or right is None or isinstance(right, str):
            if left != right:
                errors.append(f"{location} does not replay exactly")
            return
        if isinstance(right, int):
            if isinstance(left, bool) or not isinstance(left, int) or left != right:
                errors.append(f"{location} does not replay exactly")
            return
        if isinstance(right, float):
            if isinstance(left, bool) or not isinstance(left, (int, float)):
                errors.append(f"{location} must be numeric")
                return
            if location.endswith(".tolerance"):
                matches = float(left) == right
            else:
                matches = bool(np.isclose(float(left), right, rtol=tolerance, atol=tolerance))
            if not matches:
                errors.append(f"{location} does not numerically replay")
            return
        if left != right:
            errors.append(f"{location} does not replay")

    compare(recorded, recomputed, "observations")
    return errors


def _validate_origin(origin: Any, expected_generated: str) -> str:
    if origin == expected_generated:
        return "generated"
    if isinstance(origin, str) and origin.startswith("provided:") and len(origin) > 9:
        return "provided"
    raise ArtifactError(
        f"input origin must be {expected_generated!r} or a non-empty 'provided:' label"
    )


def _npy_header(path: Path) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            return shape, fortran_order, dtype, handle.tell()
        raise ArtifactError(
            f"{path.name} must use the canonical NPY 1.0 format, not version {version}"
        )


def _validate_npy_header_contract(path: Path, dimension: int) -> None:
    shape, fortran_order, dtype, data_offset = _npy_header(path)
    expected_shape = (dimension, dimension)
    if shape != expected_shape:
        raise ArtifactError(f"{path.name} must have shape {expected_shape}")
    if fortran_order:
        raise ArtifactError(f"{path.name} must use canonical C-contiguous order")
    if dtype.str != "<f8":
        raise ArtifactError(f"{path.name} must use little-endian float64 values")
    expected_size = data_offset + dimension * dimension * 8
    if path.stat().st_size != expected_size:
        raise ArtifactError(f"{path.name} has a truncated or noncanonical NPY payload")


def _validate_npy_contract(path: Path, array: np.ndarray, dimension: int) -> None:
    _validate_npy_header_contract(path, dimension)
    expected_shape = (dimension, dimension)
    if array.shape != expected_shape:
        raise ArtifactError(f"{path.name} must have shape {expected_shape}")
    if not array.flags.c_contiguous:
        raise ArtifactError(f"{path.name} must use canonical C-contiguous order")
    if array.dtype != np.dtype("<f8"):
        raise ArtifactError(f"{path.name} must use little-endian float64 values")
    if not np.all(np.isfinite(array)):
        raise ArtifactError(f"{path.name} must contain only finite values")


def verify_bundle(manifest_path: str | Path) -> VerificationReport:
    """Verify bundle hashes, WAV contract, and portable numerical replay."""

    path = Path(manifest_path)
    checks: list[str] = []
    errors: list[str] = []
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ArtifactError("manifest exceeds the verifier size limit")
    except OSError as exc:
        return VerificationReport(False, path, tuple(checks), (f"cannot stat manifest: {exc}",))
    try:
        manifest = _load_json(path)
    except ArtifactError as exc:
        return VerificationReport(False, path, tuple(checks), (str(exc),))

    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"unsupported manifest schema: {manifest.get('schema')!r}")
    if manifest.get("algorithm") != ALGORITHM_ID:
        errors.append(f"unsupported algorithm: {manifest.get('algorithm')!r}")
    expected_manifest_fields = {
        "schema",
        "algorithm",
        "artifacts",
        "runtime",
        "result",
        "claim_boundary",
    }
    if set(manifest) != expected_manifest_fields:
        errors.append("manifest has unexpected or missing top-level fields")

    artifact_entries = manifest.get("artifacts")
    if not isinstance(artifact_entries, list) or not artifact_entries:
        errors.append("manifest artifacts must be a non-empty list")
        artifact_entries = []

    base = path.parent
    seen: set[str] = set()
    artifact_paths: dict[str, Path] = {}
    artifact_metadata: dict[str, dict[str, Any]] = {}
    for entry in artifact_entries:
        if not isinstance(entry, dict):
            errors.append("manifest artifact entry must be an object")
            continue
        relative = entry.get("path")
        try:
            artifact_path = _safe_artifact_path(base, relative)
        except ArtifactError as exc:
            errors.append(str(exc))
            continue
        if relative in seen:
            errors.append(f"duplicate artifact entry: {relative}")
            continue
        seen.add(relative)
        artifact_paths[str(relative)] = artifact_path
        artifact_metadata[str(relative)] = entry
        if not artifact_path.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        actual_size = artifact_path.stat().st_size
        maximum_size = MAX_ARTIFACT_BYTES.get(str(relative))
        if maximum_size is None:
            errors.append(f"unsupported artifact in manifest: {relative}")
            continue
        if actual_size > maximum_size:
            errors.append(f"artifact exceeds verifier size limit: {relative}")
            continue
        actual_hash = sha256_file(artifact_path)
        if entry.get("bytes") != actual_size:
            errors.append(f"size mismatch: {relative}")
        if entry.get("sha256") != actual_hash:
            errors.append(f"SHA-256 mismatch: {relative}")
        if entry.get("bytes") == actual_size and entry.get("sha256") == actual_hash:
            checks.append(f"integrity:{relative}")

    required = {
        "recipe.json",
        "observations.json",
        "tensor.npy",
        "rotation.npy",
        "rotated_tensor.npy",
        "sonification.wav",
    }
    missing_entries = sorted(required - set(artifact_paths))
    if missing_entries:
        errors.append(f"manifest omits required artifacts: {', '.join(missing_entries)}")

    expected_media_types = {
        "recipe.json": "application/json",
        "observations.json": "application/json",
        "tensor.npy": "application/x-npy",
        "rotation.npy": "application/x-npy",
        "rotated_tensor.npy": "application/x-npy",
        "sonification.wav": "audio/wav",
        "comparison.png": "image/png",
    }
    base_artifact_fields = {"path", "media_type", "bytes", "sha256"}
    for name, entry in artifact_metadata.items():
        expected_media_type = expected_media_types.get(name)
        if expected_media_type is None:
            errors.append(f"unsupported artifact in manifest: {name}")
        elif entry.get("media_type") != expected_media_type:
            errors.append(f"media type mismatch: {name}")
        expected_fields = set(base_artifact_fields)
        if name.endswith(".npy"):
            expected_fields.add("dtype")
        elif name == "sonification.wav":
            expected_fields.update({"encoding", "channels", "sample_rate", "sample_count"})
        elif name == "comparison.png":
            expected_fields.add("canonical")
        if set(entry) != expected_fields:
            errors.append(f"artifact metadata fields are malformed: {name}")

    # Integrity and numerical replay are intentionally independent reports.
    # An altered optional PNG or WAV must fail the bundle overall, but it does
    # not prevent the stored matrices from being evaluated numerically.
    if required.issubset(artifact_paths) and all(
        artifact_paths[name].is_file() for name in required
    ):
        try:
            recipe = _load_json(artifact_paths["recipe.json"])
            recorded = _load_json(artifact_paths["observations.json"])
            if recipe.get("schema") != RECIPE_SCHEMA:
                raise ArtifactError(f"unsupported recipe schema: {recipe.get('schema')!r}")
            if recipe.get("algorithm") != ALGORITHM_ID:
                raise ArtifactError(f"unsupported recipe algorithm: {recipe.get('algorithm')!r}")
            expected_recipe_fields = {
                "schema",
                "algorithm",
                "software",
                "validation",
                "config",
                "inputs",
                "outputs",
                "replay_contract",
            }
            if set(recipe) != expected_recipe_fields:
                raise ArtifactError("recipe has unexpected or missing top-level fields")
            software = recipe.get("software")
            if (
                not isinstance(software, dict)
                or set(software) != {"name", "version"}
                or software.get("name") != "qsol-tft"
                or not isinstance(software.get("version"), str)
                or not software["version"]
            ):
                raise ArtifactError("recipe software identity is malformed")
            if recipe.get("validation") != VALIDATION_CONTRACT:
                raise ArtifactError("recipe validation contract is unsupported")
            if recipe.get("replay_contract") != REPLAY_CONTRACT:
                raise ArtifactError("recipe replay contract is unsupported")
            runtime = manifest.get("runtime")
            expected_runtime_fields = {
                "tft",
                "python",
                "python_implementation",
                "numpy",
                "platform",
            }
            if (
                not isinstance(runtime, dict)
                or set(runtime) != expected_runtime_fields
                or any(not isinstance(runtime[key], str) or not runtime[key] for key in runtime)
                or runtime.get("tft") != software["version"]
            ):
                raise ArtifactError("manifest runtime metadata is malformed")
            if recorded.get("schema") != OBSERVATIONS_SCHEMA:
                raise ArtifactError(
                    f"unsupported observations schema: {recorded.get('schema')!r}"
                )
            if recorded.get("algorithm") != ALGORITHM_ID:
                raise ArtifactError(
                    f"unsupported observations algorithm: {recorded.get('algorithm')!r}"
                )
            config_data = recipe.get("config")
            if not isinstance(config_data, dict):
                raise ArtifactError("recipe config must be an object")
            expected_config_fields = {field.name for field in fields(ExperimentConfig)}
            if set(config_data) != expected_config_fields:
                raise ArtifactError("recipe config has unexpected or missing fields")
            config = ExperimentConfig(**config_data)
            expected_tensor_origin = f"generated:spd(seed={config.seed})"
            rotation_seed = (config.seed + 1) & ((1 << 64) - 1)
            expected_rotation_origin = f"generated:so(seed={rotation_seed})"
            inputs = recipe.get("inputs")
            outputs = recipe.get("outputs")
            if not isinstance(inputs, dict) or set(inputs) != {"tensor", "rotation"}:
                raise ArtifactError("recipe inputs must declare tensor and rotation")
            if not isinstance(inputs["tensor"], dict) or set(inputs["tensor"]) != {
                "origin",
                "path",
            }:
                raise ArtifactError("recipe tensor input is malformed")
            if not isinstance(inputs["rotation"], dict) or set(inputs["rotation"]) != {
                "origin",
                "path",
            }:
                raise ArtifactError("recipe rotation input is malformed")
            if inputs["tensor"].get("path") != "tensor.npy":
                raise ArtifactError("recipe tensor path must be tensor.npy")
            if inputs["rotation"].get("path") != "rotation.npy":
                raise ArtifactError("recipe rotation path must be rotation.npy")
            _validate_origin(inputs["tensor"].get("origin"), expected_tensor_origin)
            _validate_origin(inputs["rotation"].get("origin"), expected_rotation_origin)
            expected_outputs = {
                "rotated_tensor": "rotated_tensor.npy",
                "observations": "observations.json",
                "sonification": "sonification.wav",
            }
            if outputs != expected_outputs:
                raise ArtifactError("recipe output paths do not match the run schema")
            for name in ("tensor.npy", "rotation.npy", "rotated_tensor.npy"):
                _validate_npy_header_contract(artifact_paths[name], config.dimension)
            tensor = np.load(artifact_paths["tensor.npy"], allow_pickle=False)
            rotation = np.load(artifact_paths["rotation.npy"], allow_pickle=False)
            stored_rotated = np.load(artifact_paths["rotated_tensor.npy"], allow_pickle=False)
            for name, array in (
                ("tensor.npy", tensor),
                ("rotation.npy", rotation),
                ("rotated_tensor.npy", stored_rotated),
            ):
                _validate_npy_contract(artifact_paths[name], array, config.dimension)
                if artifact_metadata[name].get("dtype") != "float64":
                    raise ArtifactError(f"manifest dtype mismatch: {name}")
            replay = run_experiment(config, tensor=tensor, rotation=rotation)
            if stored_rotated.shape != replay.rotated_tensor.shape or not np.allclose(
                stored_rotated,
                replay.rotated_tensor,
                rtol=config.tolerance,
                atol=config.tolerance,
            ):
                errors.append("stored rotated tensor does not numerically replay")
            else:
                checks.append("numerical:rotated_tensor")
            replay_observations = replay.observations()
            observation_errors = _compare_recorded_observations(
                recorded, replay_observations, config.tolerance
            )
            if recorded.get("status") != replay_observations.get("status"):
                observation_errors.append("recorded observation status does not numerically replay")
            if replay_observations.get("status") != "pass":
                observation_errors.append("numerical replay reports failed invariant checks")
            errors.extend(observation_errors)
            if not observation_errors:
                checks.append("numerical:observations")

            if manifest.get("result") != recorded.get("status"):
                errors.append("manifest result does not match recorded observations")
            if manifest.get("result") != replay_observations.get("status"):
                errors.append("manifest result does not match numerical replay")
            if manifest.get("claim_boundary") != recorded.get("claim_boundary"):
                errors.append("manifest claim boundary does not match recorded observations")
            if manifest.get("claim_boundary") != replay_observations.get("claim_boundary"):
                errors.append("manifest claim boundary does not match numerical replay")

            with wave.open(str(artifact_paths["sonification.wav"]), "rb") as reader:
                expected_audio_bytes = config.sample_count * 2 * 2
                audio_payload = reader.readframes(config.sample_count)
                wav_contract = (
                    reader.getnchannels() == 2
                    and reader.getsampwidth() == 2
                    and reader.getframerate() == config.sample_rate
                    and reader.getnframes() == config.sample_count
                    and reader.getcomptype() == "NONE"
                    and len(audio_payload) == expected_audio_bytes
                )
            expected_wav, expected_pcm = pcm16_wave_bytes(
                replay.stereo, replay.config.sample_rate
            )
            pcm_replay = False
            if len(audio_payload) == expected_audio_bytes:
                actual_pcm = np.frombuffer(audio_payload, dtype="<i2").reshape((-1, 2))
                pcm_delta = np.abs(
                    actual_pcm.astype(np.int32) - expected_pcm.astype(np.int32)
                )
                pcm_replay = bool(np.max(pcm_delta, initial=0) <= 2)
            wav_entry = artifact_metadata["sonification.wav"]
            wav_manifest_contract = (
                wav_entry.get("encoding") == "PCM16LE"
                and wav_entry.get("channels") == 2
                and wav_entry.get("sample_rate") == config.sample_rate
                and wav_entry.get("sample_count") == config.sample_count
                and artifact_paths["sonification.wav"].stat().st_size == len(expected_wav)
            )
            if wav_contract and wav_manifest_contract:
                checks.append("contract:sonification.wav")
            else:
                errors.append("WAV data does not match the recorded PCM16 stereo contract")
            if pcm_replay:
                checks.append("numerical:sonification-pcm16")
            else:
                errors.append("WAV samples do not numerically replay within 2 PCM16 levels")
        except (
            ArtifactError,
            EOFError,
            OSError,
            OverflowError,
            ValueError,
            TypeError,
            KeyError,
            wave.Error,
        ) as exc:
            errors.append(f"numerical replay failed: {exc}")

    return VerificationReport(not errors, path, tuple(checks), tuple(errors))
