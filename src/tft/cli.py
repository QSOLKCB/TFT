"""Command-line interface for TFT Lab 0.2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from . import __version__
from .artifacts import ArtifactError, verify_bundle, write_bundle
from .experiment import ExperimentConfig, run_experiment


MAX_MATRIX_FILE_BYTES = 1_000_000


def _add_render_options(parser: argparse.ArgumentParser, *, include_dimension: bool) -> None:
    parser.add_argument("--out", required=True, type=Path, help="new or empty output directory")
    parser.add_argument("--seed", type=int, default=1337, help="unsigned 64-bit replay seed")
    if include_dimension:
        parser.add_argument("--dim", dest="dimension", type=int, default=3, help="matrix dimension (2-32)")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="WAV sample rate")
    parser.add_argument("--duration", dest="duration_s", type=float, default=2.0, help="audio duration in seconds")
    parser.add_argument("--fmin", dest="fmin_hz", type=float, default=220.0, help="lowest mapped pitch in Hz")
    parser.add_argument("--fmax", dest="fmax_hz", type=float, default=880.0, help="highest mapped pitch in Hz")
    parser.add_argument("--peak", dest="target_peak", type=float, default=0.92, help="maximum floating peak (0, 1]")
    parser.add_argument("--fade", dest="fade_s", type=float, default=0.01, help="shared fade-in/out in seconds")
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-6,
        help="positive diagonal floor for generated SPD matrices",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-10,
        help="portable numerical replay tolerance",
    )
    parser.add_argument("--plot", action="store_true", help="include noncanonical comparison.png (requires Matplotlib)")
    parser.add_argument("--force", action="store_true", help="overwrite TFT files in a non-empty output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tft",
        description="Reproducible tensor invariance and quadrature sonification laboratory",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a seeded generated SPD matrix experiment")
    _add_render_options(demo, include_dimension=True)

    custom = subparsers.add_parser("run", help="run an experiment from a .npy, .csv, or whitespace matrix")
    custom.add_argument("--tensor", required=True, type=Path, help="real symmetric square matrix")
    custom.add_argument("--rotation", type=Path, help="optional proper-orthogonal matrix; otherwise generated")
    _add_render_options(custom, include_dimension=False)

    verify = subparsers.add_parser("verify", help="verify integrity and portable numerical replay")
    verify.add_argument("target", type=Path, help="manifest.json or its containing run directory")
    return parser


def _load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"matrix file does not exist: {path}")
    if path.stat().st_size > MAX_MATRIX_FILE_BYTES:
        raise ValueError(f"matrix file exceeds the {MAX_MATRIX_FILE_BYTES}-byte input limit: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif suffix == ".csv":
            value = np.loadtxt(path, delimiter=",")
        elif suffix in {".txt", ".dat"}:
            value = np.loadtxt(path)
        else:
            raise ValueError("matrix files must use .npy, .csv, .txt, or .dat")
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load matrix {path}: {exc}") from exc
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"matrix must be square and two-dimensional: {path}")
    return array


def _config_from_args(args: argparse.Namespace, dimension: int) -> ExperimentConfig:
    return ExperimentConfig(
        seed=args.seed,
        dimension=dimension,
        sample_rate=args.sample_rate,
        duration_s=args.duration_s,
        fmin_hz=args.fmin_hz,
        fmax_hz=args.fmax_hz,
        target_peak=args.target_peak,
        fade_s=args.fade_s,
        regularization=args.regularization,
        tolerance=args.tolerance,
    )


def _optional_plot(result: object, enabled: bool) -> bytes | None:
    if not enabled:
        return None
    from .visualization import render_comparison_png

    return render_comparison_png(result)  # type: ignore[arg-type]


def _write_and_check(
    args: argparse.Namespace,
    *,
    result: object,
    tensor_origin: str,
    rotation_origin: str,
) -> int:
    figure_png = _optional_plot(result, args.plot)
    manifest_path = write_bundle(
        args.out,
        result,  # type: ignore[arg-type]
        tensor_origin=tensor_origin,
        rotation_origin=rotation_origin,
        figure_png=figure_png,
        force=args.force,
    )
    report = verify_bundle(manifest_path)
    if not report.passed:
        details = "; ".join(report.errors)
        raise ArtifactError(f"post-write verification failed: {details}")
    observations = result.observations()  # type: ignore[attr-defined]
    residual = observations["matrix"]["residuals"]["eigenvalues"]["max_absolute"]
    print(f"TFT run: {observations['status'].upper()}")
    print(f"max eigenvalue residual: {residual:.3e}")
    print(f"manifest: {manifest_path}")
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    config = _config_from_args(args, args.dimension)
    result = run_experiment(config)
    return _write_and_check(
        args,
        result=result,
        tensor_origin=f"generated:spd(seed={config.seed})",
        rotation_origin=f"generated:so(seed={(config.seed + 1) & ((1 << 64) - 1)})",
    )


def _run_custom(args: argparse.Namespace) -> int:
    tensor = _load_matrix(args.tensor)
    dimension = int(tensor.shape[0])
    config = _config_from_args(args, dimension)
    rotation = _load_matrix(args.rotation) if args.rotation is not None else None
    result = run_experiment(config, tensor=tensor, rotation=rotation)
    rotation_origin = (
        f"provided:{args.rotation.name}"
        if args.rotation is not None
        else f"generated:so(seed={(config.seed + 1) & ((1 << 64) - 1)})"
    )
    return _write_and_check(
        args,
        result=result,
        tensor_origin=f"provided:{args.tensor.name}",
        rotation_origin=rotation_origin,
    )


def _run_verify(args: argparse.Namespace) -> int:
    manifest_path = args.target / "manifest.json" if args.target.is_dir() else args.target
    report = verify_bundle(manifest_path)
    if report.passed:
        print(f"PASS: {manifest_path}")
        print(f"checks: {len(report.checks)}")
        return 0
    print(f"FAIL: {manifest_path}", file=sys.stderr)
    for error in report.errors:
        print(f"- {error}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            return _run_demo(args)
        if args.command == "run":
            return _run_custom(args)
        if args.command == "verify":
            return _run_verify(args)
    except (ArtifactError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"tft: error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
