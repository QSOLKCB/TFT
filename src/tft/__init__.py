"""TFT Lab: finite tensor-invariance and sonification experiments."""

from .experiment import ExperimentConfig, ExperimentResult, run_experiment
from ._version import __version__
from .resonance import (
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

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "analytic_signal",
    "invariant_residuals",
    "invariants",
    "map_to_audio",
    "phi_lock_pair",
    "quadrature_metrics",
    "rng",
    "rotate_rank2",
    "rotation_matrix",
    "run_experiment",
    "spd_matrix",
]
