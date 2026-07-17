"""Optional diagnostic plotting for TFT Lab bundles."""

from __future__ import annotations

import io

import numpy as np

from .experiment import ExperimentResult


def render_comparison_png(result: ExperimentResult, *, dpi: int = 150) -> bytes:
    """Render an honest matrix/spectrum/audio comparison as PNG bytes.

    Matplotlib is an optional dependency because plots are observational and
    not part of the portable numerical replay contract.
    """

    if not isinstance(result, ExperimentResult):
        raise TypeError("result must be an ExperimentResult")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 600:
        raise ValueError("dpi must be an integer between 72 and 600")

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError('plotting requires the optional dependency: pip install ".[demo]"') from exc

    figure = Figure(figsize=(10.5, 7.0), dpi=dpi, constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 3)
    tensor_ax = figure.add_subplot(grid[0, 0])
    rotated_ax = figure.add_subplot(grid[0, 1])
    spectrum_ax = figure.add_subplot(grid[0, 2])
    waveform_ax = figure.add_subplot(grid[1, :2])
    receipt_ax = figure.add_subplot(grid[1, 2])

    bound = max(
        float(np.max(np.abs(result.tensor), initial=0.0)),
        float(np.max(np.abs(result.rotated_tensor), initial=0.0)),
        np.finfo(float).eps,
    )
    image_options = {"cmap": "coolwarm", "vmin": -bound, "vmax": bound, "interpolation": "nearest"}
    tensor_image = tensor_ax.imshow(result.tensor, **image_options)
    rotated_ax.imshow(result.rotated_tensor, **image_options)
    tensor_ax.set_title("Input tensor T")
    rotated_ax.set_title("Rotated tensor R T Rᵀ")
    for axis in (tensor_ax, rotated_ax):
        axis.set_xlabel("column")
        axis.set_ylabel("row")
    figure.colorbar(tensor_image, ax=[tensor_ax, rotated_ax], shrink=0.75, label="matrix value")

    indices = np.arange(result.config.dimension)
    spectrum_ax.plot(
        indices,
        result.initial_invariants["eigvals"],
        "o-",
        color="#e0a82e",
        linewidth=2,
        label="T",
    )
    spectrum_ax.plot(
        indices,
        result.rotated_invariants["eigvals"],
        "x--",
        color="#276d78",
        linewidth=1.5,
        label="R T Rᵀ",
    )
    spectrum_ax.set_title("Preserved eigen-spectrum")
    spectrum_ax.set_xlabel("sorted eigenvalue index")
    spectrum_ax.set_ylabel("value")
    spectrum_ax.grid(alpha=0.25)
    spectrum_ax.legend(frameon=False)

    preview_samples = min(result.stereo.shape[0], max(32, int(result.config.sample_rate * 0.025)))
    preview_time_ms = np.arange(preview_samples) * 1000.0 / result.config.sample_rate
    waveform_ax.plot(preview_time_ms, result.stereo[:preview_samples, 0], color="#222222", label="left")
    waveform_ax.plot(
        preview_time_ms,
        result.stereo[:preview_samples, 1],
        color="#b45b32",
        alpha=0.85,
        label="right quadrature partner",
    )
    waveform_ax.set_title("Sonification preview")
    waveform_ax.set_xlabel("time (ms)")
    waveform_ax.set_ylabel("amplitude")
    waveform_ax.grid(alpha=0.2)
    waveform_ax.legend(frameon=False, ncol=2)

    observations = result.observations()
    residuals = observations["matrix"]["residuals"]
    receipt_ax.axis("off")
    receipt_ax.set_title("Numerical receipt", loc="left")
    receipt_text = "\n".join(
        (
            f"status: {observations['status'].upper()}",
            f"dimension: {result.config.dimension}",
            f"seed: {result.config.seed}",
            f"det(R): {observations['matrix']['rotation_determinant']:.12g}",
            f"‖RᵀR−I‖F: {residuals['orthogonality_frobenius']:.3e}",
            f"max Δλ: {residuals['eigenvalues']['max_absolute']:.3e}",
            f"Δ‖T‖F rel: {residuals['frobenius_norm']['relative']:.3e}",
            "",
            "Mapped Hz are illustrative,",
            "not physical eigenfrequencies.",
        )
    )
    receipt_ax.text(0.0, 0.96, receipt_text, va="top", family="monospace", fontsize=10)

    figure.suptitle("TFT Lab 0.2 — Matrix Invariance & Quadrature Sonification", fontsize=14)
    buffer = io.BytesIO()
    try:
        canvas.print_png(
            buffer,
            metadata={"Software": "TFT Lab 0.2", "Description": "Noncanonical diagnostic plot"},
        )
        return buffer.getvalue()
    finally:
        figure.clear()
