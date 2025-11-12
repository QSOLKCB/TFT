# Minimal resonance + phi-lock helpers for TFT
# Deterministic, numpy-first; scipy used if present.

from __future__ import annotations
import numpy as np

def rng(seed: int | None = 1337) -> np.random.Generator:
    return np.random.default_rng(seed)

def rotation_matrix(dim: int = 3, seed: int | None = 1337) -> np.ndarray:
    """Random orthogonal R (det=+1) via QR, seeded."""
    g = rng(seed)
    A = g.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def spd_matrix(dim: int = 3, seed: int | None = 1337) -> np.ndarray:
    """Symmetric positive-definite matrix A A^T, seeded."""
    g = rng(seed)
    A = g.normal(size=(dim, dim))
    return A @ A.T

def rotate_rank2(T: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Orthogonal congruence: T' = R T R^T (preserves eigvals, Fro norm)."""
    return R @ T @ R.T

def invariants(T: np.ndarray) -> dict:
    """Numerical invariants we care about for the demo."""
    # Symmetrize to be safe numerically
    S = 0.5 * (T + T.T)
    fro = np.linalg.norm(S, "fro")
    # eigvalsh: Hermitian (real symmetric) eigenvalues
    ev = np.linalg.eigvalsh(S)
    return {"fro": float(fro), "eigvals": np.sort(ev)}

def _analytic_signal_fft(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Hilbert analytic signal via FFT (no scipy required).
    Produces y = x + i*x_hilbert with 90° quadrature.
    """
    X = np.fft.fft(x, axis=axis)
    N = x.shape[axis]
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1
        H[N//2] = 1
        H[1:N//2] = 2
    else:
        H[0] = 1
        H[1:(N+1)//2] = 2
    shape = [1] * x.ndim
    shape[axis] = N
    H = H.reshape(shape)
    return np.fft.ifft(X * H, axis=axis)

def analytic_signal(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Try scipy.signal.hilbert; fall back to FFT implementation."""
    try:
        from scipy.signal import hilbert  # type: ignore
        return hilbert(x, axis=axis)
    except Exception:
        return _analytic_signal_fft(x, axis=axis)

def phi_lock_pair(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (real, imag) quadrature pair at 90° (φ = π/2).
    Energies normalized to match.
    """
    z = analytic_signal(x, axis=axis)
    a = np.real(z)
    b = np.imag(z)
    # Normalize energies to match (avoid trivial scale mismatch)
    ea = np.sqrt(np.mean(a**2))
    eb = np.sqrt(np.mean(b**2)) + 1e-12
    b *= (ea / eb)
    return a, b

def map_to_audio(eigvals: np.ndarray, fmin=220.0, fmax=880.0) -> np.ndarray:
    """
    Map sorted eigenvalues to audible frequencies, linearly.
    Handles degenerate cases gracefully.
    """
    v = np.real(eigvals)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if np.isclose(vmax, vmin):
        return np.full_like(v, (fmin + fmax) * 0.5, dtype=float)
    return fmin + (v - vmin) * (fmax - fmin) / (vmax - vmin)
