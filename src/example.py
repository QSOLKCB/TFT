#!/usr/bin/env python3
"""
TFT demo: invariance + φ-lock → plot + WAV
- Builds SPD tensor T
- Rotates: T' = R T R^T
- Shows invariants unchanged
- Maps eigenvalues → audio freqs, synthesizes φ-locked stereo tone
- Saves: figures/phase_cube.png, figures/tensor_demo.wav
"""

from __future__ import annotations
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile  # lightweight, part of scipy

from tft.resonance import (
    spd_matrix, rotation_matrix, rotate_rank2, invariants,
    map_to_audio, phi_lock_pair, rng
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

def synth_stereo(freqs, sr=48_000, dur=2.0, seed=1337):
    """Sum of cos tones, right channel is φ=π/2 quadrature of left."""
    g = rng(seed)
    t = np.linspace(0.0, dur, int(sr*dur), endpoint=False)
    # Left: sum cos(2π f t) with tiny random phases for richness (seeded)
    phases = g.uniform(0, 2*np.pi, size=len(freqs))
    left = np.sum([np.cos(2*np.pi*f*t + p) for f, p in zip(freqs, phases)], axis=0)
    # Right: quadrature partner (φ-lock) using analytic signal
    _, right = phi_lock_pair(left)
    # Normalize to avoid clipping
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-9)
    left /= peak
    right /= peak
    stereo = np.stack([left, right], axis=-1).astype(np.float32)
    return 48_000, stereo

def plot_phase_cube(freqs, savepath):
    """
    Toy 'phase cube' visualization:
    - x: normalized freq index
    - y: frequency (Hz) normalized
    - z: time phase sweep (0..2π)
    """
    n = len(freqs)
    idx = np.linspace(0, 1, n)
    f_norm = (freqs - freqs.min()) / max(freqs.ptp(), 1e-9)
    phi = np.linspace(0, 2*np.pi, 400)
    X, Z = np.meshgrid(idx, phi)
    Y = np.interp(X, np.linspace(0,1,n), f_norm)
    # Simple quadrature surface
    V = np.cos(2*np.pi*Y*Z)  # arbitrary “vibration” field
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, V, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("Index (space)")
    ax.set_ylabel("Freq (conjugate)")
    ax.set_zlabel("Phase (time)")
    ax.set_title("TFT Phase Cube (toy)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=140)
    plt.close(fig)

def main():
    seed = int(os.environ.get("TFT_SEED", "1337"))
    dim = int(os.environ.get("TFT_DIM", "3"))

    T = spd_matrix(dim=dim, seed=seed)
    R = rotation_matrix(dim=dim, seed=seed + 1)
    T_rot = rotate_rank2(T, R)

    inv0 = invariants(T)
    inv1 = invariants(T_rot)

    print("=== TFT Invariants Demo ===")
    print(f"Frobenius norm  : {inv0['fro']:.8f}  ->  {inv1['fro']:.8f}")
    print(f"Eigenvalues     : {inv0['eigvals']}  ->  {inv1['eigvals']}")
    print("(Should match up to numeric noise.)")

    # Audio mapping
    freqs = map_to_audio(inv0["eigvals"], fmin=220.0, fmax=880.0)
    sr, stereo = synth_stereo(freqs, sr=48_000, dur=2.0, seed=seed)

    wav_out = FIGDIR / "tensor_demo.wav"
    wavfile.write(wav_out.as_posix(), sr, stereo)
    print(f"Wrote WAV: {wav_out}")

    png_out = FIGDIR / "phase_cube.png"
    plot_phase_cube(freqs, png_out.as_posix())
    print(f"Wrote figure: {png_out}")

if __name__ == "__main__":
    main()
