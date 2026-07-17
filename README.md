# TFT Lab 0.2

## Reproducible Tensor Invariance & Quadrature Sonification

[![Status: alpha](https://img.shields.io/badge/status-alpha-orange)](#project-status)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#quickstart)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey)](#license)

> **Truth is not fixed; it is a resonance that remains invariant through transformation.**

That sentence is the philosophical motivation for the wider Tensor Field Theory
(TFT) research programme. **TFT Lab makes a deliberately narrower, testable
software claim:** it studies finite, real, symmetric rank-2 matrices under
orthogonal changes of basis and turns their eigenvalue spectra into declared,
illustrative audio mappings.

The lab can generate or load a tensor, apply a reproducible proper-orthogonal
rotation, measure what remains invariant, render a quadrature stereo WAV, and
write the inputs and observations needed to inspect or replay the run.

## What the lab does

For a real symmetric matrix $T \in \mathbb{R}^{n \times n}$ and a proper
orthogonal matrix $R \in SO(n)$, TFT Lab applies

$$
T' = R T R^{\mathsf T}.
$$

It then checks, within recorded floating-point tolerances, that the transformation
preserves:

- the sorted eigenvalue spectrum; and
- the Frobenius norm.

The sorted eigenvalues are normalized into a configured frequency interval. The
resulting oscillator mixture is paired with a $\pi/2$ quadrature channel and
exported as audio. This mapping is deterministic for a recorded recipe and
environment, but it is a sonification convention—not a claim that matrix
eigenvalues possess physical audio frequencies.

## Quickstart

TFT Lab requires Python 3.10 or newer.

```bash
git clone https://github.com/QSOLKCB/TFT.git
cd TFT
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Generate the built-in seeded demonstration:

```bash
tft demo --out runs/demo
```

To include the optional visual comparison, install the plotting extra and request
the plot:

```bash
python -m pip install -e ".[demo]"
tft demo --plot --out runs/demo-with-plot
```

Verify its manifest, file integrity, and numerical observations:

```bash
tft verify runs/demo/manifest.json
```

The output directory is created if needed. Existing files are not silently
treated as evidence for a new run; use a separate directory when comparing
recipes.

## Run a custom tensor

Provide a finite, square, symmetric matrix as CSV:

```text
2.0,0.5,0.0
0.5,1.5,0.25
0.0,0.25,1.0
```

Then run:

```bash
tft run --tensor tensor.csv --out runs/custom
```

If no rotation is supplied, the recorded seed is used to generate a matrix in
$SO(n)$. A custom compatible rotation can instead be supplied explicitly:

```bash
tft run --tensor tensor.csv --rotation rotation.csv --out runs/custom-rotation
```

Use `--seed` when you want to choose the generated rotation recipe rather than
use the default:

```bash
tft run --tensor tensor.csv --seed 2025 --out runs/seed-2025
```

Invalid dimensions, non-finite values, nonsymmetric tensors, and rotations that
fail the algorithm's validation tolerances are rejected rather than silently
repaired. Those fixed validation tolerances are recorded separately from the
run's configurable invariant-comparison tolerance.

## Run artifacts

Each completed run is an inspectable directory rather than just a WAV file.

| Artifact | Purpose |
|---|---|
| `recipe.json` | Versioned operation, source, seed, dimensions, mapping parameters, tolerances, and runtime choices. |
| `observations.json` | Norms, eigenvalues, residuals, rotation checks, and numerical pass/fail observations. |
| `tensor.npy` | Exact finite matrix used as the input tensor. |
| `rotation.npy` | Exact proper-orthogonal matrix used by the run. |
| `rotated_tensor.npy` | Result of $R T R^{\mathsf T}$. |
| `sonification.wav` | Stereo PCM16 eigen-spectrum sonification with a quadrature partner. |
| `comparison.png` | Optional visual comparison when plotting support is enabled. |
| `manifest.json` | Artifact inventory, SHA-256 integrity hashes, environment information, result, and claim boundary. |

See [Artifact format](docs/ARTIFACT_FORMAT.md) for the bundle contract.

## Reproducibility boundary

TFT Lab reports two different questions separately:

1. **Artifact integrity:** does each file still match the exact SHA-256 hash in
   the manifest?
2. **Numerical replay:** when the recorded matrices are recomputed, do the
   observed invariants and residuals remain within the recorded tolerances?

An exact hash is evidence that a stored artifact has not changed. It is not, by
itself, evidence that every Python, NumPy, BLAS, libm, audio, or plotting stack
will independently generate identical bytes. The manifest therefore records the
runtime environment, and the exact `.npy` matrices are retained instead of
assuming a seed is a universal cross-platform identity.

Byte-for-byte reproduction is expected only in a suitably pinned environment.
Numerical replay is the portable contract. PNG and WAV hashes protect those
derivative files; the decoded WAV must also remain within two PCM16 levels of a
fresh synthesis. The matrix observations and tolerances carry the mathematical
verification claim.

See [Reproducibility](docs/REPRODUCIBILITY.md) for details.

## Scientific claim boundary

TFT Lab 0.2 implements:

- finite-dimensional real symmetric rank-2 matrices;
- seeded or supplied proper-orthogonal basis rotations;
- numerical checks of eigenvalue and Frobenius-norm invariance;
- a declared min-max eigenvalue-to-pitch mapping; and
- a quadrature audio construction.

It does **not** implement or validate:

- a tensor field over space-time;
- covariant derivatives or a tensor wave PDE;
- time evolution, interactions, or field dynamics;
- a physical, cosmological, or quantum model;
- a proof of self-duality or informational orthogonality; or
- experimental confirmation of the broader TFT framework.

The generated rotation changes the coordinate representation of the matrix; it
does not represent physical time evolution. The plot and audio are analytical
aids, not physical measurements. See [Scientific scope](docs/SCIENTIFIC_SCOPE.md)
for the operational model and limitations.

## Development

Install the development dependencies and run the test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The tests cover rotation construction, invariant preservation, quadrature
behaviour, invalid inputs, audio structure, artifact creation, and manifest
verification.

One run may contain at most 6,000,000 audio frames, keeping the renderer's
memory use bounded. Shorter durations or lower sample rates can be selected for
larger experiment batches.

## Project status

TFT Lab 0.2 is alpha research software. Its artifact schema and command-line
interface may evolve before a stable 1.0 release. Runs retain explicit schema
and algorithm identifiers so that later tools can reject or migrate incompatible
formats instead of guessing.

## Citation

Until a separate publication record is independently verified, cite the
repository:

> Slade, Trent. (2026). *TFT Lab 0.2: Reproducible Tensor Invariance &
> Quadrature Sonification* [Computer software]. QSOL-IMC.
> https://github.com/QSOLKCB/TFT

## License

This project is licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0).

TFT is part of the QSOL research series: QEC → UFT → TFT.
