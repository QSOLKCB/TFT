# TFT — Tensor Field Theory  
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange)](#)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.pending.svg)](https://doi.org/10.5281/zenodo.pending)

---

## Overview
**TFT (Tensor Field Theory)** is a QSOL research framework describing *self-dual φ-locked tensor dynamics* — where geometry, information, and sound evolve through invariant resonance.  
It extends the **UFT (Unified Field Framework)** into dynamics, providing a harmonic model of how invariant structures move, interact, and sustain coherence across space-time and frequency domains.

TFT formalizes **resonance as field evolution**, unifying:
- Tensor calculus (geometry)
- Fourier analysis (frequency)
- φ = π/2 self-duality (informational phase symmetry)

---

## Core Principle
> **Truth is not fixed; it is a resonance that remains invariant through transformation.**

TFT treats tensor fields as *resonant manifolds* rather than static quantities.  
Each tensor evolves under a φ-locked phase symmetry, ensuring informational orthogonality and self-duality:

\[
\nabla^\mu \nabla_\mu T_{ij\ldots} + (\phi + \psi)^2 T_{ij\ldots} = 0
\]

This general wave equation governs the dynamic balance of geometry and information.

---

## Project Structure
docs/ — theoretical papers and diagrams
src/ — simulation prototypes (tensor resonance, φ-lock dynamics)
figures/ — Tensor Wave Equation, φ-Lock Symmetry, Fourier Resonance Manifold
LICENSE — CC-BY-4.0
README.md — project overview
zenodo.json — publication metadata

yaml
Copy code

---

![Tensor Resonance Phase Cube](figures/phase_cube.png)

---

## Quickstart
Clone and run a minimal TFT simulation.

```bash
git clone https://github.com/QSOLKCB/TFT.git
cd TFT
pip install -r requirements.txt
python src/example.py
Optional parameters:

bash
Copy code
TFT_SEED=2025 TFT_DIM=4 python src/example.py
Requirements
Python 3.10 +

numpy ≥ 1.26

scipy ≥ 1.11

matplotlib ≥ 3.8

(dev) pytest ≥ 7.4

Citation
Slade, T. (2025). Tensor Field Theory: From Invariance to Dynamics.
Zenodo. https://doi.org/10.5281/zenodo.pending

(Update DOI after tagging your release.)

License
This work is licensed under the
Creative Commons Attribution 4.0 International License (CC BY 4.0).

Part of the QSOL Research Series — QEC → UFT → TFT.

yaml
Copy code
