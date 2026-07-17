# Reproducibility

## Two replay contracts

TFT Lab distinguishes exact artifact replay from numerical replay.

| Contract | Question | Evidence |
|---|---|---|
| Artifact integrity | Are these the same stored bytes? | File size and SHA-256 from `manifest.json`. |
| Numerical replay | Does recomputation preserve the declared invariants within tolerance? | Stored matrices, recipe, observations, and recomputed residuals. |

These contracts complement each other; neither should be silently substituted
for the other.

## Exact artifact integrity

SHA-256 digests identify the bytes placed in a run directory. A matching digest
means the checked file matches the file inventoried by that manifest. It does not
show that the underlying model is physically true, that the original author is
known, or that an independent environment will synthesize identical bytes.

Floating-point linear algebra and trigonometry, WAV encoders, fonts, plotting
backends, and metadata writers can vary between dependency or platform versions.
Exact regeneration therefore requires a suitably pinned software environment in
addition to the recipe.

## Numerical replay

Portable verification starts from `tensor.npy` and `rotation.npy`, not from the
seed alone. The verifier recomputes the transformed tensor and invariant
observations, then applies the tolerances recorded in `recipe.json`.

The stored matrices are important because a seed specifies an algorithmic
recipe, not an eternal byte sequence. Random-number generators, QR conventions,
BLAS libraries, and floating-point implementations can differ or change.

Numerical replay is successful only when all required validation and invariant
checks pass. A verifier must not relax tolerances after seeing the result.

## Environment record

The manifest records the software context relevant to interpretation and replay,
including the algorithm identifier, producing TFT Lab version, Python
implementation and version, NumPy version, and platform description. A release
or archive should retain its TFT Lab source revision and dependency lock
alongside the run.

The environment record serves two purposes:

- it helps reproduce exact derivatives in a matching environment; and
- it helps diagnose small numerical differences in a portable replay.

It is contextual metadata, not part of the mathematical invariant itself.

## Deterministic generation

For a fixed implementation and environment, the same valid recipe is intended to
produce the same generated matrices, oscillator phases, and audio samples. A
seeded run records both the seed and the generated arrays.

Determinism does not make a random example representative of a physical system.
The built-in seeded tensor is a reproducible test and demonstration input.

## Derivative artifacts

`sonification.wav` and optional `comparison.png` are deterministic targets in a
pinned environment, but portable numerical verification does not require their
bytes to be independently regenerated exactly. The verifier does require the
decoded WAV to remain within two PCM16 levels of a fresh synthesis; the PNG
remains a hash-bound, noncanonical visual derivative.

Their manifest hashes still matter: they tell a recipient whether the delivered
media are the files that belong to the bundle. The scientific observation remains
the explicitly recorded matrix calculation.

## Recommended archival practice

For a durable public run:

1. Keep the complete run directory, not the WAV alone.
2. Preserve `manifest.json` beside every artifact it inventories.
3. Record the source revision or release tag and dependency lock used to produce
   the run.
4. Verify the bundle before upload and after download.
5. Report both integrity and numerical verification outcomes.
6. Do not replace files inside a published bundle without issuing a new manifest
   and version.

## Interpreting a successful verification

A successful verification supports this statement:

> The stored finite matrices satisfy the declared orthogonal-transformation and
> invariant checks within the recorded tolerances, and the inventoried artifacts
> match their stored integrity hashes.

It does not validate a tensor-field PDE, space-time dynamics, quantum mechanics,
self-duality, or a broader physical interpretation. Those boundaries are detailed
in [Scientific scope](SCIENTIFIC_SCOPE.md).
