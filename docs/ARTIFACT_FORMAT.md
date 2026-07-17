# TFT Run Artifact Format

## Purpose

A TFT Lab run is a small evidence bundle. It keeps the exact matrices used by the
calculation alongside the recipe, numerical observations, derivative media, and
integrity manifest.

The format is designed to answer four separate questions:

1. What operation was requested?
2. Which exact finite arrays were used?
3. What did the software observe?
4. Have the stored files changed since the manifest was written?

## Directory layout

```text
run-directory/
├── recipe.json
├── observations.json
├── tensor.npy
├── rotation.npy
├── rotated_tensor.npy
├── sonification.wav
├── comparison.png        # optional
└── manifest.json
```

`comparison.png` is optional. Its absence is valid when plotting support is not
installed or visual output was not requested. The numerical and integrity
contract does not depend on the plot.

## Artifact roles

### `recipe.json`

The recipe describes the requested computation rather than its measured result.
It records enough context to interpret the stored arrays and derivatives,
including:

- artifact-schema and algorithm identifiers;
- producing software name and version;
- generated-input seed and source-input descriptions;
- matrix dimension;
- rotation source and seed when generated;
- sample rate, duration, frequency interval, peak, and fade parameters;
- numerical tolerances;
- the fixed symmetry and orthogonality validation tolerances;
- required output paths; and
- the exact-byte and numerical replay policies.

Observations and output hashes do not belong in the recipe. Keeping inputs and
results separate makes the computational direction clear.

### `observations.json`

The observation record contains derived numerical facts, including:

- validation results for the tensor and rotation;
- Frobenius norms before and after transformation;
- sorted eigenvalues before and after transformation;
- absolute and relative invariant residuals evaluated with the recorded
  combined absolute-plus-relative rule;
- orthogonality residual and rotation determinant; and
- the pass/fail result under the recipe's tolerances.

Values must be finite JSON numbers. A failed numerical check must remain visible
as a failure; it must not be converted into a successful manifest merely because
files were written.

### Matrix arrays

`tensor.npy`, `rotation.npy`, and `rotated_tensor.npy` retain the exact arrays
used and produced by the run as canonical NPY 1.0, C-contiguous,
little-endian float64 NumPy arrays.

The stored arrays are authoritative replay inputs. A seed remains valuable as a
recipe, but it is not treated as a universal binary identity: random-number,
linear-algebra, and platform implementations can evolve.

The transformation represented by the files is

$$
\texttt{rotated\_tensor} =
\texttt{rotation}\;\texttt{tensor}\;\texttt{rotation}^{\mathsf T}.
$$

### `sonification.wav`

The WAV is a stereo little-endian PCM16 derivative of the observed
eigen-spectrum and the audio parameters in the recipe. Its hash protects the
stored bytes. The mathematical verification does not depend on a media player,
codec interpretation, or subjective listening result.

Verification also compares its decoded samples with a fresh recipe replay. A
maximum difference of two PCM16 levels is allowed to accommodate negligible
floating-point/libm variation without accepting unrelated audio.

### `comparison.png`

When present, the comparison image is a noncanonical visual derivative of the
run. It can show the input and rotated matrices, their overlaid sorted spectra,
and the measured residuals. It is an analytical aid, not a substitute for
`observations.json`.

### `manifest.json`

The manifest closes the bundle. It inventories the artifacts and records, for
each file covered by the manifest:

- relative path;
- media type or artifact role;
- byte size; and
- SHA-256 digest.

It also identifies the schema and algorithm, records the producing TFT Lab,
Python, and NumPy versions plus platform context, and repeats the run result and
claim boundary. The recipe contains the replay policy. Paths are relative to the
run directory so that the directory can be moved without changing its logical
contents.

The manifest does not hash itself. Its own distribution integrity must be
provided by the surrounding release, archive, or version-control system.

## JSON encoding

Machine-readable JSON artifacts use UTF-8, finite values, stable key ordering,
compact separators, and a trailing newline. Non-finite spellings such as `NaN`
and `Infinity` are invalid.

The format includes an explicit schema identifier. A verifier must reject an
unknown incompatible schema rather than silently interpret it as the current
format.

## Verification procedure

The command

```bash
tft verify runs/demo/manifest.json
```

performs two logically separate checks.

### 1. Integrity verification

The verifier resolves only the manifest's safe relative paths beneath the run
directory, then compares each stored size and SHA-256 digest. Missing, changed,
unexpectedly relocated, or path-escaping entries fail integrity verification.

### 2. Numerical verification

The verifier first inspects the canonical NPY headers and bounded file sizes,
then loads the recorded matrices, confirms their shapes and finite values,
checks the rotation, recomputes $R T R^{\mathsf T}$, compares the complete
observation record using the recorded tolerances, and numerically checks the
PCM16 sonification.

Integrity can pass while numerical verification fails—for example, if a bundle
was internally generated by faulty software and then hashed consistently.
Numerical verification can pass after a derivative image is regenerated while
integrity fails because the PNG bytes changed. The command reports these outcomes
separately.

## Identity and derivatives

The identity-bearing mathematical content of a run is the recipe plus the exact
matrix inputs and the declared tolerance contract. `observations.json` records
what was measured from those inputs. WAV and PNG files are inspectable
derivatives.

This boundary prevents a waveform or image hash from being mistaken for proof of
the matrix relation. It also permits optional presentation artifacts to evolve
without silently changing the underlying numerical experiment.

## Path and mutation rules

- Artifact paths are relative and cannot escape the run directory.
- A verifier reads a bundle; it does not repair or rewrite it.
- A new calculation should use a new output directory or an explicitly approved
  replacement workflow.
- Verification never treats an unlisted extra file as evidence for the run.
- Hashes establish byte integrity, not authorship, physical validity, or
  publication priority.
