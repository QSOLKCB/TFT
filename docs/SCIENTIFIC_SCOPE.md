# Scientific Scope

## Executable model

TFT Lab 0.2 operates on a finite real symmetric matrix

$$
T \in \mathbb{R}^{n \times n}, \qquad T = T^{\mathsf T},
$$

and a proper-orthogonal matrix

$$
R \in SO(n), \qquad R^{\mathsf T}R = I, \qquad \det(R)=+1.
$$

The implemented transformation is the orthogonal congruence

$$
T' = R T R^{\mathsf T}.
$$

Because $R^{-1}=R^{\mathsf T}$, this is also a similarity transformation.
In exact arithmetic, $T$ and $T'$ therefore have the same eigenvalues. The
Frobenius norm is also preserved:

$$
\lVert R T R^{\mathsf T} \rVert_F = \lVert T \rVert_F.
$$

The software evaluates these relations numerically and records the residuals and
tolerances used for each run. A passing run means that this finite computation
satisfied its recorded numerical checks. It does not prove a wider physical
theory.

## Inputs and validation

The built-in demonstration creates a seeded symmetric positive-definite matrix.
A user run can instead load a finite square matrix from disk.

By default, a seed produces a proper-orthogonal rotation of the same dimension.
A user-supplied rotation is accepted only when its shape is compatible and its
orthogonality and determinant satisfy the configured numerical checks.

The lab rejects non-finite values and incompatible shapes. A nonsymmetric input
is not silently projected onto its symmetric part: silent symmetrization would
change the object being analysed and obscure the provenance of the result.

All computations use finite-precision floating-point arithmetic. Consequently,
an invariant residual is expected to be small rather than identically zero. The
run records the tolerances that convert those residuals into a pass or fail.

The finite-only artifact contract also requires the ordinary determinant and
other recorded observations to be representable as float64 values. Extremely
large or small matrices whose determinant overflows or loses representability
are rejected; TFT Lab does not silently substitute an extended-range result.

## Sonification model

The sonification starts with the sorted eigenvalues

$$
\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n.
$$

Their minimum and maximum are mapped linearly into a declared audible frequency
interval. If the spectrum is degenerate, the values map to the midpoint of that
interval. Each cosine component is paired directly with a sine component at the
same frequency and initial phase, producing a $\pi/2$ quadrature relationship.

This construction has three important limits:

1. Min-max mapping is relative to the spectrum. It intentionally discards
   absolute scale and offset information.
2. The configured pitch interval and oscillator design are representational
   choices. They do not assign physical acoustic units to the eigenvalues.
3. “Quadrature” describes an operational signal-processing relationship. It is
   not, on its own, a mathematical proof of self-duality or a physical phase
   symmetry.

The term *φ-lock* may appear in the surrounding TFT research vocabulary. In this
software, the implemented quantity is specifically a phase offset of
$\pi/2$; it should not be confused with the golden ratio.

## Supported and unsupported claims

| Statement | Status in TFT Lab 0.2 |
|---|---|
| A seeded finite matrix and rotation can be replayed from stored inputs. | Implemented. |
| Orthogonal similarity preserves the recorded spectrum and Frobenius norm within tolerance. | Implemented numerical check. |
| The eigen-spectrum can drive a declared, deterministic sonification mapping. | Implemented representational mapping. |
| A change of basis is a time-evolution law. | Not claimed. |
| The lab solves a covariant tensor wave equation. | Not implemented. |
| The matrices constitute a tensor field over space-time. | Not implemented. |
| Quadrature audio proves self-duality or informational symmetry. | Not claimed. |
| A passing run validates a cosmological, quantum, or other physical model. | Not claimed. |
| A plot or WAV is experimental physical evidence. | Not claimed. |

## Theory and software

The philosophical statement

> Truth is not fixed; it is a resonance that remains invariant through
> transformation.

belongs to the broader TFT research motivation. TFT Lab converts one limited
part of that motivation into an inspectable numerical experiment: a matrix
changes representation while selected invariants remain stable.

The distinction is deliberate. The software can support exploration, teaching,
and reproducible computational examples without presenting unimplemented theory
as an executed result.

## Requirements for future field-dynamics claims

A later release should describe itself as a tensor-field dynamics solver only
after it provides, at minimum:

- a precisely specified field domain, tensor type, metric, coordinates, and
  boundary and initial conditions;
- a dimensional analysis and definition of every parameter;
- a documented discretization and integration scheme;
- stability, convergence, and error analyses;
- reference solutions or external validation data; and
- tests that distinguish numerical implementation from physical interpretation.

Those additions would be a new model layer. They should not be inferred from the
finite basis-rotation experiment in version 0.2.
