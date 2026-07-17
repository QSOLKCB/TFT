# Contributing to TFT Lab

Thank you for helping improve TFT Lab. This repository is an alpha research
prototype, so small, reviewable changes with explicit numerical evidence are
preferred.

## Set up a development environment

TFT Lab supports Python 3.10 through 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Run the checks before opening a pull request:

```bash
python -m pytest
tft --help
```

When changing CLI or artifact behavior, also run a small experiment and verify
its receipt using the commands documented by `tft --help`.

## Contribution expectations

- Keep theoretical or interpretive claims separate from measured numerical
  observations.
- State the supported claim boundary in documentation and user-facing output.
- Preserve deterministic behavior for identical inputs within the documented
  runtime contract. If a change intentionally alters results, explain why and
  update the artifact schema or algorithm version.
- Add or update tests for numerical changes, invalid inputs, CLI behavior, and
  artifact verification as applicable.
- Do not commit generated WAV, image, cache, build, or environment files unless
  they are intentional, documented reference fixtures.
- Avoid unrelated dependencies. NumPy is the core runtime dependency;
  visualization and development tools belong in optional dependency groups.

## Reporting numerical changes

For changes that affect calculations, include:

1. The command, seed, dimensions, and input data used.
2. Python, NumPy, and platform versions.
3. The expected invariant or observation.
4. Before-and-after output, including tolerances.
5. Any effect on deterministic replay or verification.

## Pull requests

Keep each pull request focused. Describe the problem, the chosen solution, the
tests performed, and any compatibility or claim-boundary implications. CI must
pass on all supported Python versions before merge.

By contributing, you agree that your contribution is made available under the
repository's Creative Commons Attribution 4.0 International license.
