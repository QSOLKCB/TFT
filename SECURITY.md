# Security Policy

## Supported versions

TFT Lab is alpha research software. Security fixes are applied to the current
development branch and most recent release only.

| Version | Supported |
|---|---|
| 0.2.x | Yes |
| Earlier versions | No |

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting feature for this repository:

<https://github.com/QSOLKCB/TFT/security/advisories/new>

Include the affected version or commit, reproduction steps, likely impact, and
any suggested mitigation. Do not include real personal, confidential, or
regulated data in a report or test artifact.

If private reporting is unavailable, open a minimal issue asking the maintainer
for a private contact channel without disclosing exploit details.

## Scope and safety boundary

TFT Lab processes local numerical inputs and writes local experiment artifacts.
It is not cryptographic software, a security control, or a validated basis for
safety-critical, clinical, financial, or engineering decisions. Treat imported
files and third-party artifacts as untrusted, and inspect them before use.
