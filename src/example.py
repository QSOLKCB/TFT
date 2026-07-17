#!/usr/bin/env python3
"""Backward-compatible shortcut for a complete TFT Lab example bundle."""

from __future__ import annotations

import os
from pathlib import Path

from tft.cli import main


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    output = os.environ.get("TFT_OUT", str(repository_root / "runs" / "example"))
    raise SystemExit(
        main(
            [
                "demo",
                "--out",
                output,
                "--seed",
                os.environ.get("TFT_SEED", "1337"),
                "--dim",
                os.environ.get("TFT_DIM", "3"),
                "--plot",
                "--force",
            ]
        )
    )
