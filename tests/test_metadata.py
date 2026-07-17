import json
from pathlib import Path
import re

from tft import __version__


ROOT = Path(__file__).resolve().parents[1]


def test_project_version_metadata_stays_aligned():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))

    project_version = re.search(
        r"(?ms)^\[project\].*?^version\s*=\s*\"([^\"]+)\"",
        pyproject,
    )
    citation_version = re.search(r"(?m)^version:\s*([^\s]+)\s*$", citation)
    assert project_version is not None
    assert citation_version is not None
    assert project_version.group(1) == __version__
    assert citation_version.group(1) == __version__
    assert zenodo["version"] == __version__
