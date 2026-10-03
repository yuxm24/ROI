import re
from pathlib import Path

from zwam import __version__


def test_package_and_cli_versions_match():
    project = (Path(__file__).resolve().parents[1] / "shared/pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"$', project, re.MULTILINE)[1] == __version__
