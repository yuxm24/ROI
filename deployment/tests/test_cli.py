"""CLI routing tests require neither a model bundle nor robot hardware."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from zwam import cli


@pytest.mark.parametrize("argv", [[], ["--help"]])
def test_top_level_help_does_not_import_backends(monkeypatch, capsys, argv):
    def unexpected_import(name):
        pytest.fail(f"Help imported a backend: {name}")

    monkeypatch.setattr(cli, "import_module", unexpected_import)
    if argv:
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
        assert error.value.code == 0
    else:
        cli.main(argv)
    output = capsys.readouterr().out
    for command in cli.COMMANDS:
        assert command in output


@pytest.mark.parametrize("command", cli.COMMANDS)
def test_routes_arguments_without_changing_them(monkeypatch, command):
    calls = []

    def load(name):
        assert name == cli.COMMANDS[command][0]
        return SimpleNamespace(main=lambda args, **kwargs: calls.append((args, kwargs)))

    monkeypatch.setattr(cli, "import_module", load)
    cli.main([command, "--config", "profile with spaces.json", "--help"])
    assert calls == [
        (["--config", "profile with spaces.json", "--help"], {"prog": f"zwam {command}"})
    ]


@pytest.mark.parametrize("command", cli.COMMANDS)
def test_subcommand_help_has_correct_usage(command, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--help"])
    assert error.value.code == 0
    assert f"usage: zwam {command}" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["unknown"], ["--execute"]])
def test_rejects_unknown_top_level_commands(argv):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


def test_python_module_robot_defaults_to_offline():
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            str(root / component / "src") for component in ("shared", "inference", "robot")
        ),
    )
    result = subprocess.run(
        [sys.executable, "-m", "zwam", "robot"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "OFFLINE CHECK PASS" in result.stdout
    assert "no SDK, cameras or server opened" in result.stdout
