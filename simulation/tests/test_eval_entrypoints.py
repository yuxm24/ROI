"""Installed launchers need only the selected benchmark environment."""

import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from zvla.posttrain import evaluate as entrypoint


def arguments(tmp_path, bench):
    return SimpleNamespace(
        bench=bench,
        benchmark_root=str(tmp_path / "upstream"),
        task="task-with-hyphens",
        server="http://127.0.0.1:8000",
        output=str(tmp_path / "results"),
        seed=17,
        suite="libero_spatial",
        episodes=1,
    )


def upstream(root, bench):
    root.mkdir(parents=True)
    names = ("script/eval_policy.py",) if bench == "robotwin" else entrypoint._ROBODOJO_LAUNCH_FILES
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic upstream launch entry; never executed\n")
    return names


@pytest.fixture
def client_environment(monkeypatch):
    monkeypatch.setenv("POLICY_ENV", "policy-client")
    monkeypatch.setenv("EVAL_ENV", "selected-simulator")
    monkeypatch.setenv("POLICY_GPU", "2")
    monkeypatch.setenv("ENV_GPU", "3")
    monkeypatch.setenv("PYTHONPATH", "inherited-python-path")
    monkeypatch.setenv("EVAL_ENV_TYPE", "debug")


@pytest.mark.parametrize("bench", ["robotwin", "robodojo"])
@pytest.mark.parametrize("installed_layout", [False, True], ids=["source", "installed-like"])
def test_external_launch_copies_every_policy_file_and_uses_selected_environment(
    tmp_path, monkeypatch, client_environment, bench, installed_layout
):
    args = arguments(tmp_path, bench)
    root = Path(args.benchmark_root)
    upstream(root, bench)
    package = Path(entrypoint.__file__).resolve().parents[2]
    relative = (
        "robotwin/public_policy" if bench == "robotwin" else "robodojo/xpolicylab_policy/ZVLA"
    )
    names = (
        entrypoint._ROBOTWIN_POLICY_FILES
        if bench == "robotwin"
        else entrypoint._ROBODOJO_POLICY_FILES
    )
    original = {name: (package / relative / name).read_bytes() for name in names}
    if installed_layout:
        installed = tmp_path / "site-packages"
        for name, content in original.items():
            path = installed / relative / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        monkeypatch.setattr(entrypoint, "__file__", str(installed / "zvla/posttrain/evaluate.py"))
        package = installed
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("synthetic official harness launch\n")

    monkeypatch.setattr(entrypoint.subprocess, "run", run)
    entrypoint.evaluate(args)
    assert len(calls) == 1
    command, keywords = calls[0]
    target = root / ("policy/zvla_public" if bench == "robotwin" else "XPolicyLab/policy/ZVLA")
    assert {path.name: path.read_bytes() for path in target.iterdir()} == original
    assert keywords["cwd"] == root
    assert keywords["check"] is True
    assert keywords["stderr"] == subprocess.STDOUT
    assert keywords["env"]["ZVLA_SERVER_URL"] == args.server
    assert keywords["env"]["ZVLA_REPO_ROOT"] == str(package)
    assert keywords["env"]["PYTHONPATH"] == str(package) + ":inherited-python-path"
    assert keywords["env"]["EVAL_ENV_TYPE"] == "debug"
    assert (
        Path(args.output) / "evaluation.log"
    ).read_text() == "synthetic official harness launch\n"
    if bench == "robotwin":
        assert command == [
            sys.executable,
            "script/eval_policy.py",
            "--config",
            "policy/zvla_public/deploy_policy.yml",
            "--overrides",
            "--task_name",
            args.task,
            "--task_config",
            "demo_clean",
            "--ckpt_setting",
            "zvla",
            "--seed",
            "17",
        ]
    else:
        assert command == [
            "bash",
            "XPolicyLab/policy/ZVLA/eval.sh",
            "RoboDojo",
            args.task,
            "zvla",
            "arx_x5",
            "ee",
            "17",
            "2",
            "3",
            "policy-client",
            "selected-simulator",
        ]


@pytest.mark.parametrize("bench", ["robotwin", "robodojo"])
@pytest.mark.parametrize("field,value", [("benchmark_root", None), ("task", None), ("task", " \t")])
def test_required_arguments_fail_before_copy_output_or_subprocess(
    tmp_path, monkeypatch, client_environment, bench, field, value
):
    args = arguments(tmp_path, bench)
    root = Path(args.benchmark_root)
    upstream(root, bench)
    setattr(args, field, value)
    monkeypatch.setattr(
        entrypoint.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("launched")
    )
    monkeypatch.setattr(entrypoint.shutil, "copy2", lambda *_args: pytest.fail("copied"))
    with pytest.raises(
        ValueError, match="--benchmark-root" if field == "benchmark_root" else "--task"
    ):
        entrypoint.evaluate(args)
    assert not Path(args.output).exists()
    assert not (root / "policy").exists() and not (root / "XPolicyLab/policy").exists()


@pytest.mark.parametrize("bench", ["robotwin", "robodojo"])
def test_missing_upstream_launch_script_fails_without_side_effects(
    tmp_path, monkeypatch, client_environment, bench
):
    args = arguments(tmp_path, bench)
    root = Path(args.benchmark_root)
    names = upstream(root, bench)
    (root / names[0]).unlink()
    monkeypatch.setattr(
        entrypoint.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("launched")
    )
    with pytest.raises(FileNotFoundError, match=names[0]):
        entrypoint.evaluate(args)
    assert not Path(args.output).exists()
    assert not (root / "policy").exists() and not (root / "XPolicyLab/policy").exists()


@pytest.mark.parametrize("name", ["POLICY_ENV", "EVAL_ENV"])
@pytest.mark.parametrize("value", [None, "", " \t"])
def test_robodojo_missing_environment_names_are_reported_early(
    tmp_path, monkeypatch, client_environment, name, value
):
    args = arguments(tmp_path, "robodojo")
    root = Path(args.benchmark_root)
    upstream(root, "robodojo")
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        entrypoint.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("launched")
    )
    with pytest.raises(ValueError, match=name):
        entrypoint.evaluate(args)
    assert not Path(args.output).exists() and not (root / "XPolicyLab/policy").exists()


@pytest.mark.parametrize("bench", ["robotwin", "robodojo"])
def test_missing_bundled_plugin_is_reported_before_any_copy(
    tmp_path, monkeypatch, client_environment, bench
):
    args = arguments(tmp_path, bench)
    root = Path(args.benchmark_root)
    upstream(root, bench)
    package = tmp_path / "incomplete-install"
    monkeypatch.setattr(entrypoint, "__file__", str(package / "zvla/posttrain/evaluate.py"))
    monkeypatch.setattr(entrypoint.shutil, "copy2", lambda *_args: pytest.fail("copied"))
    with pytest.raises(FileNotFoundError, match="Installed Z-VLA"):
        entrypoint.evaluate(args)
    assert not Path(args.output).exists()
    assert not (root / "policy").exists() and not (root / "XPolicyLab/policy").exists()


@pytest.mark.parametrize("bench", ["robotwin", "robodojo", "libero", "libero-plus", "metaworld"])
def test_isolated_client_entrypoints_import_only_the_selected_benchmark(
    tmp_path, client_environment, bench
):
    package = Path(entrypoint.__file__).resolve().parents[2]
    root = tmp_path / "upstream"
    if bench in ("robotwin", "robodojo"):
        upstream(root, bench)
    script = """
import builtins
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
blocked = {'libero', 'metaworld', 'robosuite', 'sapien', 'mujoco', 'torch', 'isaacsim', 'isaaclab', 'XPolicyLab'}
attempts = []
original_import = builtins.__import__
def client_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        attempts.append(name.split('.')[0])
        raise ModuleNotFoundError('Selected benchmark library is not installed', name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = client_import
from zvla.posttrain import evaluate as module
assert attempts == [], attempts
launches = []
module.subprocess.run = lambda *args, **kwargs: launches.append(args)
bench = sys.argv[3]
args = SimpleNamespace(bench=bench, benchmark_root=sys.argv[2] if bench in {'robotwin', 'robodojo'} else None,
    task='task-with-hyphens', server='http://127.0.0.1:8000', output=sys.argv[4], seed=0,
    suite='libero_spatial', episodes=1)
if bench in {'robotwin', 'robodojo'}:
    module.evaluate(args)
    assert len(launches) == 1 and attempts == [], (launches, attempts)
else:
    try:
        module.evaluate(args)
    except ImportError as error:
        assert bench in str(error) and 'official' in str(error), str(error)
    else:
        raise AssertionError('Missing selected simulator dependencies were accepted')
    assert attempts == ['metaworld' if bench == 'metaworld' else 'libero'], attempts
    assert launches == []
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(package),
            str(root),
            bench,
            str(tmp_path / "results"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_package_declares_every_plugin_resource_and_only_client_dependencies(monkeypatch):
    import setuptools

    # Build declarations live in the source tree, including when runtime
    # entrypoints are imported from an independently installed wheel.
    package = Path(__file__).resolve().parents[1]
    captured = {}
    monkeypatch.chdir(package)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(package / "setup.py"))
    assert "robotwin.public_policy" in captured["packages"]
    assert "robodojo.xpolicylab_policy" in captured["packages"]
    assert "robodojo.xpolicylab_policy.ZVLA" in captured["packages"]
    assert captured["package_data"]["robotwin.public_policy"] == ["deploy_policy.yml"]
    assert captured["package_data"]["robodojo.xpolicylab_policy.ZVLA"] == ["*.sh", "*.yml"]
    assert captured["install_requires"] == []
    assert captured["extras_require"] == {"client": ["numpy>=1.26", "requests>=2.32"]}
