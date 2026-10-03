"""Packaged training launchers work without a source checkout or a GPU."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from zvla.posttrain import __main__ as cli
from zvla.posttrain import train
from zvla.posttrain.recipes import BENCHMARKS


@pytest.mark.parametrize("entrypoint", ["module", "public-cli", "legacy-script"])
def test_training_help_and_module_import_need_no_training_or_simulator_dependencies(
    tmp_path, entrypoint
):
    package_root = Path(train.__file__).resolve().parents[2]
    legacy_script = Path(__file__).resolve().parents[1] / "scripts/train.py"
    script = """
import builtins
import runpy
import sys
sys.path.insert(0, sys.argv[1])
entrypoint, legacy_script = sys.argv[2:]
original_import = builtins.__import__
def import_without_training(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'mmengine', 'megatron', 'numpy', 'scipy',
                             'libero', 'metaworld', 'XPolicyLab', 'h5py', 'av'}:
        raise AssertionError('Help imported training/simulator dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_training
if entrypoint == 'module':
    from zvla.posttrain.train import main
    main(['--help'])
elif entrypoint == 'public-cli':
    sys.argv = ['zvla.posttrain', 'train', '--help']
    runpy.run_module('zvla.posttrain', run_name='__main__')
else:
    sys.argv = [legacy_script, '--help']
    runpy.run_path(legacy_script, run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(package_root), entrypoint, str(legacy_script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--resume" in result.stdout
    assert (
        "--config" in result.stdout
        if entrypoint != "public-cli"
        else "--checkpoint" in result.stdout
    )


def test_training_cfg_options_keep_mmengine_nested_value_semantics():
    pytest.importorskip("mmengine")
    args = train.parse_args(
        [
            "--config",
            "launch.json",
            "--work-dir",
            "run",
            "--resume",
            "--cfg-options",
            "training.optimizer.betas=(0.9,0.95)",
            "data.train.num_workers=2",
            "training.torch_compile=False",
            "training.metric.active_trackers=['jsonl']",
        ]
    )
    assert args.resume
    assert args.cfg_options == {
        "training.optimizer.betas": (0.9, 0.95),
        "data.train.num_workers": 2,
        "training.torch_compile": False,
        "training.metric.active_trackers": ["jsonl"],
    }


def test_repeated_cfg_options_merge_and_later_values_override():
    pytest.importorskip("mmengine")
    args = train.parse_args(
        [
            "--config",
            "launch.json",
            "--work-dir",
            "run",
            "--cfg-options",
            "training.torch_compile=False",
            "data.train.num_workers=2",
            "--resume",
            "--cfg-options",
            "data.train.num_workers=3",
        ]
    )
    assert args.resume
    assert args.cfg_options == {"training.torch_compile": False, "data.train.num_workers": 3}


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("nodes", [1, 2], ids=["one-node", "two-nodes"])
def test_public_cli_launches_installed_module_and_preserves_distributed_recipe(
    tmp_path, monkeypatch, benchmark, nodes
):
    fake_mmengine = ModuleType("mmengine")

    class Config:
        def __init__(self, value):
            self.value = value

        def dump(self, filename):
            Path(filename).write_text(json.dumps(self.value), encoding="utf-8")

    fake_mmengine.Config = Config
    monkeypatch.setitem(sys.modules, "mmengine", fake_mmengine)
    launches = []

    def capture_launcher(command, **kwargs):
        launches.append((command, kwargs))

    monkeypatch.setattr(cli.subprocess, "run", capture_launcher)
    output = tmp_path / "run with spaces"
    data = tmp_path / "prepared"
    checkpoint = tmp_path / "base with spaces.safetensors"
    arguments = [
        "zvla.posttrain",
        "train",
        "--bench",
        benchmark,
        "--data",
        str(data),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--gpus",
        "3",
        "--nodes",
        str(nodes),
        "--node-rank",
        "1",
        "--master-addr",
        "training-master.example.invalid",
        "--master-port",
        "30123",
        "--batch-size",
        "4",
        "--steps",
        "17",
        "--workers",
        "0",
        "--learning-rate",
        "0.00002",
    ]
    if nodes == 2:
        arguments.append("--resume")
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert len(launches) == 1
    command, options = launches[0]
    assert options == {"check": True}
    assert command[:5] == [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "3"]
    assert command[command.index("--module") + 1] == "zvla.posttrain.train"
    assert all(not item.endswith("scripts/train.py") for item in command)
    assert command[command.index("--work-dir") + 1] == str(output)
    assert ("--resume" in command) == (nodes == 2)
    if nodes == 1:
        assert "--standalone" in command and "--nnodes" not in command
    else:
        assert "--standalone" not in command
        for flag, value in (
            ("--nnodes", "2"),
            ("--node-rank", "1"),
            ("--master-addr", "training-master.example.invalid"),
            ("--master-port", "30123"),
        ):
            assert command[command.index(flag) + 1] == value
    config = json.loads(Path(command[command.index("--config") + 1]).read_text())
    assert config["model"]["pretrained_checkpoint"] == str(checkpoint)
    assert config["data"]["train"]["dataset"]["data_root_path"] == str(data)
    assert config["data"]["train"]["per_device_batch_size"] == 4
    assert config["data"]["train"]["num_workers"] == 0
    assert config["training"]["learning_rate"] == 0.00002
    assert config["training"]["max_steps"] == 17
    assert config["training"]["grad_accumulation_steps"] == 1
    assert config["training"]["checkpoint"]["export_model_steps"] == [17]
    if benchmark == "metaworld":
        assert config["data"]["train"]["dataset"]["sampler"]["steps"] == 17


def test_package_only_installation_can_resolve_and_launch_training_without_source_scripts(tmp_path):
    package_root = Path(train.__file__).resolve().parents[2]
    installed = tmp_path / "installed"
    for relative in (
        "zvla/__init__.py",
        "zvla/posttrain/__init__.py",
        "zvla/posttrain/__main__.py",
        "zvla/posttrain/recipes.py",
        "zvla/posttrain/train.py",
    ):
        target = installed / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package_root / relative, target)
    working = tmp_path / "empty-working-directory"
    working.mkdir()
    script = """
import builtins
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def import_without_training(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'megatron', 'numpy', 'libero', 'metaworld', 'XPolicyLab'}:
        raise AssertionError('Launcher imported training/simulator dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_training
from zvla.posttrain import train
assert Path(train.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert callable(train.main)
fake_mmengine = ModuleType('mmengine')
class Config:
    def __init__(self, value):
        self.value = value
    def dump(self, filename):
        Path(filename).write_text(json.dumps(self.value))
fake_mmengine.Config = Config
sys.modules['mmengine'] = fake_mmengine
commands = []
def capture_launcher(command, **kwargs):
    assert kwargs == {'check': True}
    commands.append(command)
subprocess.run = capture_launcher
from zvla.posttrain.__main__ import main
sys.argv = ['zvla.posttrain', 'train', '--bench', 'robotwin', '--data', 'data',
            '--checkpoint', 'base.safetensors', '--output', 'run']
main()
assert len(commands) == 1
print(json.dumps(commands[0]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(installed)],
        cwd=working,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)
    assert command[command.index("--module") + 1] == "zvla.posttrain.train"
    assert not (installed / "scripts").exists()
    assert not (working / "scripts").exists()
    assert (working / "run/launch-config.json").is_file()
