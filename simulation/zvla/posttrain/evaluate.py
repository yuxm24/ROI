"""Benchmark rollouts using the exported HTTP policy."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .client import PolicyClient

_ROBOTWIN_POLICY_FILES = ("deploy.py", "__init__.py", "deploy_policy.yml")
_ROBODOJO_POLICY_FILES = (
    "__init__.py",
    "action_adapter.py",
    "deploy.py",
    "model.py",
    "eval.sh",
    "setup_eval_env_client.sh",
    "setup_eval_policy_server.sh",
    "deploy.yml",
)
_ROBODOJO_LAUNCH_FILES = (
    "XPolicyLab/setup_policy_server.py",
    "XPolicyLab/utils/get_free_port.sh",
    "XPolicyLab/utils/wait_for_policy_server.sh",
    "XPolicyLab/utils/setup_env_client.sh",
)


def _libero_dependencies(bench):
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.transform_utils import quat2axisangle
    except ImportError as error:
        raise ImportError(
            f"{bench} evaluation requires its official LIBERO/robosuite environment. "
            "Install the selected benchmark's dependencies using its official setup "
            f"instructions in this environment; missing or incompatible import: {error}"
        ) from error
    return benchmark, get_libero_path, OffScreenRenderEnv, quat2axisangle


def _metaworld_dependency():
    try:
        import metaworld
    except ImportError as error:
        raise ImportError(
            "metaworld evaluation requires its official MetaWorld/MuJoCo environment. "
            "Install the selected benchmark's dependencies using its official setup "
            f"instructions in this environment; missing or incompatible import: {error}"
        ) from error
    return metaworld


def _benchmark_root(value, *, required=False):
    if value is None or value == "":
        if required:
            raise ValueError("Set --benchmark-root to the selected benchmark's upstream checkout")
        return None
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"--benchmark-root must be an existing upstream directory: {root}")
    return root


def _required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set {name} for the selected benchmark evaluation")
    return value


def _required_files(root, names, context):
    paths = tuple(root / name for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{context} is missing required launch files: {', '.join(missing)}")
    return paths


def libero(args):
    benchmark, get_libero_path, OffScreenRenderEnv, quat2axisangle = _libero_dependencies(
        args.bench
    )
    suite = benchmark.get_benchmark_dict()[args.suite]()
    horizon = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}[
        args.suite
    ]
    client = PolicyClient(args.server)
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
        )
        try:
            starts = suite.get_task_init_states(task_id)
            for episode in range(args.episodes):
                env.seed(args.seed + episode)
                env.reset()
                obs = env.set_init_state(starts[episode])
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                client.reset(task.language, args.seed + episode)
                frames, actions, success = [], [], False
                for step in range(horizon):  # noqa: B007 - report the final step after rollout
                    frames.append(
                        np.stack(
                            [
                                obs["agentview_image"][::-1, ::-1],
                                obs["robot0_eye_in_hand_image"][::-1, ::-1],
                            ]
                        )
                    )
                    if not len(actions):
                        state = np.r_[
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]
                        actions = list(client.infer(np.stack(frames), state))
                        frames = []
                    action = actions.pop(0)
                    # Model/native training convention: +1 closes, -1 opens.
                    action[6] = 1 if action[6] >= 0 else -1
                    obs, _, done, _ = env.step(action)
                    if done:
                        success = True
                        break
                yield dict(task=task_id, episode=episode, success=success, steps=step + 1)
        finally:
            env.close()


def metaworld(args):
    metaworld = _metaworld_dependency()
    client = PolicyClient(args.server)
    mt = metaworld.MT50(seed=args.seed)
    for name, cls in mt.train_classes.items():
        env = cls(render_mode="rgb_array", width=480, height=480, camera_name="corner2")
        goals = [task for task in mt.train_tasks if task.env_name == name]
        try:
            for episode in range(args.episodes):
                env.set_task(goals[episode])
                obs, _ = env.reset(seed=args.seed + episode)
                from metaworld_benchmark.prompts import PROMPTS

                prompt = PROMPTS[name]
                client.reset(prompt, args.seed + episode)
                frames, actions, success = [], [], False
                for step in range(500):  # noqa: B007 - report the final step after rollout
                    frames.append(env.render()[None])
                    if not len(actions):
                        actions = list(client.infer(np.stack(frames), obs[:4]))
                        frames = []
                    obs, _, terminated, truncated, info = env.step(actions.pop(0))
                    success |= bool(info["success"])
                    if terminated or truncated:
                        break
                yield dict(task=name, episode=episode, success=success, steps=step + 1)
        finally:
            env.close()


def evaluate(args):
    if args.bench not in ("libero", "libero-plus", "metaworld", "robotwin", "robodojo"):
        raise ValueError(f"Unsupported evaluation benchmark: {args.bench}")
    root = _benchmark_root(args.benchmark_root, required=args.bench in ("robotwin", "robodojo"))
    if args.bench in ("robotwin", "robodojo"):
        # The official harness owns scene generation, expert feasibility and scoring.
        task = _required_text(args.task, "--task")
        package = Path(__file__).resolve().parents[2]
        if args.bench == "robotwin":
            _required_files(root, ("script/eval_policy.py",), "RoboTwin upstream checkout")
            resources = _required_files(
                package / "robotwin/public_policy",
                _ROBOTWIN_POLICY_FILES,
                "Installed Zerith VA RoboTwin plugin",
            )
            target = root / "policy/zvla_public"
            cmd = [
                sys.executable,
                "script/eval_policy.py",
                "--config",
                "policy/zvla_public/deploy_policy.yml",
                "--overrides",
                "--task_name",
                task,
                "--task_config",
                "demo_clean",
                "--ckpt_setting",
                "zvla",
                "--seed",
                str(args.seed),
            ]
        else:
            policy_env = _required_text(os.environ.get("POLICY_ENV"), "POLICY_ENV")
            eval_env = _required_text(os.environ.get("EVAL_ENV"), "EVAL_ENV")
            _required_files(root, _ROBODOJO_LAUNCH_FILES, "RoboDojo/XPolicyLab upstream checkout")
            if shutil.which("bash") is None:
                raise ValueError("RoboDojo evaluation requires bash on PATH")
            resources = _required_files(
                package / "robodojo/xpolicylab_policy/ZVLA",
                _ROBODOJO_POLICY_FILES,
                "Installed Zerith VA RoboDojo plugin",
            )
            target = root / "XPolicyLab/policy/ZVLA"
            # Official XPolicyLab launch arguments are passed explicitly by the user.
            cmd = [
                "bash",
                "XPolicyLab/policy/ZVLA/eval.sh",
                "RoboDojo",
                task,
                "zvla",
                "arx_x5",
                "ee",
                str(args.seed),
                os.environ.get("POLICY_GPU", "0"),
                os.environ.get("ENV_GPU", "0"),
                policy_env,
                eval_env,
            ]
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        for path in resources:
            shutil.copy2(path, target / path.name)
        env = dict(
            os.environ,
            ZVLA_SERVER_URL=args.server,
            ZVLA_REPO_ROOT=str(package),
            PYTHONPATH=str(package) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )
        with (output / "evaluation.log").open("w") as log:
            subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return
    if root is not None:
        sys.path.insert(0, str(root))
    if args.bench == "metaworld":
        _metaworld_dependency()
    else:
        _libero_dependencies(args.bench)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rollout = metaworld if args.bench == "metaworld" else libero
    with (output / "episodes.jsonl").open("w") as handle:
        for row in rollout(args):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(row, flush=True)
