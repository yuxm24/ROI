"""Run H1 coordinate checks or robot control. Defaults to an offline check."""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

from ..client import PolicyClient
from ..config import read_json
from ..contract import H1_POSE_FRAME
from .mapping import control19, encode_state, rotation6d


def offline_check():
    sdk = np.zeros(21, np.float32)
    hands = np.array([[0.2, 0.2, 0, 0, 0, 0, 1], [0.2, -0.2, 0, 0, 0, 0, 1]])
    encoded = encode_state(hands, sdk)
    if not np.allclose(rotation6d(encoded[3:9]), np.eye(3)):
        raise AssertionError("Rotation contract failed")
    print("OFFLINE CHECK PASS: Command23 mapping only; no SDK, cameras or server opened.")


def operator_input(prompt):
    try:
        return input(prompt)
    except EOFError:
        logging.error(
            "Terminal input closed. Holding SDK connection; no automatic cleanup. Use physical controls or explicit SIGINT for normal exit."
        )
        while True:
            time.sleep(1)


def pause_after_error(error, read_input=operator_input):
    logging.error(
        "Paused; no new targets. SDK heartbeat retained. Not an emergency stop: %s", error
    )
    reply = read_input(
        "Inspect robot. Enter = discard chunk and replan; q = normal exit (may move): "
    )
    return reply.strip().lower() != "q"


def transition(env, target, planner, duration):
    target = np.asarray(target, float)
    q = control19(target)
    if (
        np.any(q < planner.lower)
        or np.any(q > planner.upper)
        or np.any(target[[7, 15]] < 0)
        or np.any(target[[7, 15]] > 1.5)
    ):
        raise ValueError("Initial target outside limits")
    start = env.state().astype(float)
    start[[7, 15]] *= 1.5
    if not np.isfinite(duration) or duration < 2:
        raise ValueError("Initial transition must be >=2 seconds")
    # Slow explicit initial transition only; model chunks are not filtered.
    count = int(duration * 30)
    begin = time.monotonic()
    for step in range(1, count + 1):
        alpha = step / count
        alpha = alpha * alpha * (3 - 2 * alpha)
        env.send(start + alpha * (target - start))
        time.sleep(max(0, begin + step / 30 - time.monotonic()))


def control_loop(env, adapter, policy, prompt, seed=0, read_input=operator_input):
    """Prepare/validate before any send; commit IK seed only after all sends succeed."""
    head = env.state()[19:21].copy()
    need_reset = True
    while True:
        try:
            env.ready()
            if need_reset:
                adapter.reset(env.state())
                policy.reset(prompt, seed)
                history = [env.images()]
                need_reset = False
            state = adapter.state(env.state())
            commands, validity = policy.infer(np.stack(history), state, np.ones(23, bool))
            history = []
            for command, valid in zip(commands, validity, strict=False):
                begin = time.monotonic()
                measured = env.state()
                target = adapter.prepare(command, valid, measured, head)
                env.send(target)
                adapter.accept(target)
                time.sleep(max(0, begin + 1 / 30 - time.monotonic()))
                history.append(env.images())
        except Exception as error:
            need_reset = True
            if not pause_after_error(error, read_input):
                break
            # A latched SDK partial send fault cannot be cleared by pressing Enter.
            # Other failures restart from fresh measured state and a new causal history.


def execute(config, args):
    from .adapter import Adapter, load_planner
    from .preflight import acquire_lock, check_services
    from .sdk import SDKEnvironment

    if config.get("commissioned") is not True:
        raise ValueError(
            "Commission this robot/URDF/SDK and set commissioned=true; see the Zerith-VA README, H1 deployment"
        )
    if not sys.stdin.isatty():
        raise ValueError("Hardware execution requires an attached interactive terminal")
    if config.get("frame") != H1_POSE_FRAME or config.get("eef_links") != [
        "left_end_effector_link",
        "right_end_effector_link",
    ]:
        raise ValueError("Explicit H1 frame and EEF link commissioning required")
    with acquire_lock(config["controller_lock"]):
        check_services(config)
        planner = load_planner(config)
        adapter = Adapter(planner)
        policy = PolicyClient(config["server_url"], config.get("timeout_s", 60))
        if config["prompt"] not in policy.metadata["tasks"]:
            raise ValueError("Prompt is not in the server frozen text bank")
        if (
            input(
                "SDK initialization CAN MOVE hardware. Clear workspace, E-stop ready. Type EXECUTE: "
            )
            != "EXECUTE"
        ):
            return
        env = SDKEnvironment(config)

        def interrupt(*_):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            try:
                env.initialize()
                if args.initial_joints:
                    target = np.load(args.initial_joints, allow_pickle=False)
                    transition(env, target, planner, args.initial_duration)
                operator_input(
                    "Robot initialized. Verify posture/camera view; Enter to start inference: "
                )
                control_loop(env, adapter, policy, config["prompt"], args.seed)
            except Exception as error:
                logging.exception("Startup failed; no automatic deinitialization")
                operator_input(
                    f"Inspect hardware: {error}\nEnter to request normal deinitialization (may move), or use physical E-stop: "
                )
            except KeyboardInterrupt:
                print("Operator requested normal exit; deinitialization may move the robot.")
        finally:
            signal.signal(signal.SIGTERM, previous)
            try:
                env.close()
            finally:
                try:
                    policy.close()
                except Exception:
                    logging.warning(
                        "Session release failed; inspect inference service before reconnecting"
                    )


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Start interactive hardware control")
    mode.add_argument(
        "--check-server", action="store_true", help="Check server metadata without opening the SDK"
    )
    parser.add_argument(
        "--config", default="configs/h1.example.json", help="H1 JSON configuration file"
    )
    parser.add_argument(
        "--initial-joints",
        type=Path,
        help="Optional validated SDK21 .npy pose for the startup transition",
    )
    parser.add_argument("--initial-duration", type=float, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.execute and not args.check_server:
        offline_check()
        return
    config = read_json(args.config)
    if args.check_server:
        client = PolicyClient(config["server_url"])
        print(json.dumps(client.metadata, indent=2))
        return
    execute(config, args)


if __name__ == "__main__":
    main()
