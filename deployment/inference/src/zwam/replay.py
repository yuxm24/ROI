"""Run inference on recorded camera images and robot states without hardware."""

import argparse
import time

import numpy as np

from .client import PolicyClient
from .contract import images_rgb, state23


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Trusted local .npz: images[N,3,H,W,3], states[N,23]"
    )
    parser.add_argument("--url", default="http://127.0.0.1:55558")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    with np.load(args.input, allow_pickle=False) as data:
        images, states = data["images"], data["states"]
    if args.steps < 1 or len(images) < 1 + 4 * (args.steps - 1) or len(states) != len(images):
        raise ValueError("Insufficient aligned 30Hz observation rows")
    policy = PolicyClient(args.url)
    try:
        policy.reset(args.prompt)
        for i in range(args.steps):
            index = 4 * i
            frames = images[:1] if i == 0 else images[index - 3 : index + 1]
            images_rgb(frames, 1 if i == 0 else 4)
            current = states[index]
            valid = np.ones(23, bool)
            state23(current, valid)
            begin = time.monotonic()
            commands, mask = policy.infer(frames, current, valid)
            print(
                f"step={i} round_trip_ms={(time.monotonic() - begin) * 1000:.1f} shape={commands.shape} all_valid={mask.all()}"
            )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
