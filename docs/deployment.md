# H1 deployment

[Project overview](../README.md) · [简体中文](deployment_zh-CN.md)

Zerith VA runs inference on a GPU host and cameras, measured state, FK/IK and
SDK control on the H1 host. Install each role separately: the H1 host needs no
PyTorch or model weights, and the GPU host needs no robot SDK.

## 1. Install by host

The GPU host uses Linux, Python 3.10 or newer and a CUDA GPU with BF16 support.
From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ./deployment/shared ./deployment/inference
cd deployment
cp configs/server.example.json configs/server.local.json
cp configs/visual_assets.example.json configs/visual_assets.local.json
```

On the H1 host, select a Python environment matching the motion and camera SDK
binary ABI. From the repository root:

```bash
python -m pip install ./deployment/shared ./deployment/robot
cd deployment
cp configs/h1.example.json configs/h1.local.json
zwam robot
```

`zwam robot` defaults to an offline coordinate check. It opens neither the SDK
nor cameras and sends no motion commands.

## 2. Prepare assets and configuration

Obtain a compatible H1 checkpoint and its manifests, Wan2.2 VAE, matching
Wan/VGGT source and VGGT weights, and a frozen UMT5 text bank from the model
provider. Simulation benchmark exports use different codecs and cannot be
loaded directly as H1 policies. Install external source dependencies in the GPU
environment and keep the source revisions and preprocessing matched to the
checkpoint.

The example GPU layout, relative to `deployment/`, is:

```text
assets/
  policy/model.safetensors
  policy/manifest.json
  wan/Wan2.2_VAE.pth
  vggt/config.json
  vggt/model.safetensors
  text/tasks.json
  text/text_embeddings.npy
  text/text_manifest.json
external/
  Wan2.2/
  vggt/
```

A `zwam-h1-inference-v1` bundle can be placed at `assets/policy` directly. For a
`zerith-va-policy-action20-linear-g4d-teacher` export, prepare a new bundle:

```bash
zwam prepare --source /path/to/exported-policy --output assets/policy
```

The output directory must not already exist. Raw training weights alone lack
the deployment metadata. The architecture identifier is
`zerith-va-action20-linear-g4d-teacher`; the loader checks model keys, shapes,
normalization statistics and hashes. The text bank contains exact task strings
and finite float16 embeddings shaped `[len(tasks),64,4096]`. Set the H1 `prompt`
to one of those strings; adding a prompt requires its matching embedding.

Edit the local copies of the [server configuration](../deployment/configs/server.example.json),
[visual assets configuration](../deployment/configs/visual_assets.example.json)
and [H1 configuration](../deployment/configs/h1.example.json). All commands
below run from `deployment/`; configuration paths resolve from the current
working directory, not the JSON file's directory. Absolute asset paths are
also supported. Local configuration files, assets and external libraries are
excluded from Git.

| Configuration | Fields to set |
| --- | --- |
| `server.local.json` | `artifact`, `visual_assets`, `text_assets`, `device`, `bind`, `port` |
| `visual_assets.local.json` | `wan_source`, `wan_weights`, `vggt_source`, `vggt_weights`, source fingerprints and weight SHA-256 values |
| `h1.local.json` | `server_url`, exact `prompt`, SDK and planner paths, camera/service settings; keep `commissioned: false` during setup |

Use the provider's asset hashes. Source fingerprints cover the Python package
directories; compute them for the selected source checkouts with:

```bash
zwam source-hash external/Wan2.2/wan
zwam source-hash external/vggt/vggt
```

### H1 SDK and planner

Install the [Zerith public SDK](https://github.com/inFpZero/zerith_public_sdk)
version matching the robot firmware and Python ABI. The example configuration
selects `H1_SDK_1.3.9`, with `sdk_root` pointing to
`external/zerith_public_sdk/H1_SDK_1.3.9/h1_sdk_v1.3.9_python3.10` and
`camera_sdk_root` to its separate `camera_sdk_python` directory. The adapter
searches both `sdk_root` and `sdk_root/lib`. Install camera dependencies and
check camera and motor access with the SDK's examples.

The SDK supplies camera and motor interfaces. Install a compatible native IK
provider and the H1 URDF/collision assets separately. Set `planner_source` to
the provider's Python source directory and `planner_factory` to its
`module:function` entry point; the example uses `vla_source:build_planner`.
The native adapter expects `processors.fast_ik_backend` and
`fast_ik_cpp.FastIKSolver`.

The planner must expose finite control19 joint bounds (`lower`, `upper`), a
waist-fixed `arm` model, `arm_control_indices`, `root_from_body`, `eef_names`
and `body_eef_poses(q19)`. Hand poses must be returned in `body_yaw_link`, with
end-effector links ordered `left_end_effector_link`, `right_end_effector_link`.
Arm IK must not move the waist a second time.

## 3. Check and start

On the GPU host:

```bash
zwam doctor --role server --config configs/server.local.json --verify-assets
bash scripts/start_server.sh --config configs/server.local.json
```

The script runs the server in the foreground. You can also run
`zwam serve --config configs/server.local.json` directly and stop it with
Ctrl+C. On the H1 host, check the local configuration and server metadata:

```bash
zwam doctor --role h1 --config configs/h1.local.json
zwam robot --check-server --config configs/h1.local.json
```

`doctor` checks configuration and file placement without importing the SDK,
opening cameras or moving hardware. Non-loopback serving requires the same
`ZWAM_API_KEY` of at least 32 characters in both hosts' environments. Use an
SSH tunnel or TLS-protected connection, and keep keys out of JSON and command
arguments. Point `server_url` to the reachable server or local tunnel endpoint;
the examples use `http://127.0.0.1:55558`.

Before enabling control, replay checkpoint-matched observations, compare
decoded targets with FK/IK, and measure latency and memory for the selected
weights. On the target H1, check the SDK ABI, joint signs/units/zero positions,
URDF and hand offsets, camera order/orientation/timestamps, joint limits,
required services and physical emergency stop. Allow only one controller to
publish commands.

After on-site commissioning, set `commissioned: true` in `h1.local.json` and
start from an interactive terminal:

```bash
bash scripts/start_h1.sh --config configs/h1.local.json --execute
```

Type `EXECUTE` to permit SDK initialization, inspect the posture and camera
views, then press Enter to start inference. Initialization can move the robot.
An optional `--initial-joints /path/to/pose.npy` supplies a validated SDK21
startup pose; `--initial-duration` must be at least two seconds.

Stop the controller before the inference server. For processes started with
the scripts, run these on their respective hosts from another terminal:

```bash
bash scripts/stop_h1.sh
bash scripts/stop_server.sh
```

Ctrl+C also requests a normal exit. Normal SDK deinitialization can move
hardware; use the physical emergency stop for motion hazards. Start/stop
scripts track process identity under `.run/`; if the host lacks safe pidfd
signalling, stop with Ctrl+C in the owning terminal.

## Reference: coordinates and HTTP

Measured `state23` and absolute `commands` share this 23-value layout:

| Indices | Meaning | Representation |
| --- | --- | --- |
| `0:3`, `10:13` | Left/right hand position | Metres in `body_yaw_link` |
| `3:9`, `13:19` | Left/right hand rotation | First two rotation-matrix columns, flattened column by column |
| `9`, `19` | Left/right gripper | State range 0–1; command range 0–1.5 |
| `20:23` | Waist lift, pitch, yaw | Metres, radians, radians |

Checkpoint `normalization_stats.command_contract` must declare
`pose_frame: "body_yaw_link"`, nonempty `frame_evidence`, `position_unit: "m"`,
`waist_units: ["m","rad","rad"]` and `waist_joint_order` equal to
`["waist_joint1.position","waist_joint2.position","waist_joint3.position"]`.
Optional `waist_semantics` must be `["lift","pitch","yaw"]`. The loader
checks these declarations against the checkpoint statistics; editing labels
does not transform weights or coordinates. SDK21 order is left arm 7 +
gripper, right arm 7 + gripper, waist 3, head 2. Planner control19 order is
waist 3, left arm 7, right arm 7, head 2.

The protocol identifier is `zwam-h1-command23-http-npz-v3`. Use matching shared
client and inference packages. POST requests require
`X-ZWAM-Protocol: zwam-h1-command23-http-npz-v3`; configured API authentication
uses a separate bearer token.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Protocol metadata and available task strings |
| `POST /reset` | JSON `prompt` and integer `seed`; start/reset the owned episode |
| `POST /infer` | NPZ observation in, NPZ action chunk out |
| `POST /release` | Release episode ownership |

Inference input fields are `images` (RGB uint8 `[T,3,H,W,3]`), `state23`
(float32 `[23]`) and `valid23` (bool `[23]`). Camera order is head, left wrist,
right wrist (`cam_high`, `cam_left_wrist`, `cam_right_wrist` in the SDK).
The first request after reset uses one frame; later requests use four fresh
frames. Output is `commands` (float32 `[4,23]`) and `valid` (bool `[4,23]`):
the first four rows of a 16-row prediction. Invalid values must be zero;
object arrays, nonfinite values and oversized NPZ payloads are rejected.

### Sessions and recovery

The Python `PolicyClient` manages `X-Episode-Token` and `X-Sequence`. Each
service has one episode owner. Inference is never retried automatically:
after an ambiguous timeout, explicitly reset and send fresh observations.
If the reset response is lost, the same client retains its token; call
`reset(prompt, seed)` to recover or `close()` to release ownership. Ownership
has no expiry or automatic takeover.

The controller checks target changes, IK/FK residuals, joint steps, camera
freshness and synchronization before SDK publication. IK failure pauses for
operator inspection; Enter discards the old chunk and replans. A partial SDK
send latches a fault and cannot be cleared by pressing Enter. The nominal
action schedule is 30 Hz, with inference and IK adding latency. The native IK
elapsed-time check cannot interrupt a native call that hangs; software checks
and pauses do not replace the physical emergency stop.
