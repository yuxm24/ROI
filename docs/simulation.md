# Simulation post-training

[Project overview](../README.md) · [简体中文](simulation_zh-CN.md)

Zerith VA supports post-training on `libero`, `libero-plus`, `metaworld`,
`robotwin` and `robodojo`. The workflow prepares visual caches and a
benchmark-specific state/action codec, trains with Megatron-FSDP, and exports
an HTTP policy for simulator evaluation.

The base checkpoint, encoder weights, demonstrations and simulator assets
are obtained separately. Simulation artifacts use the selected benchmark's
codec; H1 deployment requires a compatible H1 checkpoint. See the
[deployment guide](deployment.md) for that workflow.

## Environment

Start from the repository root and enter `simulation/`. Unless a section
states otherwise, all commands below run in this directory.

```bash
cd simulation
conda create -n zvla-train python=3.12 pip
conda activate zvla-train
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install megatron-core==0.19.0
python -m pip install -r requirements.txt
```

This environment runs preparation, training and model serving. Use a GPU
driver compatible with the CUDA 13.0 PyTorch build. The project runs directly
from the source directory and needs no package installation.

Install each selected simulator in its own environment using its official
setup instructions. LIBERO and LIBERO-Plus use their respective checkouts;
MetaWorld needs its MuJoCo environment; RoboTwin and RoboDojo use their
official evaluation harnesses. Simulator clients need NumPy and requests.
Simulator dependencies are loaded only for the selected benchmark.

An optional environment for the HTTP client and launcher is provided:

```bash
conda env create -n zvla-client -f environment-client.yml
```

It contains Python 3.11, NumPy and requests. For RoboDojo, add the required
XPolicyLab dependencies before using it as `POLICY_ENV`; the simulator runs
in a separate `EVAL_ENV`.

## Required assets

Use the following source and weight snapshots for Wan VAE/UMT5 and VGGT.
Preparation and serving must use matching weights, tokenizer and
preprocessing.

| Asset | Source | Revision |
| --- | --- | --- |
| Wan source | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` |
| Wan weights and tokenizer | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` |
| VGGT source | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | `a288dd0f14786c93483e45524328726ab7b1b4ce` |
| VGGT weights | [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | `860abec7937da0a4c03c41d3c269c366e82abdf9` |

```bash
mkdir -p assets checkpoints
git clone https://github.com/Wan-Video/Wan2.2.git assets/Wan2.2
git -C assets/Wan2.2 checkout 42bf4cfaa384bc21833865abc2f9e6c0e67233dc
git clone https://github.com/facebookresearch/vggt.git assets/vggt
git -C assets/vggt checkout a288dd0f14786c93483e45524328726ab7b1b4ce
hf download Wan-AI/Wan2.2-TI2V-5B \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir assets/Wan2.2-TI2V-5B
hf download facebook/VGGT-1B \
  --revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --local-dir assets/VGGT-1B
```

Place your separately obtained base checkpoint at
`checkpoints/zvla-base.safetensors`, or pass its location with `--checkpoint`.
The loader initializes the neural modules from this checkpoint and uses the
target benchmark's `codec.json` produced during preparation.

## Demonstrations and preparation

| Benchmark | Demonstration source and revision | `--source` points to |
| --- | --- | --- |
| `libero` | [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds), `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551` | Root containing the four RLDS suites |
| `libero-plus` | [Sylvest/libero_plus_lerobot](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot), `22c57433fef692b5b9ecc0795344daac7fa867a5` | LeRobot root with `meta/info.json`; preparation reads the train split |
| `metaworld` | [MINT-SJTU/Evo1_MetaWorld_Dataset](https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset), `612c4e83d0e3b4e0be5227970bd8f5f07fc0c488` | LeRobot root with `meta/info.json` |
| `robotwin` | [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0), `981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042` | Root with `dataset/<task>/demo_clean.zip` |
| `robodojo` | [RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo), `91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd` | Extracted `ee_lerobot_v30` with `meta/info.json` |

The MetaWorld dataset is provided by a third party. RoboTwin preparation
uses EEF commands from the official `demo_clean` HDF5 files; RoboDojo uses
the dual-arm EEF LeRobot v3.0 data.

For example, download and prepare LIBERO-Plus:

```bash
hf download Sylvest/libero_plus_lerobot --repo-type dataset \
  --revision 22c57433fef692b5b9ecc0795344daac7fa867a5 --local-dir data/raw/libero-plus
python -m zvla.posttrain prepare --bench libero-plus \
  --source data/raw/libero-plus --output data/libero-plus \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0
```

Use the same command for other benchmarks with the corresponding `--bench`,
`--source` and output directory. To inspect a small preparation run, add
`--limit 2` and use a separate output such as `data/libero-plus-smoke`.

Preparation uses one GPU and processes episodes sequentially. It writes
`dataset.json`, `codec.json`, `tasks.json`, `text_embeddings.npy` and
per-episode arrays. State/action arrays use FP32, masks use bool, and visual
and text caches use FP16. Instructions may contain at most 64 valid tokens.
Plan storage for the visual caches and per-episode language tables.

To recover an interrupted preparation, rerun the same command with
`--resume`. The preparation manifest binds the cache to source content,
episode order, images, instructions, encoder assets, code, software settings
and `--limit`. Matching completed episodes are reused; unfinished episodes
are encoded again. A completed dataset is checked without loading encoders
or rewriting files. Input hashing and CPU image decoding still take time.

Use a new output directory after changing inputs, assets or settings. Keep
it separate from the source and encoder directories, and run one preparation
process per output. Weight-file symlinks are supported; nested directory
symlinks are rejected. Training can start after preparation has published
the final `dataset.json`.

### Action and sampling conventions

Each causal visual anchor spans four native control steps. Training predicts
16 steps; evaluation executes the first four. Episode tails and unavailable
future images are masked, and samples stay within their episode.

LIBERO/Plus use native OSC7, MetaWorld uses clipped Native4, and the dual-arm
benchmarks use absolute EEF commands converted through their codec to
relative actions. MetaWorld samples an episode uniformly, then an anchor
within it. Other benchmarks shuffle and partition valid anchors across
ranks. RoboTwin selects an official seen instruction deterministically for
each episode and epoch.

## Training

```bash
python -m zvla.posttrain train --bench libero-plus --data data/libero-plus \
  --checkpoint checkpoints/zvla-base.safetensors --output runs/libero-plus \
  --gpus 8 --batch-size 10
```

`--gpus` specifies GPUs per node; `--batch-size` specifies examples per GPU.
The global batch is **nodes × GPUs per node × batch size**. The launcher uses
`torchrun` with Megatron-FSDP and no gradient accumulation. Changing GPU
count changes the global batch; it does not automatically scale the learning
rate. Choose a batch that fits the available memory.

The generated recipes use these defaults:

| Setting | Default |
| --- | --- |
| LIBERO / LIBERO-Plus / RoboDojo | 30,000 steps |
| MetaWorld / RoboTwin | 50,000 steps |
| Base learning rate | `1e-5` |
| Wan / Reader backbone LR multiplier | 1 |
| Reader heads / geometry / G4D LR multiplier | 10 |
| AdamW | betas `(0.9, 0.95)`, weight decay `0.01` |
| Schedule | cosine, 5% warmup, minimum LR 10% of the initial value |
| Gradient clipping / precision | 1.0 / BF16 |
| Action gradient into Wan | zero for the first 5% of steps; linear ramp to 1 over the next 5% |
| Checkpoints | every 1,000 steps, retain two recovery points |

Set `--steps`, `--learning-rate`, `--batch-size` and `--workers` to override
the launcher defaults. Add `--dry-run` to print the generated configuration
and global batch without starting training.

### Multiple nodes and recovery

Run the same command on every node with a distinct `--node-rank`:

```bash
python -m zvla.posttrain train --bench robotwin --data /data/robotwin \
  --checkpoint /models/zvla-base.safetensors --output /runs/robotwin \
  --nodes 4 --node-rank 0 --gpus 8 --batch-size 10 \
  --master-addr TRAINING_HOST --master-port 29500
```

Use node ranks `0` through `3`; `TRAINING_HOST` is the reachable hostname of
node 0. Data and checkpoint paths must resolve identically on each node,
and the output directory must be shared. This example has global batch 320.

To resume, repeat the training command with `--resume`, keeping the
topology, batch, data and hyperparameters unchanged. The newest complete
checkpoint restores the optimizer, scheduler, random state and sampler
cursor. Resume from a run directory you created or trust.

For direct recipe editing, use [configs/posttrain](../simulation/configs/posttrain/):

```bash
torchrun --standalone --nproc-per-node 8 --module zvla.posttrain.train \
  --config configs/posttrain/libero.py --work-dir runs/libero \
  --cfg-options model.pretrained_checkpoint=checkpoints/zvla-base.safetensors
```

## Export and serve

The final training step exports a `model-step-*.safetensors` file in the
run's `checkpoints/` directory. Replace the example filename below with the
one produced by your run:

```bash
python -m zvla.posttrain export --run runs/libero-plus \
  --checkpoint runs/libero-plus/checkpoints/model-step-030000-epoch-XX.safetensors \
  --output artifacts/libero-plus
python -m zvla.posttrain serve --artifact artifacts/libero-plus \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

Export writes `model.safetensors` and `manifest.json`, including the
architecture and target benchmark's codec. Serving loads these files and
the same visual assets used for preparation.

Each service process maintains one episode's causal state. Clients call
`/reset`, send one frame on the first inference request, then send four new
frames per request. The supplied simulator clients maintain this sequence.
UMT5 encodes instructions on the GPU as needed, moves back to the CPU, and
caches the resulting embeddings.

## Evaluation

Keep the matching benchmark's model service running. In another terminal,
activate that simulator's environment and enter this repository's
`simulation/` directory:

```bash
conda activate YOUR_SIM_ENV
cd /path/to/Zerith-VA/simulation
```

### LIBERO and LIBERO-Plus

```bash
python -m zvla.posttrain eval --bench libero --benchmark-root /path/to/LIBERO \
  --suite libero_spatial --episodes 10 --server http://127.0.0.1:8000 --output results/libero-spatial
```

For LIBERO-Plus, use its own checkout, environment and configuration:

```bash
python -m zvla.posttrain eval --bench libero-plus --benchmark-root /path/to/LIBERO-plus \
  --suite libero_spatial --episodes 1 --server http://127.0.0.1:8000 --output results/plus-spatial
```

Suites are `libero_spatial`, `libero_object`, `libero_goal` and `libero_10`.
Each command evaluates all tasks in one suite; run all four for the full
benchmark. Defaults are 10 episodes per task for LIBERO and one for Plus.

### MetaWorld

Use `metaworld==3.1.1` and `mujoco==3.3.0` in the simulator environment:

```bash
python -m zvla.posttrain eval --bench metaworld --episodes 50 \
  --server http://127.0.0.1:8000 --output results/metaworld
```

The evaluator runs all 50 MT50 tasks, with 50 goals per task by default,
a horizon of 500 steps and no warmup. LIBERO/Plus and MetaWorld write
per-episode results to `episodes.jsonl` in the output directory.

### RoboTwin and RoboDojo

These adapters call the official harnesses for scene generation and
scoring. Use the interface versions targeted by the adapters:

| Component | Revision |
| --- | --- |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/bf44be51cf5717a5595ce59447f2cf5263d2aa95) | `bf44be51cf5717a5595ce59447f2cf5263d2aa95` |
| RoboDojo | `25691aa78fb34bbbe798aa3d880d57ef2788f696` |
| XPolicyLab | `72b3a52f2b069220088ef824ade5890885858379` |

Install the corresponding official assets, then run one task per command:

```bash
python -m zvla.posttrain eval --bench robotwin --benchmark-root /path/to/RoboTwin \
  --task adjust_bottle --server http://127.0.0.1:8000 --output results/robotwin
```

```bash
POLICY_ENV=zvla-client EVAL_ENV=robodojo POLICY_GPU=0 ENV_GPU=1 \
python -m zvla.posttrain eval --bench robodojo --benchmark-root /path/to/RoboDojo \
  --task stack_bowls --server http://127.0.0.1:8000 --output results/robodojo
```

Replace the environment names and GPU indices with your setup. `POLICY_ENV`
hosts the HTTP bridge with the official XPolicyLab dependencies; `EVAL_ENV`
hosts the simulator. RoboTwin uses `demo_clean` and unseen instructions.

The launcher copies its policy adapter into the specified benchmark
checkout and adds this repository's simulation source to the subprocess
`PYTHONPATH`. Official score files stay in the benchmark directory; the
requested output directory contains `evaluation.log`.

## Optional FlashAttention-3

The custom extension supports Hopper SM90, FP16/BF16, 128-wide attention
heads and boolean masks. Other cases use PyTorch SDPA. The extension source
and pinned CUTLASS headers are included in
[fa2_custom_mask](../simulation/fa2_custom_mask/), with their
[third-party notices](../simulation/fa2_custom_mask/NOTICE).

To prebuild on a Hopper GPU, install a compatible CUDA toolkit with `nvcc`,
a C++ compiler and Ninja, then run in the core environment:

```bash
python -m pip install ninja
python -c 'from fa2_custom_mask import prebuild_flash_attention3; print(prebuild_flash_attention3())'
```

Eligible calls may build on first use and fall back to SDPA if the build
fails. `ZVLA_REQUIRE_FLASH_ATTENTION3=1` instead rejects unsupported boolean
layouts and missing or broken builds. Build caches are keyed by source and
Python/PyTorch/CUDA ABI fingerprints. `ZVLA_FA3_CUTLASS_ROOT` may select an
identical pinned CUTLASS snapshot that passes the fingerprint check.
Compare numerical results, speed and memory use on your target hardware
when changing attention backends.

## Development checks

```bash
python -m pytest -q tests
```

Unit tests cover codecs, masking, sampling, model loading and artifact
roundtrips. Full training and closed-loop evaluation require the actual
model assets and simulator environments; unit tests do not establish
benchmark performance.
