# 仿真后训练

[项目概览](../README_zh-CN.md) · [English](simulation.md)

Zerith VA 支持 `libero`、`libero-plus`、`metaworld`、`robotwin` 和
`robodojo` 后训练。流程包括生成视觉缓存与 benchmark 专属状态/动作 codec、
使用 Megatron-FSDP 训练，以及导出用于模拟器评测的 HTTP 策略。

基础 checkpoint、编码器权重、演示数据与模拟器资产需单独获取。仿真导出物
使用所选 benchmark 的 codec；H1 部署需要兼容的 H1 checkpoint，详见
[部署指南](deployment_zh-CN.md)。

## 环境

从仓库根目录进入 `simulation/`。除特别说明外，下文命令均在此目录执行。

```bash
cd simulation
conda create -n zvla-train python=3.12 pip
conda activate zvla-train
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install megatron-core==0.19.0
python -m pip install -r requirements.txt
```

该环境用于数据准备、训练与模型服务。GPU 驱动需兼容 CUDA 13.0 的 PyTorch
构建。本项目直接从源码目录运行，无需安装为 Python 包。

按官方说明为所选模拟器建立独立环境。LIBERO 和 LIBERO-Plus 使用各自的
源码；MetaWorld 使用其 MuJoCo 环境；RoboTwin 和 RoboDojo 使用官方评测
harness。模拟器客户端需要 NumPy 和 requests，只会加载所选 benchmark
的仿真依赖。

独立 HTTP 客户端与 launcher 可使用以下环境：

```bash
conda env create -n zvla-client -f environment-client.yml
```

它包含 Python 3.11、NumPy 和 requests。用于 RoboDojo 的 `POLICY_ENV` 时，
还需补充官方 XPolicyLab 依赖；模拟器在独立的 `EVAL_ENV` 中运行。

## 所需资产

Wan VAE/UMT5 与 VGGT 使用以下源码和权重版本。数据准备与服务端须保持
权重、tokenizer 和预处理一致。

| 资产 | 来源 | Revision |
| --- | --- | --- |
| Wan 源码 | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` |
| Wan 权重与 tokenizer | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` |
| VGGT 源码 | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | `a288dd0f14786c93483e45524328726ab7b1b4ce` |
| VGGT 权重 | [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | `860abec7937da0a4c03c41d3c269c366e82abdf9` |

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

将单独获取的基础 checkpoint 放在 `checkpoints/zvla-base.safetensors`，
或通过 `--checkpoint` 指定其位置。模型从 checkpoint 初始化网络模块，
目标 benchmark 的 codec 使用准备流程生成的 `codec.json`。

## 演示数据与准备

| Benchmark | 演示数据源与 revision | `--source` 指向 |
| --- | --- | --- |
| `libero` | [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)，`6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551` | 四套任务的 RLDS 根目录 |
| `libero-plus` | [Sylvest/libero_plus_lerobot](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot)，`22c57433fef692b5b9ecc0795344daac7fa867a5` | 含 `meta/info.json` 的 LeRobot 根目录；准备时只读取 train split |
| `metaworld` | [MINT-SJTU/Evo1_MetaWorld_Dataset](https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset)，`612c4e83d0e3b4e0be5227970bd8f5f07fc0c488` | 含 `meta/info.json` 的 LeRobot 根目录 |
| `robotwin` | [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)，`981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042` | 含 `dataset/<task>/demo_clean.zip` 的下载根目录 |
| `robodojo` | [RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)，`91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd` | 解压后的 `ee_lerobot_v30`，含 `meta/info.json` |

MetaWorld 演示集由第三方提供。RoboTwin 使用官方 `demo_clean` HDF5 中的
EEF command；RoboDojo 使用双臂 EEF LeRobot v3.0 数据。

以下载和准备 LIBERO-Plus 为例：

```bash
hf download Sylvest/libero_plus_lerobot --repo-type dataset \
  --revision 22c57433fef692b5b9ecc0795344daac7fa867a5 --local-dir data/raw/libero-plus
python -m zvla.posttrain prepare --bench libero-plus \
  --source data/raw/libero-plus --output data/libero-plus \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0
```

其他 benchmark 使用相同命令，替换 `--bench`、`--source` 与输出目录。
小规模检查可加 `--limit 2`，并使用独立输出，例如 `data/libero-plus-smoke`。

准备流程一次使用一张 GPU，逐 episode 编码。结果包含 `dataset.json`、
`codec.json`、`tasks.json`、`text_embeddings.npy` 和逐 episode 数组。
状态/动作使用 FP32，掩码使用 bool，视觉与文本缓存使用 FP16。指令最多包含
64 个有效 token；存储空间需要覆盖视觉缓存和逐 episode 语言表。

准备中断后，在相同命令上加 `--resume`。准备记录将缓存绑定到源文件内容、
episode 顺序、图像、指令、编码器资产、代码、软件设置及 `--limit`。
一致的已完成 episode 会复用，未完成的 episode 重新编码；完整数据集只校验，
不加载编码器或改写文件。恢复仍需读取哈希和在 CPU 解码图像。

输入、模型资产或设置变化后使用新输出目录。输出目录须与源数据、编码器
源码及权重目录分开，每个输出目录只运行一个准备进程。权重文件符号链接
可以使用，内部目录符号链接会被拒绝。准备完成并生成最终 `dataset.json`
后再启动训练。

### 动作与采样约定

每个因果视觉 anchor 对应四个原生控制步。训练预测 16 步，评测执行前四步。
episode 尾部和不可用的未来图像通过掩码排除，样本始终位于所属 episode 内。

LIBERO/Plus 使用原生 OSC7，MetaWorld 使用 clipped Native4；双臂 benchmark
使用绝对 EEF command，通过 codec 转为相对动作。MetaWorld 先均匀抽取
episode，再抽取其中的 anchor；其余 benchmark 对有效 anchor 打乱并分配到
各 rank。RoboTwin 每个 epoch 为每个 episode 确定性选择官方 seen instruction。

## 训练

```bash
python -m zvla.posttrain train --bench libero-plus --data data/libero-plus \
  --checkpoint checkpoints/zvla-base.safetensors --output runs/libero-plus \
  --gpus 8 --batch-size 10
```

`--gpus` 是每节点 GPU 数，`--batch-size` 是每卡样本数。
**全局 batch = 节点数 × 每节点卡数 × 每卡 batch**。Launcher 使用 `torchrun`
与 Megatron-FSDP，不使用梯度累积。改变卡数会改变全局 batch，学习率不会
自动缩放；应根据显存选择 batch。

生成的 recipe 默认参数如下：

| 设置 | 默认值 |
| --- | --- |
| LIBERO / LIBERO-Plus / RoboDojo | 30,000 steps |
| MetaWorld / RoboTwin | 50,000 steps |
| 基础学习率 | `1e-5` |
| Wan / Reader backbone LR 倍率 | 1 |
| Reader heads / geometry / G4D LR 倍率 | 10 |
| AdamW | betas `(0.9, 0.95)`，weight decay `0.01` |
| 调度 | cosine，5% warmup，最低 LR 为初始值的 10% |
| 梯度裁剪 / 精度 | 1.0 / BF16 |
| 动作损失到 Wan 的梯度 | 前 5% 步数为 0，随后 5% 线性增加到 1 |
| Checkpoint | 每 1,000 步，保留两个恢复点 |

可通过 `--steps`、`--learning-rate`、`--batch-size` 和 `--workers` 修改参数。
加 `--dry-run` 只打印生成的配置和全局 batch，不启动训练。

### 多机与恢复

每个节点执行相同命令，各自设置不同的 `--node-rank`：

```bash
python -m zvla.posttrain train --bench robotwin --data /data/robotwin \
  --checkpoint /models/zvla-base.safetensors --output /runs/robotwin \
  --nodes 4 --node-rank 0 --gpus 8 --batch-size 10 \
  --master-addr TRAINING_HOST --master-port 29500
```

节点序号使用 `0` 至 `3`，`TRAINING_HOST` 是节点 0 的可达主机名。
数据和 checkpoint 路径需在每个节点一致，输出目录须共享。此例全局 batch 为 320。

恢复时保留原拓扑、batch、数据和超参数，在原训练命令上加 `--resume`。
最新完整 checkpoint 会恢复优化器、调度器、随机状态与采样游标。
使用自己生成或可信的训练目录恢复。

也可直接编辑 [configs/posttrain](../simulation/configs/posttrain/) 后启动：

```bash
torchrun --standalone --nproc-per-node 8 --module zvla.posttrain.train \
  --config configs/posttrain/libero.py --work-dir runs/libero \
  --cfg-options model.pretrained_checkpoint=checkpoints/zvla-base.safetensors
```

## 导出与服务

训练最后一步在输出目录的 `checkpoints/` 中生成 `model-step-*.safetensors`。
将下例文件名替换为本次训练实际生成的文件：

```bash
python -m zvla.posttrain export --run runs/libero-plus \
  --checkpoint runs/libero-plus/checkpoints/model-step-030000-epoch-XX.safetensors \
  --output artifacts/libero-plus
python -m zvla.posttrain serve --artifact artifacts/libero-plus \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

导出结果为 `model.safetensors` 和 `manifest.json`，包含架构与目标 benchmark
的 codec。服务端加载导出物，以及与数据准备一致的视觉资产。

每个服务进程维护一个 episode 的因果状态。客户端先调用 `/reset`，第一次
推理传一帧，之后每次传四个新帧。项目提供的模拟器客户端会维护这一顺序。
UMT5 按需在 GPU 编码指令后移回 CPU，并缓存编码结果。

## 评测

保持对应 benchmark 的模型服务运行。在另一个终端激活相应模拟器环境，
进入本仓库的 `simulation/`：

```bash
conda activate YOUR_SIM_ENV
cd /path/to/Zerith-VA/simulation
```

### LIBERO 与 LIBERO-Plus

```bash
python -m zvla.posttrain eval --bench libero --benchmark-root /path/to/LIBERO \
  --suite libero_spatial --episodes 10 --server http://127.0.0.1:8000 --output results/libero-spatial
```

LIBERO-Plus 使用独立的源码、环境和配置：

```bash
python -m zvla.posttrain eval --bench libero-plus --benchmark-root /path/to/LIBERO-plus \
  --suite libero_spatial --episodes 1 --server http://127.0.0.1:8000 --output results/plus-spatial
```

Suite 可选 `libero_spatial`、`libero_object`、`libero_goal` 和 `libero_10`。
每条命令运行一套的全部任务，完整评测需运行四套。
LIBERO 默认每任务 10 个 episode，Plus 默认每任务一个。

### MetaWorld

模拟器环境使用 `metaworld==3.1.1`、`mujoco==3.3.0`：

```bash
python -m zvla.posttrain eval --bench metaworld --episodes 50 \
  --server http://127.0.0.1:8000 --output results/metaworld
```

评测覆盖全部 50 个 MT50 任务，默认每任务 50 个目标、horizon 500，不加 warmup。
LIBERO/Plus 与 MetaWorld 在输出目录中生成逐 episode 的 `episodes.jsonl`。

### RoboTwin 与 RoboDojo

Adapter 调用官方 harness 生成场景和评分。使用与 adapter 接口对应的版本：

| 组件 | Revision |
| --- | --- |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/bf44be51cf5717a5595ce59447f2cf5263d2aa95) | `bf44be51cf5717a5595ce59447f2cf5263d2aa95` |
| RoboDojo | `25691aa78fb34bbbe798aa3d880d57ef2788f696` |
| XPolicyLab | `72b3a52f2b069220088ef824ade5890885858379` |

安装对应官方资产后，每条命令评测一个任务：

```bash
python -m zvla.posttrain eval --bench robotwin --benchmark-root /path/to/RoboTwin \
  --task adjust_bottle --server http://127.0.0.1:8000 --output results/robotwin
```

```bash
POLICY_ENV=zvla-client EVAL_ENV=robodojo POLICY_GPU=0 ENV_GPU=1 \
python -m zvla.posttrain eval --bench robodojo --benchmark-root /path/to/RoboDojo \
  --task stack_bowls --server http://127.0.0.1:8000 --output results/robodojo
```

按实际环境修改环境名和 GPU 序号。`POLICY_ENV` 是已安装官方 XPolicyLab
依赖的 HTTP bridge 环境，`EVAL_ENV` 是模拟器环境。
RoboTwin 使用 `demo_clean` 和 unseen instructions。

Launcher 会将 policy adapter 复制到指定 benchmark 源码目录，并将本仓库
仿真源码加入子进程的 `PYTHONPATH`。官方评分文件保留在 benchmark 目录，
指定输出目录记录 `evaluation.log`。

## 可选 FlashAttention-3

自定义扩展支持 Hopper SM90、FP16/BF16、128 维 attention head 和布尔 mask，
其他情况使用 PyTorch SDPA。扩展源码与固定版本的 CUTLASS 头文件位于
[fa2_custom_mask](../simulation/fa2_custom_mask/)，并保留
[第三方署名和许可](../simulation/fa2_custom_mask/NOTICE)。

在 Hopper GPU 上预编译需要兼容的 CUDA toolkit（含 `nvcc`）、C++ 编译器
和 Ninja。进入核心环境后运行：

```bash
python -m pip install ninja
python -c 'from fa2_custom_mask import prebuild_flash_attention3; print(prebuild_flash_attention3())'
```

符合条件的调用可首次使用时编译，构建失败则回退 SDPA。
`ZVLA_REQUIRE_FLASH_ATTENTION3=1` 会对不支持的布尔布局、缺失或损坏的构建
报错。构建缓存使用源码和 Python/PyTorch/CUDA ABI 指纹；
`ZVLA_FA3_CUTLASS_ROOT` 可指向通过相同指纹检查的固定版本 CUTLASS 副本。
切换 attention 后端时，应在目标硬件比较数值结果、速度和显存占用。

## 开发检查

```bash
python -m pytest -q tests
```

单元测试覆盖 codec、掩码、采样、模型加载和导出物回读。
完整训练与闭环评测需要实际模型资产和模拟器环境，单元测试不代表 benchmark 效果。
