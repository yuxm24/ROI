<p align="center"><img src="media/zerith-logo.png" alt="Zerith Robotics 标志" width="168"></p>
<h1 align="center">Zerith VA</h1>
<p align="center">面向机器人控制的视觉语言动作模型</p>
<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>

Zerith VA 是面向机器人控制的视觉语言动作模型。本仓库提供数据准备、后训练、
策略导出、基准评测和 H1 真机部署工具。

## 项目概览

- **基准后训练：** 支持 LIBERO、LIBERO-Plus、MetaWorld、RoboTwin 和 RoboDojo。
- **策略评测：** 通过 HTTP 服务运行导出策略，接入对应基准的模拟器评测。
- **H1 部署：** 分别运行 GPU 推理服务和机器人控制端，集成相机、SDK 与 FK/IK。

| 组件 | 用途 |
| --- | --- |
| [simulation/](simulation/) | 数据准备、训练、导出与评测 |
| [deployment/shared/](deployment/shared/) | HTTP 客户端与公共接口 |
| [deployment/inference/](deployment/inference/) | 模型加载、视觉编码与推理服务 |
| [deployment/robot/](deployment/robot/) | H1 SDK 适配与机器人控制 |

本仓库不包含模型权重，Zerith VA 权重计划单独发布。数据集和编码器资产
从对应上游项目获取。仿真导出采用各基准的动作接口；H1 部署需要兼容的 H1 权重。

## 安装

仿真流程使用 Linux、CUDA GPU 和 Conda 环境。从仓库根目录执行：

```bash
conda create -n zvla-train python=3.12 pip
conda activate zvla-train
cd simulation
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install megatron-core==0.19.0
python -m pip install -r requirements.txt
python -m zvla.posttrain --help
```

仿真 CLI 从 `simulation/` 目录运行，无需安装本项目包。
选用的模拟器在独立环境中安装。编码器资产、公开数据集和各基准的配置见
[仿真指南](docs/simulation_zh-CN.md)。

## 快速开始

下面以 LIBERO-Plus 为例，假定已将演示数据、Wan/VGGT 资产和兼容的基础权重
放在示例路径。所需资产的版本及下载命令见[仿真指南](docs/simulation_zh-CN.md#所需资产)。

在 `simulation/` 目录准备数据并启动训练：

```bash
python -m zvla.posttrain prepare --bench libero-plus \
  --source data/raw/libero-plus --output data/libero-plus \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0

python -m zvla.posttrain train --bench libero-plus --data data/libero-plus \
  --checkpoint checkpoints/zvla-base.safetensors --output runs/libero-plus \
  --gpus 8 --batch-size 10
```

`--gpus` 为每节点 GPU 数，`--batch-size` 为每卡 batch，需按硬件条件调整。
[训练与恢复](docs/simulation_zh-CN.md#训练)、[策略导出与服务](docs/simulation_zh-CN.md#导出与服务)
及[基准评测](docs/simulation_zh-CN.md#评测)见详细指南。

## H1 部署

按主机角色安装部署包，以下命令从仓库根目录执行：

```bash
# GPU 推理主机：使用兼容的 CUDA/PyTorch 环境
python -m pip install ./deployment/shared ./deployment/inference

# H1 主机：使用与 SDK 匹配的 Python 环境
python -m pip install ./deployment/shared ./deployment/robot
```

GPU 主机运行模型服务，H1 主机运行图像采集、状态估计与关节控制，
无需安装 PyTorch 或模型权重。真机运行还需要 H1 SDK、匹配的 IK 提供方、
机器人资产及现场调试验收。配置、连接检查和控制端启动见[H1 部署指南](docs/deployment_zh-CN.md)。

## 文档

| 指南 | English | 简体中文 |
| --- | --- | --- |
| 数据、训练与评测 | [Simulation guide](docs/simulation.md) | [仿真指南](docs/simulation_zh-CN.md) |
| H1 推理与机器人控制 | [Deployment guide](docs/deployment.md) | [部署指南](docs/deployment_zh-CN.md) |

开发与本地检查见[贡献指南](CONTRIBUTING.md)，安全问题请按[SECURITY.md](SECURITY.md)报告。

## 许可

项目代码采用 [Apache-2.0](LICENSE)，允许按协议条款商用。
第三方组件保留各自的许可，具体见[许可范围](LICENSE_POLICY.md)和[NOTICE](NOTICE)。

模型权重适用独立许可。未来发布的 Zerith 自有权重用于商业目的时，
需取得单独书面许可，详见[权重政策](MODEL_LICENSE_POLICY.md)。
