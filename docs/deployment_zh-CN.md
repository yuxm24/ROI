# H1 部署

[项目概览](../README_zh-CN.md) · [English](deployment.md)

Zerith VA 在 GPU 主机运行推理，在 H1 主机处理相机、实测状态、FK/IK 和 SDK
控制。两个角色分别安装：H1 主机无需 PyTorch 或模型权重，GPU 主机无需机器人 SDK。

## 1. 分主机安装

GPU 主机使用 Linux、Python 3.10 或更新版本，以及支持 BF16 的 CUDA GPU。
在仓库根目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ./deployment/shared ./deployment/inference
cd deployment
cp configs/server.example.json configs/server.local.json
cp configs/visual_assets.example.json configs/visual_assets.local.json
```

H1 主机选择与运动和相机 SDK 二进制 ABI 匹配的 Python 环境，在仓库根目录执行：

```bash
python -m pip install ./deployment/shared ./deployment/robot
cd deployment
cp configs/h1.example.json configs/h1.local.json
zwam robot
```

`zwam robot` 默认只运行离线坐标检查，不打开 SDK 或相机，也不发送运动指令。

## 2. 准备资产与配置

从模型提供方获取兼容的 H1 checkpoint 及 manifest、Wan2.2 VAE、匹配的
Wan/VGGT 源码与 VGGT 权重，以及固定的 UMT5 文本嵌入库。仿真 benchmark
导出使用不同的 codec，不能直接作为 H1 策略加载。在 GPU 环境安装外部源码所需
依赖，并保持源码版本和预处理与 checkpoint 一致。

GPU 端示例布局如下，路径相对 `deployment/`：

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

`zwam-h1-inference-v1` bundle 可直接放入 `assets/policy`。如果拿到的是
`zerith-va-policy-action20-linear-g4d-teacher` 导出，先生成部署 bundle：

```bash
zwam prepare --source /path/to/exported-policy --output assets/policy
```

输出目录必须尚不存在。只有原始训练权重不足以部署，还需要对应元数据。架构标识为
`zerith-va-action20-linear-g4d-teacher`；加载器检查模型键名、形状、归一化统计和
文件哈希。文本库包含准确的任务字符串，以及形状为 `[任务数,64,4096]` 的有限
float16 嵌入。H1 配置的 `prompt` 必须与其中一个字符串完全一致；新增指令时也需
提供对应嵌入。

编辑[服务端配置](../deployment/configs/server.example.json)、
[视觉资产配置](../deployment/configs/visual_assets.example.json)和
[H1 配置](../deployment/configs/h1.example.json)的本地副本。以下命令都从
`deployment/` 执行；配置中的相对路径按当前工作目录解析，不按 JSON 所在目录
解析，也支持绝对路径。本地配置、模型资产和外部库均被 Git 忽略。

| 配置文件 | 需要填写的字段 |
| --- | --- |
| `server.local.json` | `artifact`、`visual_assets`、`text_assets`、`device`、`bind`、`port` |
| `visual_assets.local.json` | `wan_source`、`wan_weights`、`vggt_source`、`vggt_weights`，以及源码指纹和权重 SHA-256 |
| `h1.local.json` | `server_url`、准确的 `prompt`、SDK 与规划器路径、相机及服务设置；配置期间保持 `commissioned: false` |

资产哈希使用提供方给出的值。源码指纹覆盖 Python 包目录，可对所选源码执行：

```bash
zwam source-hash external/Wan2.2/wan
zwam source-hash external/vggt/vggt
```

### H1 SDK 与规划器

安装与机器人固件及 Python ABI 匹配的 [Zerith 公开 SDK](https://github.com/inFpZero/zerith_public_sdk)。
示例配置选择 `H1_SDK_1.3.9`：`sdk_root` 指向
`external/zerith_public_sdk/H1_SDK_1.3.9/h1_sdk_v1.3.9_python3.10`，
`camera_sdk_root` 指向独立的 `camera_sdk_python` 目录。适配器同时搜索
`sdk_root` 与 `sdk_root/lib`。安装相机依赖后，先用 SDK 的示例检查相机和电机访问。

SDK 提供相机与电机接口。兼容的原生 IK 提供方，以及 H1 URDF 和碰撞资产需另外
安装。将 `planner_source` 设置为提供方的 Python 源码目录，`planner_factory`
设置为 `module:function` 入口；示例使用 `vla_source:build_planner`。
原生适配器要求 `processors.fast_ik_backend` 和 `fast_ik_cpp.FastIKSolver`。

规划器须提供有限的 control19 关节上下限（`lower`、`upper`）、腰部固定的
`arm` 模型、`arm_control_indices`、`root_from_body`、`eef_names`，以及
`body_eef_poses(q19)`。双手位姿须位于 `body_yaw_link`，末端 link 按
`left_end_effector_link`、`right_end_effector_link` 排列。手臂 IK 不得再次移动腰部。

## 3. 检查与启动

GPU 主机执行：

```bash
zwam doctor --role server --config configs/server.local.json --verify-assets
bash scripts/start_server.sh --config configs/server.local.json
```

脚本以前台方式运行服务。也可直接运行
`zwam serve --config configs/server.local.json`，并用 Ctrl+C 停止。
H1 主机检查本地配置和服务端元信息：

```bash
zwam doctor --role h1 --config configs/h1.local.json
zwam robot --check-server --config configs/h1.local.json
```

`doctor` 检查配置和文件位置，不导入 SDK、不打开相机，也不移动机器人。服务绑定
非回环地址时，两端环境须设置相同的 `ZWAM_API_KEY`，长度至少 32 字符。通过
SSH 隧道或 TLS 保护连接，不将密钥写入 JSON 或命令参数。`server_url` 应指向
可达的服务或本地隧道端点；示例为 `http://127.0.0.1:55558`。

启用控制前，使用与 checkpoint 匹配的记录进行回放，对比解码目标和 FK/IK，并
测量所选权重的延迟与内存需求。在目标 H1 上核对 SDK ABI、关节符号/单位/零点、
URDF 和末端偏移、相机顺序/方向/时间戳、关节限位、所需服务与实体急停。
只允许一个控制器发布指令。

完成现场验收后，在 `h1.local.json` 中设置 `commissioned: true`，从交互终端
显式启动：

```bash
bash scripts/start_h1.sh --config configs/h1.local.json --execute
```

输入 `EXECUTE` 允许 SDK 初始化，检查姿态和相机视图，再按 Enter 开始推理。
初始化可能使机器人移动。可选的 `--initial-joints /path/to/pose.npy` 用于指定
经过核对的 SDK21 起始姿态；`--initial-duration` 至少为两秒。

先停止控制器，再停止推理服务。若使用启动脚本，在各自主机的另一个终端执行：

```bash
bash scripts/stop_h1.sh
bash scripts/stop_server.sh
```

Ctrl+C 也会请求正常退出。正常 SDK 反初始化可能使硬件移动；出现运动危险时使用
实体急停。启停脚本在 `.run/` 中记录进程身份；主机不支持安全的 pidfd 信号发送
时，在原终端使用 Ctrl+C。

## 参考：坐标与 HTTP

实测 `state23` 和绝对目标 `commands` 使用相同的 23 维布局：

| 索引 | 含义 | 表示方式 |
| --- | --- | --- |
| `0:3`、`10:13` | 左/右手位置 | `body_yaw_link` 坐标系下的米 |
| `3:9`、`13:19` | 左/右手旋转 | 旋转矩阵前两列，按列展开 |
| `9`、`19` | 左/右夹爪 | 状态范围 0–1；命令范围 0–1.5 |
| `20:23` | 腰部升降、俯仰、偏航 | 米、弧度、弧度 |

checkpoint 的 `normalization_stats.command_contract` 须声明
`pose_frame: "body_yaw_link"`、非空 `frame_evidence`、`position_unit: "m"`、
`waist_units: ["m","rad","rad"]`，以及
`waist_joint_order: ["waist_joint1.position","waist_joint2.position","waist_joint3.position"]`。
可选的 `waist_semantics` 必须为 `["lift","pitch","yaw"]`。加载器核对这些声明
与 checkpoint 统计；修改标签不会转换权重或坐标。SDK21 顺序为左臂 7 + 夹爪、
右臂 7 + 夹爪、腰 3、头 2；规划器 control19 顺序为腰 3、左臂 7、右臂 7、头 2。

协议标识为 `zwam-h1-command23-http-npz-v3`，共享客户端与推理包须使用匹配
版本。POST 请求需要 `X-ZWAM-Protocol: zwam-h1-command23-http-npz-v3`；
配置 API 授权后，另用 bearer token 传递密钥。

| 端点 | 用途 |
| --- | --- |
| `GET /health` | 协议元信息和可用任务字符串 |
| `POST /reset` | JSON `prompt` 与整数 `seed`；启动或重置当前拥有的 episode |
| `POST /infer` | 输入 NPZ 观测，输出 NPZ 动作块 |
| `POST /release` | 释放 episode 所有权 |

推理输入字段为 `images`（RGB uint8 `[T,3,H,W,3]`）、`state23`
（float32 `[23]`）和 `valid23`（bool `[23]`）。相机顺序为头部、左腕、右腕，
对应 SDK 的 `cam_high`、`cam_left_wrist`、`cam_right_wrist`。
reset 后首次请求用一帧，后续每次用四帧新图像。输出 `commands` 为 float32
`[4,23]`，`valid` 为 bool `[4,23]`，即 16 行预测中的前四行。无效值须置零；
对象数组、非有限值和超限 NPZ 会被拒绝。

### 会话与恢复

Python `PolicyClient` 管理 `X-Episode-Token` 和 `X-Sequence`。每个服务只有
一个 episode 所有者。推理不会自动重试；超时后若无法确定请求是否完成，先显式
reset，再发送新观测。reset 响应丢失时，同一个客户端保留 token，可调用
`reset(prompt, seed)` 恢复，或调用 `close()` 释放所有权。所有权没有过期时间或
其他客户端自动接管机制。

控制器在发送 SDK 指令前检查目标变化、IK/FK 残差、关节步幅、图像新鲜度和同步。
IK 失败时暂停等待操作员检查；按 Enter 丢弃旧动作块并重新规划。部分 SDK 发送
失败会锁定故障，无法通过 Enter 清除。动作按名义 30 Hz 调度，推理和 IK 增加
延迟。原生 IK 的耗时检查无法中断卡住的原生调用；软件检查和暂停不能代替实体急停。
