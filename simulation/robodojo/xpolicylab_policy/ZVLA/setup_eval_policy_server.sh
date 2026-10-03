#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Adapted from XPolicyLab's policy/DP/setup_eval_policy_server.sh.
# Upstream revision: 72b3a52f2b069220088ef824ade5890885858379. See ../../../NOTICE.
set -Eeuo pipefail

[[ $# -ge 9 ]] || {
  echo "usage: $0 BENCH TASK CKPT ENV_CFG ACTION_TYPE SEED GPU CONDA_ENV PORT [HOST]" >&2
  exit 2
}

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

[[ "${action_type}" == "ee" ]] || {
  echo "ZVLA RoboDojo adapter requires action_type=ee" >&2
  exit 2
}
[[ -n "${ZVLA_SERVER_URL:-}" ]] || {
  echo "ZVLA_SERVER_URL must point to one dedicated Z-VLA HTTP service" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REAL_SCRIPT_DIR="$(cd -P "${SCRIPT_DIR}" && pwd)"
ZVLA_REPO_ROOT="${ZVLA_REPO_ROOT:-$(cd "${REAL_SCRIPT_DIR}/../../.." && pwd)}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${XPL_ROOT}:${XPL_ROOT}/..:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  NO_PROXY='*' \
  no_proxy='*' \
  ZVLA_REPO_ROOT="${ZVLA_REPO_ROOT}" \
  python "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" \
    --overrides \
      "port=${policy_server_port}" \
      "host=${policy_server_host}" \
      "bench_name=${bench_name}" \
      "task_name=${task_name}" \
      "ckpt_name=${ckpt_name}" \
      "env_cfg_type=${env_cfg_type}" \
      "seed=${seed}" \
      "policy_name=ZVLA" \
      "action_type=${action_type}"
