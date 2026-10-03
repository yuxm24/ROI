#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Adapted from XPolicyLab's policy/demo_policy/setup_eval_env_client.sh.
# Upstream revision: 72b3a52f2b069220088ef824ade5890885858379. See ../../../NOTICE.
set -Eeuo pipefail

[[ $# -ge 10 ]] || {
  echo "usage: $0 BENCH TASK CKPT ENV_CFG ACTION_TYPE SEED GPU CONDA_ENV INFO PORT [HOST]" >&2
  exit 2
}

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

bash "${XPL_ROOT}/utils/setup_env_client.sh" \
  "${XPL_ROOT}/utils" \
  "${SCRIPT_DIR}/deploy.yml" \
  "${eval_env_conda_env}" \
  "${policy_server_port}" \
  "${bench_name}" \
  "${task_name}" \
  "${env_cfg_type}" \
  ZVLA \
  "${additional_info}" \
  "${BENCH_ROOT}" \
  "${seed}" \
  "${env_gpu_id}" \
  "${policy_server_ip}"
