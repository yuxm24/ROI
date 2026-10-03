#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -eq 10 ]] || {
  echo "usage: $0 BENCH TASK CKPT ENV_CFG ACTION_TYPE SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_ENV" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
port="$(bash "${XPL_ROOT}/utils/get_free_port.sh")"
server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
  "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "${port}" &
server_pid=$!
bash "${XPL_ROOT}/utils/wait_for_policy_server.sh" \
  localhost "${port}" "${server_pid}" "ZVLA bridge" 120
bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
  "$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}" \
  "ckpt_name=$3,action_type=$5" "${port}" localhost
