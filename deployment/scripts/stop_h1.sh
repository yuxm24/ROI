#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${ZWAM_PYTHON:-python}" -m zwam.process stop h1 "$@"
