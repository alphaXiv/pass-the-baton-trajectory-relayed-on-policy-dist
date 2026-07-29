#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REPO_ROOT}/reproduction/config.env"

echo "=== Relay-OPD autonomous reproduction ==="
echo "REPRO_MODE=${REPRO_MODE}"
echo "OFFICIAL_RELAY_OPD_COMMIT=${OFFICIAL_RELAY_OPD_COMMIT}"
echo "RUN_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "JOB_COMPLETION_INDEX=${JOB_COMPLETION_INDEX:-0}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${REPO_ROOT}/relay-opd"

python3 -m pip install --disable-pip-version-check --no-cache-dir -e .
python3 -m pip install --disable-pip-version-check --no-cache-dir \
  -r requirements-relay-opd.txt

case "${REPRO_MODE}" in
  verify)
    RELAY_OPD_REQUIRE_CUDA=1 python3 environment/verify_install.py
    ;;
  *)
    echo "Unknown REPRO_MODE=${REPRO_MODE}" >&2
    exit 64
    ;;
esac

echo "REPRO_RESULT status=PASS mode=${REPRO_MODE} official_commit=${OFFICIAL_RELAY_OPD_COMMIT} worker=${JOB_COMPLETION_INDEX:-0}"
