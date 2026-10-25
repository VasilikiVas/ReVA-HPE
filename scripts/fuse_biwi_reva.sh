#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELATIVE_POSE_ROOT="${RELATIVE_POSE_ROOT:-${ROOT}/outputs/biwi_pnp_relative}"
DIRECT_BANK_ROOT="${DIRECT_BANK_ROOT:-${ROOT}/outputs/biwi_direct5r}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/biwi_reva_fused}"
PYTHON="${PYTHON:-python}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${ROOT}:${ROOT}/evaluation:${ROOT}/evaluation/eval_scripts:${PYTHONPATH:-}"
exec "${PYTHON}" "${ROOT}/reva/avatar_pipeline/build_true_avatar_relative_hybrid.py" \
  --relative-pose-root "${RELATIVE_POSE_ROOT}" \
  --avatar-root "${DIRECT_BANK_ROOT}" \
  --out-dir "${OUTPUT_DIR}" \
  --front-direction-z-sign -1 \
  --front-min-abs-z 0 \
  --trim-fraction 0.2 \
  "$@"
