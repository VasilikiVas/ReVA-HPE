#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIRECT_BANK_ROOT="${DIRECT_BANK_ROOT:-${ROOT}/outputs/biwi_direct5r}"
PYTHON="${PYTHON:-python}"

export PYTHONPATH="${ROOT}:${ROOT}/evaluation:${ROOT}/evaluation/eval_scripts:${PYTHONPATH:-}"
exec "${PYTHON}" "${ROOT}/reva/avatar_pipeline/aggregate_metric_anchor_evaluation.py" \
  --input-dir "${DIRECT_BANK_ROOT}" \
  "$@"
