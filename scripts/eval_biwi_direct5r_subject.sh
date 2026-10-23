#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ENV="${ASSET_ENV:-${ROOT}/assets.env}"
[[ ! -f "${ASSET_ENV}" ]] || source "${ASSET_ENV}"

: "${VGGT_BASE_CHECKPOINT:?Set VGGT_BASE_CHECKPOINT or provide assets.env}"
: "${REVA_HPE_CHECKPOINT:?Set REVA_HPE_CHECKPOINT or provide assets.env}"
: "${BIWI_ROOT:?Set BIWI_ROOT or provide assets.env}"
: "${BIWI_AVATAR_BANK_ROOT:?Set BIWI_AVATAR_BANK_ROOT or provide assets.env}"

SUBJECT="${1:?Usage: scripts/eval_biwi_direct5r_subject.sh SUBJECT_ID}"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/biwi_direct5r}"
RUNTIME_DIR="${OUTPUT_DIR}/runtime/subject_${SUBJECT}"
BANK_NAMES="${BANK_NAMES:-metric_anchor_bank_front20_r085,metric_anchor_bank_front20_r095,metric_anchor_bank_front20_r100,metric_anchor_bank_front20_r105,metric_anchor_bank_front20_r115}"
mkdir -p "${OUTPUT_DIR}" "${RUNTIME_DIR}"

export PYTHONPATH="${ROOT}:${ROOT}/evaluation:${ROOT}/evaluation/eval_scripts:${ROOT}/reva/experiments:${PYTHONPATH:-}"
export TMPDIR="${RUNTIME_DIR}"
export MPLCONFIGDIR="${RUNTIME_DIR}/matplotlib"
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache"
export TORCH_HOME="${MTCNN_CACHE:-${RUNTIME_DIR}/torch}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}"

exec "${PYTHON}" "${ROOT}/evaluation/eval_biwi_facelift_metric_anchor_bank.py" \
  --biwi-dir "${BIWI_ROOT}" \
  --bank-root "${BIWI_AVATAR_BANK_ROOT}" \
  --bank-name "${BANK_NAMES}" \
  --subject "${SUBJECT}" \
  --out-dir "${OUTPUT_DIR}" \
  --base-checkpoint "${VGGT_BASE_CHECKPOINT}" \
  --lora-checkpoint "${REVA_HPE_CHECKPOINT}" \
  --pose-encoding-type absT_quaR_FoV \
  --translation-target-type depth_norm_delta_t \
  --pair-batch-size "${PAIR_BATCH_SIZE:-4}" \
  --crop-size 256 \
  --mtcnn-ad 0.4 \
  --ransac-rot-threshold-deg 10 \
  --ransac-trans-threshold-mm 50 \
  --ransac-min-inliers 3 \
  --frame-selection post_enrollment \
  --max-visualizations "${MAX_VISUALIZATIONS:-0}" \
  --gpu "${GPU:-0}" \
  "${@:2}"
