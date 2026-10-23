#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ENV="${ASSET_ENV:-${ROOT}/assets.env}"
[[ ! -f "${ASSET_ENV}" ]] || source "${ASSET_ENV}"

: "${VGGT_BASE_CHECKPOINT:?Set VGGT_BASE_CHECKPOINT or provide assets.env}"
: "${REVA_HPE_CHECKPOINT:?Set REVA_HPE_CHECKPOINT or provide assets.env}"
: "${FAMOS_READY_ROOT:?Set FAMOS_READY_ROOT or provide assets.env}"
: "${FAMOS_AVATAR_BANK_ROOT:?Set FAMOS_AVATAR_BANK_ROOT or provide assets.env}"

RAW_SUBJECT="${1:?Usage: scripts/eval_famos_direct5r_subject.sh FaMoS_subject_NNN}"
PROTOCOL="${PROTOCOL:-famos_view00_mtcnn_visible_no091}"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/famos_direct5r}"
RUNTIME_DIR="${OUTPUT_DIR}/runtime/${RAW_SUBJECT}"
BANK_NAMES="${BANK_NAMES:-metric_anchor_bank_front20_r085,metric_anchor_bank_front20_r095,metric_anchor_bank_front20_r100,metric_anchor_bank_front20_r105,metric_anchor_bank_front20_r115}"
mkdir -p "${OUTPUT_DIR}" "${RUNTIME_DIR}"

export PYTHONPATH="${ROOT}:${ROOT}/evaluation:${ROOT}/evaluation/eval_scripts:${ROOT}/reva/experiments:${PYTHONPATH:-}"
export TMPDIR="${RUNTIME_DIR}"
export MPLCONFIGDIR="${RUNTIME_DIR}/matplotlib"
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache"
export TORCH_HOME="${MTCNN_CACHE:-${RUNTIME_DIR}/torch}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}"

exec "${PYTHON}" "${ROOT}/reva/experiments/eval_famos_facelift_bank_hybrid.py" \
  --protocol "${PROTOCOL}" \
  --ready-root "${FAMOS_READY_ROOT}" \
  --bank-root "${FAMOS_AVATAR_BANK_ROOT}" \
  --bank-name "${BANK_NAMES}" \
  --bank-layout manual_metric_raw_subject \
  --raw-subject "${RAW_SUBJECT}" \
  --out-root "${OUTPUT_DIR}" \
  --base-checkpoint "${VGGT_BASE_CHECKPOINT}" \
  --lora-checkpoint "${REVA_HPE_CHECKPOINT}" \
  --translation-target-type depth_norm_delta_t \
  --translation-scale 1000 \
  --bank-translation-mode metric \
  --require-gt-free-bank \
  --pair-batch-size "${PAIR_BATCH_SIZE:-4}" \
  --crop-size 256 \
  --mtcnn-ad 0.4 \
  --ransac-rot-threshold-deg 10 \
  --ransac-trans-threshold-mm 50 \
  --ransac-min-inliers 3 \
  --method-name "ReVA-HPE (Direct-5r)" \
  --max-visualizations "${MAX_VISUALIZATIONS:-0}" \
  --gpu "${GPU:-0}" \
  "${@:2}"
