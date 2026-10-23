#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ENV="${ASSET_ENV:-${ROOT}/assets.env}"
[[ ! -f "${ASSET_ENV}" ]] || source "${ASSET_ENV}"

: "${VGGT_BASE_CHECKPOINT:?Set VGGT_BASE_CHECKPOINT or provide assets.env}"
: "${REVA_HPE_CHECKPOINT:?Set REVA_HPE_CHECKPOINT or provide assets.env}"
: "${BIWI_ROOT:?Set BIWI_ROOT or provide assets.env}"
: "${BIWI_PNP_ALIGNMENT_ROOT:?Set BIWI_PNP_ALIGNMENT_ROOT or provide assets.env}"

SUBJECT="${1:?Usage: scripts/eval_biwi_pnp_relative_subject.sh SUBJECT_ID}"
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/biwi_pnp_relative}"
SUBJECT_PADDED="$(printf '%02d' "$((10#${SUBJECT}))")"
OUTPUT_DIR="${OUTPUT_ROOT}/subject_${SUBJECT_PADDED}"
RUNTIME_DIR="${OUTPUT_DIR}/runtime"
mkdir -p "${OUTPUT_DIR}" "${RUNTIME_DIR}"

export PYTHONPATH="${ROOT}:${ROOT}/evaluation:${ROOT}/evaluation/eval_scripts:${PYTHONPATH:-}"
export TMPDIR="${RUNTIME_DIR}"
export MPLCONFIGDIR="${RUNTIME_DIR}/matplotlib"
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache"
export TORCH_HOME="${MTCNN_CACHE:-${RUNTIME_DIR}/torch}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}"

exec "${PYTHON}" "${ROOT}/evaluation/eval_biwi_crop_mtcnn_square_compare.py" \
  --biwi_dir "${BIWI_ROOT}" \
  --test_subjects "${SUBJECT}" \
  --base_checkpoint "${VGGT_BASE_CHECKPOINT}" \
  --lora_checkpoint "${REVA_HPE_CHECKPOINT}" \
  --vggt_pose_mode relative_h2c \
  --vggt_pose_encoding_type absT_quaR_FoV \
  --vggt_crop_schema mtcnn_square_pair_union \
  --vggt_translation_scale 1000 \
  --vggt_translation_target_type depth_norm_delta_t \
  --vggt_translation_rotation_mode predicted \
  --anchor_pose_source flame_pnp \
  --anchor_pnp_alignment_root "${BIWI_PNP_ALIGNMENT_ROOT}" \
  --anchor_pnp_reference mean_neutral \
  --anchor_pnp_reprojection_threshold_px 5 \
  --pose_prediction_diagnostics \
  --no_sixdrepnet \
  --no_sixdof_face \
  --batch_size "${BATCH_SIZE:-4}" \
  --gpu "${GPU:-0}" \
  --crop_size 256 \
  --vis_dir "${OUTPUT_DIR}" \
  "${@:2}"
