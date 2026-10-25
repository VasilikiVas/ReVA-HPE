#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ENV="${ASSET_ENV:-${ROOT}/assets.env}"
[[ ! -f "${ASSET_ENV}" ]] || source "${ASSET_ENV}"

: "${VGGT_BASE_CHECKPOINT:?Set VGGT_BASE_CHECKPOINT or provide assets.env}"
: "${SYNTHETIC_FIXED_HAIR_ROOT:?Set SYNTHETIC_FIXED_HAIR_ROOT or provide assets.env}"
: "${SYNTHETIC_FIXED_NOHAIR_ROOT:?Set SYNTHETIC_FIXED_NOHAIR_ROOT or provide assets.env}"
: "${REVA_TRAIN_PAIR_CSV:?Set REVA_TRAIN_PAIR_CSV or provide assets.env}"
: "${REVA_VAL_PAIR_CSV:?Set REVA_VAL_PAIR_CSV or provide assets.env}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/training}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="${CONFIG:-flame_fromvggt_largepairs_fixed_fov_unified_crop_depth_norm_delta_t_rot1_vggtinit}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export TMPDIR="${OUTPUT_DIR}/runtime"
mkdir -p "${WANDB_DIR}" "${TMPDIR}"

# Checkpoints are written under OUTPUT_DIR, outside the repository. Set
# RESUME_CHECKPOINT to resume a run instead of initialising from VGGT-1B.
INIT_CHECKPOINT="${RESUME_CHECKPOINT:-${VGGT_BASE_CHECKPOINT}}"
cd "${ROOT}/training"
exec torchrun --master_port "${MASTER_PORT}" --nproc_per_node "${NPROC_PER_NODE}" launch.py \
  --config "${CONFIG}" \
  "checkpoint.resume_checkpoint_path=${INIT_CHECKPOINT}" \
  "checkpoint.save_dir=${OUTPUT_DIR}/ckpts" \
  "logging.log_dir=${OUTPUT_DIR}" \
  "data.train.dataset.dataset_configs.0.root_dirs.0.path=${SYNTHETIC_FIXED_HAIR_ROOT}" \
  "data.train.dataset.dataset_configs.0.root_dirs.1.path=${SYNTHETIC_FIXED_NOHAIR_ROOT}" \
  "data.train.dataset.dataset_configs.0.pair_csv=${REVA_TRAIN_PAIR_CSV}" \
  "data.val.dataset.dataset_configs.0.root_dirs.0.path=${SYNTHETIC_FIXED_HAIR_ROOT}" \
  "data.val.dataset.dataset_configs.0.root_dirs.1.path=${SYNTHETIC_FIXED_NOHAIR_ROOT}" \
  "data.val.dataset.dataset_configs.0.pair_csv=${REVA_VAL_PAIR_CSV}" \
  "$@"
