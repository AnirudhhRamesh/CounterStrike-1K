#!/usr/bin/env bash
# Run the locked DIAMOND-CSGO low-res baseline on a CS2-WM manifest, then
# evaluate the selected checkpoint into paper-table JSON.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${BASELINE_DIR}/.." && pwd)}"

DATA_DIR="${DATA_DIR:-/opt/dlami/nvme/cs2-data}"
RUN_NAME="${RUN_NAME:-diamond-b1-lowres-paper}"
OUT_DIR="${OUT_DIR:-${BASELINE_DIR}/runs/${RUN_NAME}}"
CONFIG="${CONFIG:-${BASELINE_DIR}/configs/diamond_csgo_lowres_paper.json}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-128}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

cd "${BASELINE_DIR}"

echo "==> Training ${RUN_NAME}"
echo "    data:   ${DATA_DIR}"
echo "    config: ${CONFIG}"
echo "    out:    ${OUT_DIR}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  uv run --project "${PROJECT_ROOT}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m src.train \
    --config "${CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    "$@"
else
  uv run --project "${PROJECT_ROOT}" python -m src.train \
    --config "${CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    "$@"
fi

CKPT="${OUT_DIR}/best.pt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${OUT_DIR}/latest.pt"
fi

echo "==> Evaluating ${CKPT}"
uv run --project "${PROJECT_ROOT}" python -m src.evaluate \
  --config "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}/eval_${EVAL_SPLIT}" \
  --split "${EVAL_SPLIT}" \
  --max-batches "${EVAL_MAX_BATCHES}"

echo "==> Metrics: ${OUT_DIR}/eval_${EVAL_SPLIT}/metrics.json"
