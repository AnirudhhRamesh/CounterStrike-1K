#!/usr/bin/env bash
# Evaluate the pre-test resource-amended DIAMOND step-20k endpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/cs2_train/configs/diamond_cs1k_dust2_360p_rebuttal_v1.json}"
AMENDMENT="${AMENDMENT:-${REPO_ROOT}/cs2_train/configs/diamond_cs1k_dust2_360p_20k_resource_amendment_v1.json}"
DATA_DIR="${DATA_DIR:-/home/ubuntu/projects/cs2_clean/data/cs1k-360p}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the existing DIAMOND training run}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EXPECTED_ANALYSIS_COMMIT="${EXPECTED_ANALYSIS_COMMIT:?Set EXPECTED_ANALYSIS_COMMIT to the reviewed public commit}"
ENDPOINT_ROOT="${ENDPOINT_ROOT:-${RUN_ROOT}/evaluation_20k}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/true/step_0020000.pt}"
REVIEW_BUCKET="${REVIEW_BUCKET:-cs2-wm-rollout-preview-377114445113}"
REVIEW_PREFIX="${REVIEW_PREFIX:-diamond-cs1k/rebuttal-v2}"
REVIEW_RUN_ID="${REVIEW_RUN_ID:-diamond-cs1k-dust2-360p-rebuttal-v2}"

EXPECTED_TRAINING_COMMIT=34524f6b6f1f805200d72ab4e77f3a55dd6415f8
EXPECTED_CONFIG_SHA256=9f612a74f09697c2cde4cfdc602f3bbd95b5be41f1319f7d58609ebc04102fa5
EXPECTED_MANIFEST_SHA256=33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e
EXPECTED_PROVENANCE_SHA256=3f6419f9414576c88773874c8009c0814e827186c8782831cc373a27be70c2ef
MANIFEST="${DATA_DIR}/manifest_dust2_confirmatory_spatial_v1.parquet"
PROVENANCE="${DATA_DIR}/manifest_dust2_confirmatory_spatial_v1.provenance.json"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python environment not found: ${PYTHON_BIN}"
[[ -d "${RUN_ROOT}" ]] || fail "Training run is absent: ${RUN_ROOT}"
[[ ! -e "${ENDPOINT_ROOT}" ]] || fail "Endpoint output already exists: ${ENDPOINT_ROOT}"
[[ -f "${CHECKPOINT}" ]] || fail "Atomic step-20k checkpoint is absent: ${CHECKPOINT}"

cd "${REPO_ROOT}"
analysis_commit="$(git rev-parse HEAD)"
[[ "${analysis_commit}" == "${EXPECTED_ANALYSIS_COMMIT}" ]] ||
  fail "analysis commit is ${analysis_commit}, expected ${EXPECTED_ANALYSIS_COMMIT}"
[[ -z "$(git status --porcelain=v1)" ]] || fail "analysis checkout is dirty"
[[ "$(cat "${RUN_ROOT}/provenance/training_commit.txt")" == "${EXPECTED_TRAINING_COMMIT}" ]] ||
  fail "training commit does not match the frozen trajectory"
[[ ! -s "${RUN_ROOT}/provenance/git_status.txt" ]] ||
  fail "training checkout was not clean at launch"
[[ "$(sha256sum "${CONFIG}" | awk '{print $1}')" == "${EXPECTED_CONFIG_SHA256}" ]] ||
  fail "original training/evaluator config drifted"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA256}" ]] ||
  fail "confirmatory manifest drifted"
[[ "$(sha256sum "${PROVENANCE}" | awk '{print $1}')" == "${EXPECTED_PROVENANCE_SHA256}" ]] ||
  fail "confirmatory provenance drifted"

checkpoint_step="$(
  "${PYTHON_BIN}" - "${CHECKPOINT}" <<'PY'
import sys

import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(payload.get("step", -1)))
PY
)"
[[ "${checkpoint_step}" == "20000" ]] ||
  fail "checkpoint payload step is ${checkpoint_step}, expected 20000"

mkdir -p "${ENDPOINT_ROOT}/provenance"
printf '%s\n' "${analysis_commit}" >"${ENDPOINT_ROOT}/provenance/analysis_commit.txt"
printf '%s\n' "${EXPECTED_TRAINING_COMMIT}" >"${ENDPOINT_ROOT}/provenance/training_commit.txt"
sha256sum "${CONFIG}" "${AMENDMENT}" "${CHECKPOINT}" "${MANIFEST}" "${PROVENANCE}" \
  >"${ENDPOINT_ROOT}/provenance/input_sha256.txt"
cp "${AMENDMENT}" "${ENDPOINT_ROOT}/provenance/resource_amendment.json"
git status --porcelain=v1 >"${ENDPOINT_ROOT}/provenance/analysis_git_status.txt"
nvidia-smi -q >"${ENDPOINT_ROOT}/provenance/nvidia_smi_q.txt"

for window_mode in midpoint first-death; do
  eval_dir="${ENDPOINT_ROOT}/${window_mode}"
  command=(
    "${PYTHON_BIN}" -m cs2_train.src.evaluate_action_sensitivity
    --config "${CONFIG}"
    --data-dir "${DATA_DIR}"
    --checkpoint "${CHECKPOINT}"
    --out-dir "${eval_dir}"
    --window-mode "${window_mode}"
    --eval-seeds 37 41 43
    --action-modes true shuffled zeros
    --expected-samples 690
    --weights ema
    --save-rollout-archive
  )
  mkdir -p "${eval_dir}"
  printf '%q ' "${command[@]}" >"${eval_dir}/command.sh"
  printf '\n' >>"${eval_dir}/command.sh"
  "${command[@]}" 2>&1 | tee "${eval_dir}/evaluator.log"

  motion_command=(
    "${PYTHON_BIN}" -m cs2_train.src.evaluate_rollout_motion
    --archive-dir "${eval_dir}/rollout_archive"
    --out-dir "${eval_dir}/motion_metrics"
    --bootstrap-replicates 10000
  )
  printf '%q ' "${motion_command[@]}" >"${eval_dir}/motion_command.sh"
  printf '\n' >>"${eval_dir}/motion_command.sh"
  "${motion_command[@]}" 2>&1 | tee "${eval_dir}/motion.log"
done

touch "${ENDPOINT_ROOT}/COMPLETE"
if [[ -n "${REVIEW_BUCKET}" ]]; then
  "${PYTHON_BIN}" -m cs2_train.scripts.publish_diamond_20k_review \
    --endpoint-root "${ENDPOINT_ROOT}" \
    --bucket "${REVIEW_BUCKET}" \
    --prefix "${REVIEW_PREFIX}" \
    --run-id "${REVIEW_RUN_ID}" \
    --step 20000 \
    2>&1 | tee "${ENDPOINT_ROOT}/publish_review.log"
fi
echo "DIAMOND step-20k endpoint and RAFT evaluation complete: ${ENDPOINT_ROOT}"
