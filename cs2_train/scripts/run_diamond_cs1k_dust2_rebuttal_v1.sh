#!/usr/bin/env bash
# Frozen matched-compute DIAMOND-CSGO rebuttal run followed by confirmatory
# midpoint and first-death evaluation. Run from a clean, pinned checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/cs2_train/configs/diamond_cs1k_dust2_360p_rebuttal_v1.json}"
DATA_DIR="${DATA_DIR:-/home/ubuntu/projects/cs2_clean/data/cs1k-360p}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/projects/diamond_runs/diamond-cs1k-dust2-360p-rebuttal-v1}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
REVIEW_S3_URI="${REVIEW_S3_URI:-s3://cs2-wm-rollout-preview-377114445113/diamond-cs1k/rebuttal-v1}"
REVIEW_RUN_ID="${REVIEW_RUN_ID:-diamond-cs1k-dust2-360p-rebuttal-v1}"
MANIFEST="${DATA_DIR}/manifest_dust2_confirmatory_spatial_v1.parquet"
PROVENANCE="${DATA_DIR}/manifest_dust2_confirmatory_spatial_v1.provenance.json"
EXPECTED_MANIFEST_SHA256="33abbb623072932431871a612620110c473d4b664c52010e5763c273c6daf10e"
EXPECTED_PROVENANCE_SHA256="3f6419f9414576c88773874c8009c0814e827186c8782831cc373a27be70c2ef"

cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to run from a dirty tracked worktree." >&2
  git status --short >&2
  exit 2
fi

echo "${EXPECTED_MANIFEST_SHA256}  ${MANIFEST}" | sha256sum --check -
echo "${EXPECTED_PROVENANCE_SHA256}  ${PROVENANCE}" | sha256sum --check -

mkdir -p "${RUN_ROOT}/provenance"
git rev-parse HEAD > "${RUN_ROOT}/provenance/training_commit.txt"
git status --short > "${RUN_ROOT}/provenance/git_status.txt"
sha256sum "${CONFIG}" "${MANIFEST}" "${PROVENANCE}" > "${RUN_ROOT}/provenance/input_sha256.txt"
cp "${CONFIG}" "${RUN_ROOT}/provenance/config.json"
cp "${PROVENANCE}" "${RUN_ROOT}/provenance/dataset_provenance.json"
"${PYTHON_BIN}" - <<'PY' > "${RUN_ROOT}/provenance/python_environment.txt"
from importlib.metadata import distributions

packages = {
    f"{distribution.metadata['Name']}=={distribution.version}"
    for distribution in distributions()
    if distribution.metadata["Name"]
}
print("\n".join(sorted(packages, key=str.casefold)))
PY
"${PYTHON_BIN}" - <<'PY' > "${RUN_ROOT}/provenance/torch_environment.json"
import json
import platform

import torch

print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0),
}, indent=2))
PY
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_q.txt"

for arm in true shuffled; do
  out_dir="${RUN_ROOT}/${arm}"
  mkdir -p "${out_dir}"
  command=(
    "${PYTHON_BIN}" -m cs2_train.src.train
    --config "${CONFIG}"
    --data-dir "${DATA_DIR}"
    --out-dir "${out_dir}"
    --action-mode "${arm}"
    --review-s3-uri "${REVIEW_S3_URI}"
    --review-run-id "${REVIEW_RUN_ID}"
    --review-arm "${arm}"
  )
  printf '%q ' "${command[@]}" > "${out_dir}/command.sh"
  printf '\n' >> "${out_dir}/command.sh"
  "${command[@]}" 2>&1 | tee -a "${out_dir}/launcher.log"
done

for arm in true shuffled; do
  checkpoint="${RUN_ROOT}/${arm}/latest.pt"
  for window_mode in midpoint first-death; do
    eval_dir="${RUN_ROOT}/${arm}/evaluation/${window_mode}"
    command=(
      "${PYTHON_BIN}" -m cs2_train.src.evaluate_action_sensitivity
      --config "${CONFIG}"
      --data-dir "${DATA_DIR}"
      --checkpoint "${checkpoint}"
      --out-dir "${eval_dir}"
      --window-mode "${window_mode}"
    )
    mkdir -p "${eval_dir}"
    printf '%q ' "${command[@]}" > "${eval_dir}/command.sh"
    printf '\n' >> "${eval_dir}/command.sh"
    "${command[@]}" 2>&1 | tee -a "${eval_dir}/evaluator.log"
  done
done

echo "DIAMOND rebuttal run and confirmatory evaluation complete: ${RUN_ROOT}"
