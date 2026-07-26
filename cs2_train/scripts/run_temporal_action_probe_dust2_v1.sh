#!/usr/bin/env bash
# Freeze the shared real-video temporal action probe used by ARR.
#
# This script reads train and validation only. It never opens the held-out test
# split or a world-model rollout. Run it from a clean, pinned checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/cs2_train/configs/temporal_arr_cs1k_dust2_v1.json}"
DATA_DIR="${DATA_DIR:-/data/cs1k-360p}"
ARR_ROOT="${ARR_ROOT:-${REPO_ROOT}/runs/temporal-arr-cs1k-dust2-v1}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -e "${ARR_ROOT}" ]]; then
  echo "Refusing to overwrite ARR output root: ${ARR_ROOT}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run from a dirty worktree." >&2
  git status --short >&2
  exit 2
fi

read -r \
  manifest_name manifest_sha256 provenance_name provenance_sha256 \
  train_windows train_seed val_windows val_seed decode_height decode_width \
  loader_batch_size loader_workers loader_prefetch frame_batch_size \
  hidden_dim num_layers num_heads dropout batch_size epochs learning_rate \
  weight_decay max_positive_weight probe_seed \
  < <(
    "${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
dataset = config["dataset"]
probe = config["probe"]
print(
    dataset["manifest_name"],
    dataset["manifest_sha256"],
    dataset["provenance_name"],
    dataset["provenance_sha256"],
    dataset["train_windows"],
    dataset["train_sampling_seed"],
    dataset["validation_windows"],
    dataset["validation_sampling_seed"],
    *dataset["decode_resize"],
    dataset["loader_batch_size"],
    dataset["loader_workers"],
    dataset["loader_prefetch_factor"],
    dataset["frame_encoder_batch_size"],
    probe["hidden_dim"],
    probe["num_layers"],
    probe["num_heads"],
    probe["dropout"],
    probe["batch_size"],
    probe["epochs"],
    probe["learning_rate"],
    probe["weight_decay"],
    probe["max_positive_weight"],
    probe["seed"],
)
PY
  )

manifest="${DATA_DIR}/${manifest_name}"
provenance="${DATA_DIR}/${provenance_name}"
echo "${manifest_sha256}  ${manifest}" | sha256sum --check -
echo "${provenance_sha256}  ${provenance}" | sha256sum --check -

train_features="${ARR_ROOT}/features/train"
val_features="${ARR_ROOT}/features/val"
probe_root="${ARR_ROOT}/probe"
mkdir -p "${ARR_ROOT}/provenance"
git rev-parse HEAD > "${ARR_ROOT}/provenance/release_commit.txt"
git status --short > "${ARR_ROOT}/provenance/git_status.txt"
sha256sum "${CONFIG}" "${manifest}" "${provenance}" \
  > "${ARR_ROOT}/provenance/input_sha256.txt"
cp "${CONFIG}" "${ARR_ROOT}/provenance/config.json"
"${PYTHON_BIN}" - <<'PY' > "${ARR_ROOT}/provenance/python_environment.txt"
from importlib.metadata import distributions

packages = {
    f"{distribution.metadata['Name']}=={distribution.version}"
    for distribution in distributions()
    if distribution.metadata["Name"]
}
print("\n".join(sorted(packages, key=str.casefold)))
PY
"${PYTHON_BIN}" - <<'PY' > "${ARR_ROOT}/provenance/torch_environment.json"
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
nvidia-smi -q > "${ARR_ROOT}/provenance/nvidia_smi_q.txt"

train_command=(
  "${PYTHON_BIN}" -m cs2_train.src.extract_temporal_action_features
  --data-dir "${DATA_DIR}"
  --manifest-name "${manifest_name}"
  --split train
  --out-dir "${train_features}"
  --num-windows "${train_windows}"
  --sampling-seed "${train_seed}"
  --batch-size "${loader_batch_size}"
  --num-workers "${loader_workers}"
  --prefetch-factor "${loader_prefetch}"
  --frame-batch-size "${frame_batch_size}"
  --decode-height "${decode_height}"
  --decode-width "${decode_width}"
  --device "${DEVICE}"
)
printf '%q ' "${train_command[@]}" \
  > "${ARR_ROOT}/provenance/train_feature_command.sh"
printf '\n' >> "${ARR_ROOT}/provenance/train_feature_command.sh"
"${train_command[@]}" 2>&1 | tee "${ARR_ROOT}/train_feature_extraction.log"

val_command=(
  "${PYTHON_BIN}" -m cs2_train.src.extract_temporal_action_features
  --data-dir "${DATA_DIR}"
  --manifest-name "${manifest_name}"
  --split val
  --out-dir "${val_features}"
  --num-windows "${val_windows}"
  --sampling-seed "${val_seed}"
  --batch-size "${loader_batch_size}"
  --num-workers "${loader_workers}"
  --prefetch-factor "${loader_prefetch}"
  --frame-batch-size "${frame_batch_size}"
  --decode-height "${decode_height}"
  --decode-width "${decode_width}"
  --device "${DEVICE}"
)
printf '%q ' "${val_command[@]}" \
  > "${ARR_ROOT}/provenance/validation_feature_command.sh"
printf '\n' >> "${ARR_ROOT}/provenance/validation_feature_command.sh"
"${val_command[@]}" 2>&1 | tee "${ARR_ROOT}/validation_feature_extraction.log"

probe_command=(
  "${PYTHON_BIN}" -m cs2_train.src.train_temporal_action_probe
  --train-features "${train_features}"
  --val-features "${val_features}"
  --out-dir "${probe_root}"
  --hidden-dim "${hidden_dim}"
  --num-layers "${num_layers}"
  --num-heads "${num_heads}"
  --dropout "${dropout}"
  --batch-size "${batch_size}"
  --epochs "${epochs}"
  --lr "${learning_rate}"
  --weight-decay "${weight_decay}"
  --max-pos-weight "${max_positive_weight}"
  --seed "${probe_seed}"
  --device "${DEVICE}"
)
printf '%q ' "${probe_command[@]}" \
  > "${ARR_ROOT}/provenance/probe_training_command.sh"
printf '\n' >> "${ARR_ROOT}/provenance/probe_training_command.sh"
"${probe_command[@]}" 2>&1 | tee "${ARR_ROOT}/probe_training.log"

echo "Temporal action probe complete: ${probe_root}/temporal_action_probe.pt"
