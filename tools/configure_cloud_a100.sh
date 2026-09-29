#!/usr/bin/env bash
set -euo pipefail

# Generate a cloud-local training config without changing the tracked training
# config or dataset manifest. The generated files live outside the repository.

DEFAULT_ROOT="/public/home/zuozuokan/lingbot_workspace"
ROOT="${DEFAULT_ROOT}"
RUN_NAME="robotwin_clean_a100_${USER:-user}"
MICRO_BATCH=1
GLOBAL_BATCH=32
FORCE=0

usage() {
  cat <<'USAGE'
Usage: bash tools/configure_cloud_a100.sh [options]

Options:
  --root PATH          Workspace root (default: /public/home/zuozuokan/lingbot_workspace)
  --run-name NAME      Unique output/config name
  --micro-batch N      Per-GPU micro batch: use 1 first; try 2 only after a smoke test
  --global-batch N     Effective global batch (default: 32)
  --force              Overwrite an existing generated config (never deletes checkpoints)
  -h, --help           Show this help

Example:
  bash tools/configure_cloud_a100.sh \
    --run-name robotwin_clean_a100_zuozuokan \
    --micro-batch 1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="${2:?--root requires a path}"
      shift 2
      ;;
    --run-name)
      RUN_NAME="${2:?--run-name requires a name}"
      shift 2
      ;;
    --micro-batch)
      MICRO_BATCH="${2:?--micro-batch requires an integer}"
      shift 2
      ;;
    --global-batch)
      GLOBAL_BATCH="${2:?--global-batch requires an integer}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${ROOT}" != /* ]]; then
  echo "--root must be an absolute path: ${ROOT}" >&2
  exit 2
fi
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--run-name may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
if [[ ! "${MICRO_BATCH}" =~ ^[1-9][0-9]*$ || ! "${GLOBAL_BATCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--micro-batch and --global-batch must be positive integers." >&2
  exit 2
fi
if (( GLOBAL_BATCH % MICRO_BATCH != 0 )); then
  echo "global batch (${GLOBAL_BATCH}) must be divisible by micro batch (${MICRO_BATCH})." >&2
  exit 2
fi

GRAD_ACC=$((GLOBAL_BATCH / MICRO_BATCH))
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_CONFIG="${REPO_ROOT}/configs/vla/robotwin/robotwin_clean_unfreeze_vision_bf16_masked_gbs32.yaml"

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "Source config not found: ${SOURCE_CONFIG}" >&2
  exit 1
fi

CONFIG_DIR="${ROOT}/local_configs/${RUN_NAME}"
CONFIG_PATH="${CONFIG_DIR}/train.yaml"
MANIFEST_PATH="${CONFIG_DIR}/robotwin_clean_only.txt"
LAUNCH_PATH="${CONFIG_DIR}/launch_train.sh"
OUTPUT_DIR="${ROOT}/train_outputs/${RUN_NAME}"
DATASET_DIR="${ROOT}/datasets/RoboTwin_lerobot_v30"
BASE_MODEL_DIR="${ROOT}/models/robbyant--lingbot-vla-v2-6b/snapshots/master"
QWEN_DIR="${ROOT}/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
MOGE_PATH="${ROOT}/models/depth/moge2-vitb-normal.pt"

if [[ -e "${CONFIG_PATH}" && "${FORCE}" != "1" ]]; then
  echo "Generated config already exists: ${CONFIG_PATH}" >&2
  echo "Use --force only if you intentionally want to regenerate it." >&2
  exit 1
fi

mkdir -p \
  "${CONFIG_DIR}" \
  "${ROOT}/datasets" \
  "${ROOT}/models/depth" \
  "${ROOT}/train_outputs"

printf 'robotwin %s\n' "${DATASET_DIR}" > "${MANIFEST_PATH}"

python3 - \
  "${SOURCE_CONFIG}" \
  "${CONFIG_PATH}" \
  "${ROOT}" \
  "${MANIFEST_PATH}" \
  "${OUTPUT_DIR}" \
  "${MICRO_BATCH}" \
  "${GRAD_ACC}" \
  "${GLOBAL_BATCH}" <<'PY'
import re
import sys
from pathlib import Path

(
    source_path,
    output_path,
    root,
    manifest_path,
    train_output_dir,
    micro_batch,
    grad_acc,
    global_batch,
) = sys.argv[1:]

text = Path(source_path).read_text(encoding="utf-8")
text = text.replace("/scratch/YF_Data/lingbot_workspace", root)

replacements = (
    (r"(?m)^(\s*train_path:)\s*.*$", rf"\1 {manifest_path}"),
    (r"(?m)^(\s*output_dir:)\s*.*$", rf"\1 {train_output_dir}"),
    (r"(?m)^(\s*micro_batch_size:)\s*.*$", rf"\1 {micro_batch}"),
    (r"(?m)^(\s*gradient_accumulation_steps:)\s*.*$", rf"\1 {grad_acc}"),
    (r"(?m)^(\s*global_batch_size:)\s*.*$", rf"\1 {global_batch}"),
)
for pattern, replacement in replacements:
    text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match for {pattern!r}, found {count}")

Path(output_path).write_text(text, encoding="utf-8")
PY

cat > "${LAUNCH_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="\${MASTER_PORT:-62500}"
bash train.sh tasks/vla/train_lingbotvla.py "${CONFIG_PATH}"
EOF
chmod +x "${LAUNCH_PATH}"

echo
echo "Cloud config generated successfully"
echo "  Root:                  ${ROOT}"
echo "  Config:                ${CONFIG_PATH}"
echo "  Dataset manifest:      ${MANIFEST_PATH}"
echo "  Output directory:      ${OUTPUT_DIR}"
echo "  micro_batch_size:      ${MICRO_BATCH}"
echo "  gradient_accumulation: ${GRAD_ACC}"
echo "  global_batch_size:     ${GLOBAL_BATCH}"
echo "  Launcher:              ${LAUNCH_PATH}"

missing=0
check_path() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}" ]]; then
    printf '  [OK]      %-22s %s\n' "${label}" "${path}"
  else
    printf '  [MISSING] %-22s %s\n' "${label}" "${path}"
    missing=1
  fi
}

echo
echo "Required asset check"
check_path "${DATASET_DIR}/meta" "dataset meta"
check_path "${DATASET_DIR}/data" "dataset data"
check_path "${DATASET_DIR}/videos" "dataset videos"
check_path "${BASE_MODEL_DIR}/config.json" "base model config"
check_path "${BASE_MODEL_DIR}/depth/model.pt" "LingBot-Depth"
check_path "${BASE_MODEL_DIR}/dino_video/teacher_step_10000.pth" "DINO-VIDEO weights"
check_path "${BASE_MODEL_DIR}/dino_video/config.yaml" "DINO-VIDEO config"
check_path "${QWEN_DIR}/config.json" "Qwen config"
check_path "${QWEN_DIR}/tokenizer_config.json" "Qwen tokenizer"
check_path "${QWEN_DIR}/preprocessor_config.json" "Qwen processor"
check_path "${MOGE_PATH}" "MoGe weights"

if ! find "${BASE_MODEL_DIR}" -maxdepth 1 -type f \
    \( -name 'model.safetensors' -o -name 'model.safetensors.index.json' -o -name 'pytorch_model.bin' -o -name 'pytorch_model.bin.index.json' \) \
    -print -quit 2>/dev/null | grep -q .; then
  printf '  [MISSING] %-22s %s\n' "base model weights" "${BASE_MODEL_DIR}"
  missing=1
else
  printf '  [OK]      %-22s %s\n' "base model weights" "${BASE_MODEL_DIR}"
fi

echo
if [[ "${missing}" == "1" ]]; then
  echo "Some assets are missing. Download/copy them before training."
  echo "Re-run this script with --force to repeat the check."
else
  echo "All required asset entry points are present."
  echo "Start training with:"
  echo "  bash \"${LAUNCH_PATH}\""
fi

if (( MICRO_BATCH > 1 )); then
  echo
  echo "WARNING: micro_batch_size=${MICRO_BATCH} is a throughput candidate, not the safe default."
  echo "Run a short smoke test and watch nvidia-smi before committing to the full run."
fi
