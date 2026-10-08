#!/bin/bash

set -euo pipefail
set -x

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
export HF_HUB_DISABLE_TELEMETRY=1 
export DISABLE_TELEMETRY=1 

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
export WORKSPACE=$(cd "${WORKSPACE:-$PROJECT_ROOT}" && pwd)
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

# TorchCodec needs FFmpeg shared libraries installed in the active environment.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
else
  NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"
NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=$NPROC_PER_NODE}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=62500}

CONFIG_ARG=${2:-training}
RUN_LABEL=$(basename "$CONFIG_ARG")
RUN_LABEL=${RUN_LABEL%.*}
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_RANDOM_ID=$(python -c 'import uuid; print(uuid.uuid4().hex[:8])')
export LINGBOT_TRAIN_RUN_ID=${LINGBOT_TRAIN_RUN_ID:-${LOG_TIMESTAMP}_${RUN_RANDOM_ID}}
export NODE_RANK
export LINGBOT_TRAIN_RUN_DIR=$(python - "$@" <<'PY'
import json
import os
from pathlib import Path
import sys
import yaml
from lingbotvla.utils.arguments import _string_to_bool, cli_value, training_output_path, workspace_path

arguments = sys.argv[2:]
config = {}
label = Path(sys.argv[1]).stem
if arguments and Path(arguments[0]).suffix in {".yaml", ".yml", ".json"}:
    path = workspace_path(arguments[0])
    label = path.stem
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle) if path.suffix == ".json" else yaml.safe_load(handle)
output = training_output_path(config, arguments, label)
resume = _string_to_bool(cli_value(arguments, "--train.enable_resume", config.get("train", {}).get("enable_resume", False)))
resume = resume or bool(cli_value(arguments, "--train.load_checkpoint_path", config.get("train", {}).get("load_checkpoint_path")))
if int(os.environ["NODE_RANK"]) == 0:
    output.mkdir(parents=True, exist_ok=resume)
elif not output.is_dir():
    raise ValueError(f"Start node 0 first to create the shared run directory: {output}")
print(output)
PY
)
# Do not hide failures inside an export/command substitution.
[ -n "$LINGBOT_TRAIN_RUN_DIR" ] || exit 1
export TRAIN_LOG_FILE=$(python - "$RUN_LABEL" <<'PY'
import os
import sys
from lingbotvla.utils.arguments import workspace_path

directory = workspace_path(os.environ.get("TRAIN_LOG_DIR") or os.environ["LINGBOT_TRAIN_RUN_DIR"])
node_suffix = f"_node{os.environ['NODE_RANK']}" if os.environ["NODE_RANK"] != "0" else ""
path = workspace_path(os.environ.get("TRAIN_LOG_FILE") or directory / f"{sys.argv[1]}_{os.environ['LINGBOT_TRAIN_RUN_ID']}{node_suffix}.log")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8"):
    pass
print(path)
PY
)
[ -n "$TRAIN_LOG_FILE" ] || exit 1
echo "Training stdout/stderr log: $TRAIN_LOG_FILE"
echo "Training visualization run id: $LINGBOT_TRAIN_RUN_ID"

torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT "$@" \
  --train.output_dir "$LINGBOT_TRAIN_RUN_DIR" 2>&1 | tee "$TRAIN_LOG_FILE"
