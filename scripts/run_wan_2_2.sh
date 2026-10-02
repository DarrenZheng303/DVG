#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
WAN_ROOT="${REPO_ROOT}/Wan2.2"
PROMPT_LIST_T2V="${PROMPT_LIST_T2V:-${REPO_ROOT}/prompts_t2v.jsonl}"
PROMPT_LIST_I2V="${PROMPT_LIST_I2V:-${REPO_ROOT}/prompts_i2v.jsonl}"
IMAGE_PATH="${IMAGE_PATH:-${WAN_ROOT}/examples/i2v_input.JPG}"

##### variables to edit
# Enable torch.compile to speed up subsequent runs after the first compilation.
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"
# Enable DVG and set the computation budget ratio.
ENABLE_DVG="${ENABLE_DVG:-true}"              # only valid for t2v
DVG_BUDGET="${DVG_BUDGET:-0.5}"

ENABLE_TAE="${ENABLE_TAE:-false}"

TASK_MODE="${TASK_MODE:-t2v}"                  # t2v or i2v
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5.0}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-false}"       # auto / true / false
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-29512}"
ULYSSES_SIZE="${ULYSSES_SIZE:-2}"
ENABLE_DIT_FSDP="${ENABLE_DIT_FSDP:-true}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/Wan2.2-T2V-A14B}"
##### variables to edit

NPROC_PER_NODE="${NPROC_PER_NODE:-${ULYSSES_SIZE}}"

if [[ "${TASK_MODE}" == "i2v" ]]; then
  TASK_NAME="i2v-A14B"
  PROMPT_LIST="${PROMPT_LIST_I2V}"
  DEFAULT_MODEL_PATH="${REPO_ROOT}/checkpoints/Wan2.2-I2V-A14B"
  ENABLE_DVG="false"
else
  TASK_NAME="t2v-A14B"
  PROMPT_LIST="${PROMPT_LIST_T2V}"
  DEFAULT_MODEL_PATH="${REPO_ROOT}/checkpoints/Wan2.2-T2V-A14B"
fi

if [[ "${MODEL_PATH}" == "${REPO_ROOT}/checkpoints/Wan2.2-T2V-A14B" && "${TASK_MODE}" == "i2v" ]]; then
  MODEL_PATH="${DEFAULT_MODEL_PATH}"
fi

if [[ "${TASK_MODE}" != "t2v" && "${ENABLE_DVG}" == "true" ]]; then
  echo "Error: ENABLE_DVG=true is only supported for TASK_MODE=t2v." >&2
  exit 1
fi


OUTPUT_SUFFIX=""
if [[ "${ENABLE_DVG}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_dvg${DVG_BUDGET}"
fi
if [[ "${ENABLE_TAE}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_tae"
fi
if [[ "${ENABLE_TORCH_COMPILE}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_compiled"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/wan22_${TASK_MODE}_${SIZE//\*/x}${OUTPUT_SUFFIX}}"
LOG_FILE="${OUTPUT_DIR}/run.log"

export CUDA_VISIBLE_DEVICES
if [[ -n "${CONDA_ENV_PATH}" ]]; then
  export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

cd "${REPO_ROOT}"

CMD=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  "${WAN_ROOT}/generate.py"
  --task "${TASK_NAME}"
  --size "${SIZE}"
  --ckpt_dir "${MODEL_PATH}"
  --prompt_file "${PROMPT_LIST}"
  --output_dir "${OUTPUT_DIR}"
  --sample_steps "${NUM_INFERENCE_STEPS}"
  --sample_shift "${SAMPLE_SHIFT}"
  --sample_guide_scale "${GUIDE_SCALE}"
  --frame_num "${FRAME_NUM}"
  --base_seed "42"
  --convert_model_dtype
  --enable_torch_compile "${ENABLE_TORCH_COMPILE}"
  --ulysses_size "${ULYSSES_SIZE}"
)

if [[ "${OFFLOAD_MODEL}" != "auto" ]]; then
  CMD+=(--offload_model "${OFFLOAD_MODEL}")
fi

if [[ "${TASK_MODE}" == "i2v" ]]; then
  CMD+=(--image "${IMAGE_PATH}")
fi

CMD+=(--enable_tae "${ENABLE_TAE}")

if [[ "${ENABLE_DVG}" == "true" ]]; then
  CMD+=(
    --enable_dvg
    --dvg_budget "${DVG_BUDGET}"
  )
fi

if [[ "${ENABLE_DIT_FSDP}" == "true" ]]; then
  CMD+=(--dit_fsdp)
fi

"${CMD[@]}"
