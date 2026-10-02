#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
HV15_ROOT="${REPO_ROOT}/HunyuanVideo-1.5"
PROMPT_LIST_T2V="${PROMPT_LIST_T2V:-${REPO_ROOT}/prompts_t2v.jsonl}"
PROMPT_LIST_I2V="${PROMPT_LIST_I2V:-${REPO_ROOT}/prompts_i2v.jsonl}"

##### variable to edit
# Enable torch.compile to speed up subsequent runs after the first compilation.
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-true}"
# Enable DVG and set the computation budget ratio.
ENABLE_DVG="${ENABLE_DVG:-true}" # enable dynamic video gen
DVG_BUDGET="${DVG_BUDGET:-0.5}" # dvg budget
# Enable TAE for higher speedup; may reduce quality for I2V.
ENABLE_TAE="${ENABLE_TAE:-false}"

TASK_MODE="${TASK_MODE:-i2v}"                  # t2v or i2v
ENABLE_STEP_DISTILL="${ENABLE_STEP_DISTILL:-false}" 
VIDEO_LENGTH="${VIDEO_LENGTH:-121}"
RESOLUTION="${RESOLUTION:-480p}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/HunyuanVideo-1.5}"
##### variable to edit

if [[ "${TASK_MODE}" == "i2v" ]]; then
  PROMPT_LIST="${PROMPT_LIST_I2V}"
else
  PROMPT_LIST="${PROMPT_LIST_T2V}"
fi

if [[ "${TASK_MODE}" == "t2v" && "${ENABLE_STEP_DISTILL}" == "true" ]]; then
  echo "Error: t2v does not support ENABLE_STEP_DISTILL=true in HunyuanVideo1.5." >&2
  exit 1
fi

if [[ "${ENABLE_STEP_DISTILL}" == "true" ]]; then
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-12}"
else
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
fi

OUTPUT_SUFFIX=""
if [[ "${ENABLE_DVG}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_dvg"
fi
if [[ "${ENABLE_TAE}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_tae"
fi
if [[ "${ENABLE_STEP_DISTILL}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_step_dist"
fi
if [[ "${ENABLE_TORCH_COMPILE}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_compiled"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/hy15_${TASK_MODE}_${RESOLUTION}${OUTPUT_SUFFIX}}"
LOG_FILE="${OUTPUT_DIR}/run.log"


export CUDA_VISIBLE_DEVICES
if [[ -n "${CONDA_ENV_PATH}" ]]; then
  export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

cd "${REPO_ROOT}"

torchrun --nproc_per_node=1 --master_port 29502 "${HV15_ROOT}/generate.py" \
  --prompt_file "${PROMPT_LIST}" \
  --resolution "${RESOLUTION}" \
  --model_path "${MODEL_PATH}" \
  --aspect_ratio "16:9" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --video_length "${VIDEO_LENGTH}" \
  --seed "42" \
  --sr false \
  --rewrite false \
  --cfg_distilled false \
  --enable_step_distill "${ENABLE_STEP_DISTILL}" \
  --enable_torch_compile "${ENABLE_TORCH_COMPILE}" \
  --offloading false \
  --group_offloading false \
  --overlap_group_offloading false \
  --dtype bf16 \
  --enable_tae "${ENABLE_TAE}" \
  --output_path "${OUTPUT_DIR}/output.mp4" \
  --enable_dvg "${ENABLE_DVG}" \
  --dvg_budget "${DVG_BUDGET}"
