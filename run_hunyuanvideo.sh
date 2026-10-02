#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
HV_ROOT="${REPO_ROOT}/HunyuanVideo"

##### variable to edit
# Enable torch.compile to speed up subsequent runs after the first compilation.
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-true}"
# Enable DVG and set the computation budget ratio.
ENABLE_DVG="${ENABLE_DVG:-true}"
DVG_BUDGET="${DVG_BUDGET:-0.5}"
# Enable TAE for higher speedup.
ENABLE_TAE="${ENABLE_TAE:-false}"

VIDEO_LENGTH="${VIDEO_LENGTH:-121}"
RESOLUTION="${RESOLUTION:-720p}"            # 540p or 720p
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
MODEL_BASE="${MODEL_BASE:-${REPO_ROOT}/checkpoints/HunyuanVideo}"
PROMPT_LIST="${PROMPT_LIST:-${REPO_ROOT}/prompts_t2v.jsonl}"
##### variable to edit

if [[ "${RESOLUTION}" == "720p" ]]; then
  VIDEO_H=720
  VIDEO_W=1280
elif [[ "${RESOLUTION}" == "540p" ]]; then
  VIDEO_H=544
  VIDEO_W=960
fi

OUTPUT_SUFFIX=""
if [[ "${ENABLE_DVG}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_dvg"
fi
if [[ "${ENABLE_TAE}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_tae"
fi
if [[ "${ENABLE_TORCH_COMPILE}" == "true" ]]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_compiled"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/hy_t2v_${RESOLUTION}${OUTPUT_SUFFIX}}"
LOG_FILE="${OUTPUT_DIR}/run.log"

export CUDA_VISIBLE_DEVICES
export MODEL_BASE
if [[ -n "${CONDA_ENV_PATH}" ]]; then
  export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

cd "${REPO_ROOT}"

CMD=(
  python "${HV_ROOT}/sample_video.py"
  --model-base "${MODEL_BASE}"
  --prompt-file "${PROMPT_LIST}"
  --video-size "${VIDEO_H}" "${VIDEO_W}"
  --video-length "${VIDEO_LENGTH}"
  --infer-steps 50
  --seed 42
  --flow-shift 7.0
  --flow-reverse
  --embedded-cfg-scale 6.0
  --save-path "${OUTPUT_DIR}"
  --dvg_budget "${DVG_BUDGET}"
)

if [[ "${ENABLE_TORCH_COMPILE}" == "true" ]]; then
  CMD+=(--enable_torch_compile)
fi

if [[ "${ENABLE_DVG}" == "true" ]]; then
  CMD+=(--enable_dvg)
fi
if [[ "${ENABLE_TAE}" == "true" ]]; then
  CMD+=(--enable_tae)
fi

"${CMD[@]}"
