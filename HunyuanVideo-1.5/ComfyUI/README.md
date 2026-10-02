# ComfyUI Usage Guide for HunyuanVideo-1.5

This guide helps you use ComfyUI to run HunyuanVideo-1.5 model.

## 📋 Table of Contents

- [Installation](#installation)
- [Download Models](#download-models)
- [Workflow Templates](#workflow-templates)
- [Community Contributions](#community-contributions)
- [Links](#links)

## 🚀 Installation

Install the latest version of ComfyUI from the official repository:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

For detailed installation instructions, please refer to the [ComfyUI official repository](https://github.com/comfyanonymous/ComfyUI).


## 📥 Download Models

### HunyuanVideo-1.5 Checkpoint

Download the ComfyUI checkpoint from Hugging Face:

**Model**: https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged

## 🎨 Workflow Templates

Find workflow templates at:

- **720p Image-to-Video**: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_hunyuan_video_1.5_720p_i2v.json
- **720p Text-to-Video**: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_hunyuan_video_1.5_720p_t2v.json

**Important**: If you enable the Tiled VAE decoding node, please set the 'temporal_size' to the maximum value (4096) to avoid generating artifacts.


* **Important**: Workflows are still being updated. Please ensure you are using the latest version of the workflow templates to guarantee the best generation results.
* **Important**: The distilled models include both CFG-distilled models and step-distilled models. CFG-distilled models should run with 50 inference steps for best generation quality. Step-distilled models (480p I2V) can run with 8 or 12 steps (recommended) for faster generation, or 4 steps for even faster speed with slightly reduced quality. The recommended configurations are provided in the following table.
* **Important**: Please strictly follow the configuration below to guarantee the best generation results.


### Recommended Inference Configurations

**Important!** The following table is for the best generation quality. If you want to generate faster, you can use the following configurations:


| Model | CFG Scale | Embedded CFG Scale | Flow Shift | Inference Steps |
|-------|-----------|-------------------|------------|-----------------|
| 480p T2V | 6 | None | 5 | 50 |
| 480p I2V | 6 | None | 5 | 50 |
| 720p T2V | 6 | None | 9 | 50 |
| 720p I2V | 6 | None | 7 | 50 |
| 480p T2V CFG Distilled | 1 | None | 5 | 50 |
| 480p I2V CFG Distilled | 1 | None | 5 | 50 |
| 480p I2V Step Distilled | 1 | None | 7 | 8 or 12 (recommended) |
| 720p T2V CFG Distilled | 1 | None | 9 | 50 |
| 720p I2V CFG Distilled | 1 | None | 7 | 50 |
| 720p T2V CFG Distilled Sparse | 1 | None | 9 | 50 |
| 720p I2V CFG Distilled Sparse | 1 | None | 7 | 50 |
| 480→720 SR Step Distilled | 1 | None | 2 | 6 |
| 720→1080 SR Step Distilled | 1 | None | 2 | 8 |


### Prompt Writing Tips

- Use detailed, descriptive prompts for better results
- Refer to the [HunyuanVideo-1.5 Prompt Handbook](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo_1_5_Prompt_Handbook_EN.md) for guidance
- Include details about camera movement, style, lighting, etc.

## 🧩 Community Contributions

If you develop/use HunyuanVideo-1.5 in your ComfyUI projects, welcome to let us know.

- **ComfyUI-MagCache** - [ComfyUI-MagCache](https://github.com/Zehong-Ma/ComfyUI-MagCache): MagCache is a training-free caching approach that accelerates video generation by estimating fluctuating differences among model outputs across timesteps. It achieves 1.7x speedup for HunyuanVideo-1.5 with 20 inference steps.

## 🔗 Links

- **ComfyUI Repository**: https://github.com/comfyanonymous/ComfyUI
- **HunyuanVideo-1.5 Repository**: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
- **Model Checkpoint**: https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged
- **Workflow Templates**: https://github.com/Comfy-Org/workflow_templates/tree/main/templates
- **Prompt Handbook**: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/assets/HunyuanVideo_1_5_Prompt_Handbook_EN.md

---

For issues or questions, please visit the [HunyuanVideo-1.5 repository](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) to submit issues or check documentation.
