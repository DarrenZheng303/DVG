import os
import time
import json
import re
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


def sanitize_filename(value, max_length=100):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    safe = re.sub(r"_+", "_", safe).strip("._")
    if not safe:
        safe = "prompt"
    return safe[:max_length]


def load_prompts_from_jsonl(prompt_file):
    prompt_entries = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {prompt_file}: {exc}"
                ) from exc

            if isinstance(item, str):
                prompt_text = item
            elif isinstance(item, dict):
                prompt_text = item.get("prompt") or item.get("prompt_en") or item.get("text")
                if prompt_text is None:
                    raise ValueError(
                        f"Line {line_no} of {prompt_file} must contain one of 'prompt', 'prompt_en', or 'text'."
                    )
            else:
                raise ValueError(
                    f"Line {line_no} of {prompt_file} must be a JSON string or object, got {type(item).__name__}."
                )

            prompt_entries.append(str(prompt_text).strip())

    if not prompt_entries:
        raise ValueError(f"No prompts found in prompt file: {prompt_file}")

    return prompt_entries


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    if args.prompt_file is None and args.prompt is None:
        raise ValueError("One of `--prompt` or `--prompt-file` must be provided.")
    if args.prompt_file is not None and args.prompt is not None:
        logger.warning(
            "Both `--prompt` and `--prompt-file` were provided. `--prompt-file` will be used."
        )

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if args.prompt_file is not None:
        prompt_entries = load_prompts_from_jsonl(args.prompt_file)
        logger.info(f"Loaded {len(prompt_entries)} prompt(s) from {args.prompt_file}")
    else:
        prompt_entries = [args.prompt]

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

    # Get the updated args
    args = hunyuan_video_sampler.args

    for prompt_index, prompt_text in enumerate(prompt_entries):
        if args.prompt_file is not None:
            logger.info(
                f"Generating prompt {prompt_index + 1}/{len(prompt_entries)} from {args.prompt_file}: {prompt_text}"
            )

        outputs = hunyuan_video_sampler.predict(
            prompt=prompt_text,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            enable_dvg=args.enable_dvg,
            dvg_budget=args.dvg_budget,
        )
        samples = outputs["samples"]

        # Save samples
        if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                prompt_slug = sanitize_filename(outputs["prompts"][i])
                cur_save_path = (
                    f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{prompt_slug}.mp4"
                )
                save_videos_grid(sample, cur_save_path, fps=24)
                logger.info(f"Sample save to: {cur_save_path}")

if __name__ == "__main__":
    main()
