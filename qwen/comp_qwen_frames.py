from __future__ import annotations

import json
from pathlib import Path
from packaging.version import Version

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "Missing dependency: torch. "
        "Install it with `python -m pip install torch`."
    ) from exc

from PIL import Image

try:
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError as exc:
    raise ImportError(
        "Missing dependency: transformers. "
        "Install it with `python -m pip install transformers accelerate`."
    ) from exc


DEFAULT_MODEL_PATH = Path(r"E:\pycharmProject\clipcap\qwen\Qwen3.5")
DEFAULT_FRAME_DIR = Path(r"E:\pycharmProject\clipcap\dataset\panda\frames")
DEFAULT_OUTPUT_PATH = Path(r"E:\pycharmProject\clipcap\dataset\panda\frame_descriptions_sample.json")
MIN_TRANSFORMERS_VERSION = Version("4.57.0.dev0")
ACTION_PROMPT = (
    "Describe the panda's overall action in one short English sentence using the simple present tense. "
    "Only describe the main action or behavior. "
    "Do not mention local body parts. "
    "Vary the sentence structure naturally instead of following a fixed template."
    "Example style: A giant panda sits and eats."
)
DETAIL_PROMPT = (
    "Describe the panda's overall action in one short English sentence using the simple present tense. "
    "Focus on subject, action, and object or scene. "
    "Do not describe local body parts. "
    "Keep it natural and concise. "
    "Vary the sentence structure naturally instead of following a fixed template. "
    "For example, the description may mention what the panda is doing, what it interacts with, or where it is."
)


def load_model_and_processor(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    current_version = Version(transformers.__version__)
    if current_version < MIN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "The local Qwen3.5 checkpoint requires a newer transformers version. "
            f"Current version: {transformers.__version__}. "
            f"Required version: >= {MIN_TRANSFORMERS_VERSION}. "
            "Please upgrade the opcv environment with either "
            "`pip install -U transformers accelerate` or, if that still does not "
            "support qwen3_5, `pip install git+https://github.com/huggingface/transformers.git`."
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    except ValueError as exc:
        if "qwen3_5" in str(exc):
            raise RuntimeError(
                "Failed to load the local Qwen3.5 checkpoint because the installed "
                f"transformers version ({transformers.__version__}) does not support "
                "`qwen3_5`. Upgrade transformers in the opcv environment first."
            ) from exc
        raise

    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()
    return processor, model


def load_image(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    return Image.open(image_path).convert("RGB")


def generate_description(
    image: Image.Image,
    processor,
    model,
    prompt: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, prompt_length:]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def describe_single_image(
    image_path: str | Path,
    processor,
    model,
) -> dict[str, dict[str, str]]:
    image_path = Path(image_path)
    image = load_image(image_path)
    action = generate_description(image, processor, model, ACTION_PROMPT)
    detail = generate_description(image, processor, model, DETAIL_PROMPT)
    return {
        image_path.name: {
            "action": action,
            "detail": detail,
        }
    }


def describe_first_frame(
    frame_dir: str | Path = DEFAULT_FRAME_DIR,
    output_json_path: str | Path = DEFAULT_OUTPUT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, dict[str, str]]:
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")

    image_paths = sorted(
        [path for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if not image_paths:
        raise ValueError(f"No image files found in: {frame_dir}")

    processor, model = load_model_and_processor(model_path)
    result = describe_single_image(image_paths[0], processor, model)

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    return result


def describe_all_frames(
    frame_dir: str | Path = DEFAULT_FRAME_DIR,
    output_json_path: str | Path = DEFAULT_OUTPUT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, dict[str, str]]:
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")

    image_paths = sorted(
        [
            path for path in frame_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if not image_paths:
        raise ValueError(f"No image files found in: {frame_dir}")

    processor, model = load_model_and_processor(model_path)

    results = {}
    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{total}] Processing: {image_path.name}")
        try:
            result = describe_single_image(image_path, processor, model)
            results.update(result)
        except Exception as exc:
            print(f"Failed to process {image_path.name}: {exc}")
            results[image_path.name] = {
                "action": "",
                "detail": "",
                "error": str(exc),
            }

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return results


if __name__ == "__main__":
    results = describe_all_frames()
    print(f"Saved descriptions to: {DEFAULT_OUTPUT_PATH}")
