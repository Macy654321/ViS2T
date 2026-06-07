from __future__ import annotations

import csv
import json
from pathlib import Path


try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "Missing dependency: opencv-python-headless. "
        "Install it with `python -m pip install opencv-python-headless`."
    ) from exc

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "Missing dependency: torch. "
        "Install it with `python -m pip install torch`."
    ) from exc

from PIL import Image

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError as exc:
    raise ImportError(
        "Missing dependency: transformers. "
        "Install it with `python -m pip install transformers accelerate`."
    ) from exc


DEFAULT_MODEL_PATH = Path(r"E:\pycharmProject\clipcap\qwen\Qwen3.5")
DEFAULT_VIDEO_PATH = Path(r"E:\pycharmProject\clipcap\dataset\panda\videos\D000001.mp4")
DEFAULT_OUTPUT_PATH = Path(r"E:\pycharmProject\clipcap\video_description.txt")
DEFAULT_TEST_CSV_PATH = Path(r"E:\pycharmProject\clipcap\dataset\panda\test_data.csv")
DEFAULT_VIDEO_DIR = Path(r"E:\pycharmProject\clipcap\dataset\panda\videos")
DEFAULT_BATCH_OUTPUT_PATH = Path(r"E:\pycharmProject\clipcap\dataset\panda\test_video_descriptions.json")
DEFAULT_PROMPT = (
    "Briefly describe the overall content of the video in one sentence. "
    "Focus on the main scene and the panda's general behavior. "
    "Do not describe detailed local body movements. "
    "Avoid excessive adjectives and adverbs. "
    "If the action does not clearly change, keep the description as a single sentence."
)

def extract_video_frames(video_path: str | Path, num_frames: int = 8) -> list[Image.Image]:
    """Uniformly sample frames from a video and return them as RGB PIL images."""
    video_path = Path(video_path)
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than 0.")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video has no readable frames: {video_path}")

        sample_count = min(num_frames, total_frames)
        frame_indices = sorted({int(i * total_frames / sample_count) for i in range(sample_count)})

        frames: list[Image.Image] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        if not frames:
            raise ValueError(f"Failed to decode sampled frames from: {video_path}")

        return frames
    finally:
        cap.release()


def load_model_and_processor(model_path: str | Path):
    """Load the Qwen multimodal processor and generation model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()
    return processor, model


def generate_description(
    frames: list[Image.Image],
    processor,
    model,
    prompt: str = DEFAULT_PROMPT,
) -> str:
    """Generate one description from a list of sampled video frames."""
    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in frames),
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
        images=frames,
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


def describe_video_with_qwen(
    video_path: str | Path,
    output_txt_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    num_frames: int = 8,
    prompt: str = DEFAULT_PROMPT,
) -> str:
    """Sample frames from a video and generate a description with Qwen."""
    frames = extract_video_frames(video_path, num_frames=num_frames)
    processor, model = load_model_and_processor(model_path)
    description = generate_description(frames, processor, model, prompt=prompt)

    output_path = Path(output_txt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(description, encoding="utf-8")

    return description


def read_video_ids_from_csv(csv_path: str | Path) -> list[str]:
    """Read video ids from a single-column csv or the first column of a multi-column csv."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    video_ids: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue

            video_id = row[0].strip()
            if not video_id:
                continue

            if video_id.lower() in {"video", "video_id", "video_name", "id"}:
                continue

            video_ids.append(video_id)

    if not video_ids:
        raise ValueError(f"No video ids found in CSV: {csv_path}")

    return video_ids


def describe_videos_from_csv(
    csv_path: str | Path,
    video_dir: str | Path,
    output_json_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    num_frames: int = 8,
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, str]:
    """Generate descriptions for all video ids in a CSV and save them as a JSON mapping."""
    video_ids = read_video_ids_from_csv(csv_path)
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    processor, model = load_model_and_processor(model_path)
    descriptions: dict[str, str] = {}

    total = len(video_ids)
    for index, video_id in enumerate(video_ids, start=1):
        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        frames = extract_video_frames(video_path, num_frames=num_frames)
        descriptions[video_id] = generate_description(frames, processor, model, prompt=prompt)
        print(f"[{index}/{total}] {video_id} done")

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(descriptions, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return descriptions


if __name__ == "__main__":
    result = describe_videos_from_csv(
        csv_path=DEFAULT_TEST_CSV_PATH,
        video_dir=DEFAULT_VIDEO_DIR,
        output_json_path=DEFAULT_BATCH_OUTPUT_PATH,
        model_path=DEFAULT_MODEL_PATH,
        num_frames=8,
    )
    print(f"Saved {len(result)} descriptions to: {DEFAULT_BATCH_OUTPUT_PATH}")
