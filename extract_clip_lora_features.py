from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
from PIL import Image

try:
    import clip
except ImportError as exc:
    raise ImportError("Missing dependency: clip.") from exc

from train_clip_lora import inject_lora_into_clip_visual


def read_ids_from_csv(csv_path: str | Path) -> list[str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    video_ids: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value:
                video_ids.append(value)
    return video_ids


def load_clip_with_lora(
    clip_model_name: str,
    download_root: str | Path,
    lora_path: str | Path,
    device: torch.device,
):
    model, preprocess = clip.load(clip_model_name, device=device, download_root=str(download_root))
    checkpoint = torch.load(lora_path, map_location="cpu")

    rank = int(checkpoint["rank"])
    alpha = float(checkpoint["alpha"])
    dropout = float(checkpoint["dropout"])

    inject_lora_into_clip_visual(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )

    state_dict = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected LoRA keys: {unexpected}")

    model = model.to(device)
    model.eval()
    return model, preprocess


def sample_middle_frame_indices(video_path: str | Path, max_seconds: int = 30) -> tuple[list[int], float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid fps for video: {video_path}")
    if total_frames <= 0:
        raise ValueError(f"Invalid frame count for video: {video_path}")

    duration_seconds = total_frames / fps
    usable_seconds = min(int(duration_seconds), max_seconds)

    frame_indices: list[int] = []
    for second_idx in range(usable_seconds):
        middle_frame = int(round((second_idx + 0.5) * fps))
        middle_frame = min(middle_frame, total_frames - 1)
        frame_indices.append(middle_frame)

    return frame_indices, fps, total_frames


def extract_video_feature(
    video_path: str | Path,
    model,
    preprocess,
    device: torch.device,
    max_seconds: int = 30,
) -> torch.Tensor:
    video_path = Path(video_path)
    frame_indices, _, _ = sample_middle_frame_indices(video_path, max_seconds=max_seconds)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    features: list[torch.Tensor] = []
    try:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_input = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model.encode_image(image_input).squeeze(0)
            features.append(feature.detach().to("cpu", dtype=torch.float16))
    finally:
        cap.release()

    if not features:
        raise ValueError(f"No valid frames were extracted from: {video_path}")

    feature_tensor = torch.stack(features, dim=0)
    if feature_tensor.shape[0] < max_seconds:
        pad = torch.zeros((max_seconds - feature_tensor.shape[0], feature_tensor.shape[1]), dtype=feature_tensor.dtype)
        feature_tensor = torch.cat([feature_tensor, pad], dim=0)
    else:
        feature_tensor = feature_tensor[:max_seconds]

    return feature_tensor


def save_features_for_csv(
    csv_path: str | Path,
    video_dir: str | Path,
    output_dir: str | Path,
    model,
    preprocess,
    device: torch.device,
    max_videos: int | None = None,
) -> int:
    video_ids = read_ids_from_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    total = len(video_ids) if max_videos is None else min(len(video_ids), max_videos)
    for index, video_id in enumerate(video_ids, start=1):
        video_path = Path(video_dir) / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"[skip] missing video: {video_path}")
            continue

        feature = extract_video_feature(video_path, model, preprocess, device, max_seconds=30)
        save_path = output_dir / f"{video_id}.pt"
        torch.save(feature, save_path)
        processed += 1
        print(f"[{processed}/{total}] saved {save_path} shape={tuple(feature.shape)} dtype={feature.dtype}")

        if max_videos is not None and processed >= max_videos:
            break

    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract LoRA-adapted CLIP video features.")
    parser.add_argument("--text_mode", choices=["action", "detail"], default="detail")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--train_csv", type=str, default="dataset/panda/train_data.csv")
    parser.add_argument("--test_csv", type=str, default="dataset/panda/test_data.csv")
    parser.add_argument("--video_dir", type=str, default="dataset/panda/videos")
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
    parser.add_argument("--download_root", type=str, default="pretrained/vit")
    parser.add_argument("--lora_root", type=str, default="pretrained/lora")
    parser.add_argument("--output_root", type=str, default="dataset/panda")
    parser.add_argument("--max_videos", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lora_path = Path(args.lora_root) / f"lora_{args.text_mode}.pt"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

    model, preprocess = load_clip_with_lora(
        clip_model_name=args.clip_model_name,
        download_root=args.download_root,
        lora_path=lora_path,
        device=device,
    )

    output_dir = Path(args.output_root) / f"lora_{args.text_mode}"

    if args.split in {"train", "both"}:
        save_features_for_csv(
            csv_path=args.train_csv,
            video_dir=args.video_dir,
            output_dir=output_dir,
            model=model,
            preprocess=preprocess,
            device=device,
            max_videos=args.max_videos,
        )

    if args.split in {"test", "both"}:
        save_features_for_csv(
            csv_path=args.test_csv,
            video_dir=args.video_dir,
            output_dir=output_dir,
            model=model,
            preprocess=preprocess,
            device=device,
            max_videos=args.max_videos,
        )


if __name__ == "__main__":
    main()
