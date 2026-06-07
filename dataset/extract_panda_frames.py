from pathlib import Path

import cv2
import pandas as pd


CSV_PATH = Path("dataset/panda/test_data.csv")
VIDEO_DIR = Path("dataset/panda/videos")
OUTPUT_DIR = Path("dataset/panda/frames")
TARGET_SECONDS = [5, 15, 25]


def extract_frames_for_video(video_name: str) -> None:
    video_path = VIDEO_DIR / f"{video_name}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps else 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for index, second in enumerate(TARGET_SECONDS, start=1):
            if second > duration_seconds:
                raise ValueError(
                    f"Video {video_name} is only {duration_seconds:.2f}s long, "
                    f"cannot extract frame at {second}s."
                )

            frame_index = int(round(second * fps))
            frame_index = min(frame_index, max(total_frames - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if not success:
                raise RuntimeError(
                    f"Failed to read frame {frame_index} ({second}s) from {video_name}."
                )

            output_path = OUTPUT_DIR / f"{video_name}_{index}.jpg"
            cv2.imwrite(str(output_path), frame)
            print(f"Saved: {output_path}")
    finally:
        cap.release()


def main(max_videos: int = 1) -> None:
    video_names = pd.read_csv(CSV_PATH, header=None, dtype=str).iloc[:, 0].tolist()

    processed = 0
    for video_name in video_names:
        extract_frames_for_video(video_name.strip())
        processed += 1
        if processed >= max_videos:
            break


if __name__ == "__main__":
    main(max_videos=1200)
