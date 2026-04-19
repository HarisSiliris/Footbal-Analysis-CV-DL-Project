import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2


Color = Tuple[int, int, int]


CLASS_COLORS: Dict[str, Color] = {
    "player": (80, 220, 100),
    "referee": (80, 180, 255),
    "ball": (0, 210, 255),
    "unknown": (255, 255, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render tracking overlays from track_results.json to an MP4 video."
    )
    parser.add_argument("--track-json", required=True, help="Path to tracking JSON output")
    parser.add_argument(
        "--image-dir",
        help="Directory containing the tracked frames. Required when tracking source is images.",
    )
    parser.add_argument(
        "--video",
        help="Original video file. Required when tracking source is a video.",
    )
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Output frames per second. Used for image sequences and output encoding.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="1-based frame index to start rendering from",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to render",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Render this many seconds from the start frame. Overrides max-frames if provided.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Bounding box line thickness",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.6,
        help="Overlay text font scale",
    )
    return parser.parse_args()


def load_track_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def color_for_track(track_id: int, class_name: str) -> Color:
    base_color = CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])
    rng = random.Random(track_id)
    # Keep class colors recognizable while varying them per track.
    jitter = [rng.randint(-35, 35) for _ in range(3)]
    return tuple(max(0, min(255, base_color[i] + jitter[i])) for i in range(3))


def annotate_frame(
    image,
    tracks: Iterable[dict],
    line_thickness: int,
    font_scale: float,
):
    for track in tracks:
        x, y, w, h = track["bbox"]
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = max(x1 + 1, int(round(x + w)))
        y2 = max(y1 + 1, int(round(y + h)))

        class_name = track.get("class_name", "unknown")
        track_id = track.get("track_id", -1)
        score = track.get("score", 0.0)
        color = color_for_track(track_id, class_name)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        label = f"{class_name} #{track_id} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            max(1, line_thickness - 1),
        )
        text_top = max(0, y1 - text_h - baseline - 6)
        text_bottom = text_top + text_h + baseline + 6
        text_right = min(image.shape[1], x1 + text_w + 8)

        cv2.rectangle(image, (x1, text_top), (text_right, text_bottom), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 4, text_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            max(1, line_thickness - 1),
            cv2.LINE_AA,
        )

    return image


def frame_window(args: argparse.Namespace, total_frames: int) -> Tuple[int, int]:
    start_idx = max(0, args.start_frame - 1)
    end_idx = total_frames

    if args.duration is not None:
        max_frames = max(1, int(round(args.duration * args.fps)))
        end_idx = min(total_frames, start_idx + max_frames)
    elif args.max_frames is not None:
        end_idx = min(total_frames, start_idx + max(1, args.max_frames))

    if start_idx >= total_frames:
        raise ValueError(
            f"start-frame {args.start_frame} is beyond the available {total_frames} frames"
        )

    return start_idx, end_idx


def read_image_frame(image_dir: Path, frame_info: dict):
    image_path = image_dir / frame_info["file_name"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def render_from_images(track_data: dict, image_dir: Path, args: argparse.Namespace) -> None:
    frames = track_data.get("frames", [])
    start_idx, end_idx = frame_window(args, len(frames))

    first_image = read_image_frame(image_dir, frames[start_idx])
    height, width = first_image.shape[:2]
    writer = make_writer(Path(args.output), args.fps, width, height)

    try:
        for frame_info in frames[start_idx:end_idx]:
            image = read_image_frame(image_dir, frame_info)
            annotated = annotate_frame(
                image,
                frame_info.get("tracks", []),
                args.line_thickness,
                args.font_scale,
            )
            writer.write(annotated)
    finally:
        writer.release()


def render_from_video(video_path: Path, track_data: dict, args: argparse.Namespace) -> None:
    frames = track_data.get("frames", [])
    start_idx, end_idx = frame_window(args, len(frames))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        ok, image = cap.read()
        if not ok or image is None:
            raise RuntimeError(f"Could not read frame {start_idx + 1} from {video_path}")

        height, width = image.shape[:2]
        writer = make_writer(Path(args.output), args.fps, width, height)

        try:
            current_idx = start_idx
            while current_idx < end_idx:
                if current_idx > start_idx:
                    ok, image = cap.read()
                    if not ok or image is None:
                        break

                annotated = annotate_frame(
                    image.copy(),
                    frames[current_idx].get("tracks", []),
                    args.line_thickness,
                    args.font_scale,
                )
                writer.write(annotated)
                current_idx += 1
        finally:
            writer.release()
    finally:
        cap.release()


def make_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")
    return writer


def main() -> None:
    args = parse_args()
    track_json_path = Path(args.track_json)
    track_data = load_track_data(track_json_path)
    source_type = track_data.get("source", "")

    if source_type == "images":
        if not args.image_dir:
            raise ValueError("--image-dir is required when the tracking JSON source is images")
        render_from_images(track_data, Path(args.image_dir), args)
    else:
        if not args.video:
            raise ValueError("--video is required when the tracking JSON source is a video")
        render_from_video(Path(args.video), track_data, args)

    print(f"Saved overlay video to {args.output}")


if __name__ == "__main__":
    main()
