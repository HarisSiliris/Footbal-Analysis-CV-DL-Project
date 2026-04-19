import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


def parse_names(names: Optional[str]) -> Optional[List[str]]:
    if not names:
        return None
    return [name.strip() for name in names.split(",") if name.strip()]


def find_image_files(source: str) -> List[str]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(source, pattern)))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {source}")
    return sorted(image_paths)


def make_detection(result, class_names):
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        width = x2 - x1
        height = y2 - y1
        detections.append(([x1, y1, width, height], score, class_name))
    return detections


def track_frames(
    model: YOLO,
    tracker: DeepSort,
    image_paths: List[str],
    class_names: Optional[List[str]],
    output_path: str,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
) -> None:
    frames = []
    for frame_idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(image_path)
        if image is None:
            continue

        results = model(image, imgsz=imgsz, device=device, conf=conf, iou=iou)
        if len(results) == 0:
            continue
        result = results[0]

        raw_detections = make_detection(result, class_names)
        tracks = tracker.update_tracks(raw_detections, frame=image)

        output_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = track.to_ltrb()
            width = r - l
            height = b - t
            class_name = track.get_det_class() or "unknown"
            class_id = -1
            if class_names and class_name in class_names:
                class_id = class_names.index(class_name)
            output_tracks.append({
                "track_id": int(track.track_id),
                "class_name": class_name,
                "class_id": class_id,
                "bbox": [float(l), float(t), float(width), float(height)],
                "score": float(track.get_det_conf() or 0.0),
            })

        frames.append(
            {
                "frame_id": frame_idx,
                "file_name": os.path.basename(image_path),
                "tracks": output_tracks,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "images",
                "image_count": len(frames),
                "frames": frames,
            },
            f,
            indent=2,
        )

    print(f"Saved tracking output to {output_path}")


def track_video(
    model: YOLO,
    tracker: DeepSort,
    video_path: str,
    class_names: Optional[List[str]],
    output_path: str,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")

    frames = []
    frame_idx = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(image, imgsz=imgsz, device=device, conf=conf, iou=iou)
        if len(results) == 0:
            continue
        result = results[0]

        raw_detections = make_detection(result, class_names)
        tracks = tracker.update_tracks(raw_detections, frame=image)

        output_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = track.to_ltrb()
            width = r - l
            height = b - t
            class_name = track.get_det_class() or "unknown"
            class_id = -1
            if class_names and class_name in class_names:
                class_id = class_names.index(class_name)
            output_tracks.append({
                "track_id": int(track.track_id),
                "class_name": class_name,
                "class_id": class_id,
                "bbox": [float(l), float(t), float(width), float(height)],
                "score": float(track.get_det_conf() or 0.0),
            })

        frames.append(
            {
                "frame_id": frame_idx,
                "file_name": f"{frame_idx:06d}.jpg",
                "tracks": output_tracks,
            }
        )

    cap.release()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": video_path,
                "frame_count": frame_idx,
                "frames": frames,
            },
            f,
            indent=2,
        )

    print(f"Saved video tracking output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection and DeepSort tracking.")
    parser.add_argument("--source", required=True, help="Image folder or video file to track")
    parser.add_argument("--model", required=True, help="YOLO model weights path")
    parser.add_argument("--output", required=True, help="Output JSON tracking file")
    parser.add_argument("--names", help="Comma-separated class names in order, e.g. player,referee,ball")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="", help="Device to run inference on, e.g. 0 or cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold for YOLO")
    parser.add_argument("--max-age", type=int, default=30, help="DeepSort max age for lost tracks")
    parser.add_argument("--n-init", type=int, default=3, help="DeepSort frames required to confirm a track")
    parser.add_argument("--embedder-gpu", action="store_true", help="Run DeepSort appearance embeddings on GPU")

    args = parser.parse_args()
    class_names = parse_names(args.names)

    model = YOLO(args.model)
    tracker = DeepSort(max_age=args.max_age, n_init=args.n_init, embedder="mobilenet", embedder_gpu=args.embedder_gpu)

    if os.path.isdir(args.source):
        image_paths = find_image_files(args.source)
        track_frames(
            model=model,
            tracker=tracker,
            image_paths=image_paths,
            class_names=class_names,
            output_path=args.output,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
        )
    elif os.path.isfile(args.source):
        track_video(
            model=model,
            tracker=tracker,
            video_path=args.source,
            class_names=class_names,
            output_path=args.output,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
        )
    else:
        raise ValueError(f"Source path does not exist: {args.source}")


if __name__ == "__main__":
    main()
