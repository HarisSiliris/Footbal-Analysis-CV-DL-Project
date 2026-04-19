import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import motmetrics as mm


def load_coco_annotations(coco_path: str) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_id_to_name = {img["id"]: img for img in coco.get("images", [])}
    frames = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_info = image_id_to_name.get(ann["image_id"])
        if image_info is None:
            continue
        file_name = image_info["file_name"]
        frames[file_name].append(
            {
                "track_id": str(ann.get("track_id", ann.get("id", ""))),
                "category_id": int(ann["category_id"]),
                "bbox": ann["bbox"],
            }
        )

    return image_id_to_name, frames


def load_prediction_tracks(pred_path: str) -> Dict[str, List[dict]]:
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = defaultdict(list)
    for frame in data.get("frames", []):
        file_name = frame.get("file_name")
        for track in frame.get("tracks", []):
            frames[file_name].append(
                {
                    "track_id": str(track["track_id"]),
                    "category_id": track.get("class_id", -1),
                    "class_name": track.get("class_name", ""),
                    "bbox": track["bbox"],
                }
            )

    return frames


def iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def assign_matches(
    gt_objects: List[dict],
    pred_objects: List[dict],
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    if not gt_objects or not pred_objects:
        return []

    cost_matrix = []
    for gt in gt_objects:
        row = []
        for pred in pred_objects:
            if gt["category_id"] != pred["category_id"]:
                row.append(1.0)
                continue
            score = iou(gt["bbox"], pred["bbox"])
            row.append(1.0 - score)
        cost_matrix.append(row)

    row_ind, col_ind = mm.lap.linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < len(gt_objects) and c < len(pred_objects):
            match_iou = 1.0 - cost_matrix[r][c]
            if match_iou >= iou_threshold:
                matches.append((r, c, match_iou))
    return matches


def compute_hota(
    pair_matches: Dict[Tuple[str, str, int], int],
    gt_presence: Dict[Tuple[str, int], int],
    pred_presence: Dict[Tuple[str, int], int],
    total_tp: int,
    total_fp: int,
    total_fn: int,
) -> Tuple[float, float, float]:
    if total_tp == 0:
        return 0.0, 0.0, 0.0

    det_a = total_tp / (total_tp + total_fp + total_fn)
    assoc_scores = []
    for (gt_id, pred_id, class_id), matched_frames in pair_matches.items():
        gt_count = gt_presence.get((gt_id, class_id), 0)
        pred_count = pred_presence.get((pred_id, class_id), 0)
        denom = matched_frames + (gt_count - matched_frames) + (pred_count - matched_frames)
        if denom > 0:
            assoc_scores.append(matched_frames / denom)

    assoc_a = sum(assoc_scores) / len(assoc_scores) if assoc_scores else 0.0
    hota = math.sqrt(det_a * assoc_a)
    return det_a, assoc_a, hota


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking performance from YOLO+DeepSort output.")
    parser.add_argument("--gt", required=True, help="COCO-style annotations JSON for ground truth")
    parser.add_argument("--pred", required=True, help="Predicted tracking JSON from scripts/track_yolo.py")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    args = parser.parse_args()

    _, gt_frames = load_coco_annotations(args.gt)
    pred_frames = load_prediction_tracks(args.pred)

    frame_names = sorted(set(gt_frames) | set(pred_frames))
    acc = mm.MOTAccumulator(auto_id=True)

    total_tp = total_fp = total_fn = 0
    pair_matches = defaultdict(int)
    gt_presence = defaultdict(int)
    pred_presence = defaultdict(int)

    for frame_name in frame_names:
        gt_objects = gt_frames.get(frame_name, [])
        pred_objects = pred_frames.get(frame_name, [])

        for gt in gt_objects:
            gt_presence[(gt["track_id"], gt["category_id"])] += 1
        for pred in pred_objects:
            pred_presence[(pred["track_id"], pred["category_id"])] += 1

        matches = assign_matches(gt_objects, pred_objects, args.iou_threshold)
        matched_gt_ids = set()
        matched_pred_ids = set()

        gt_ids = [gt["track_id"] for gt in gt_objects]
        pred_ids = [pred["track_id"] for pred in pred_objects]
        distances = []
        for gt in gt_objects:
            for pred in pred_objects:
                if gt["category_id"] != pred["category_id"]:
                    distances.append(1.0)
                else:
                    distances.append(1.0 - iou(gt["bbox"], pred["bbox"]))

        if gt_ids or pred_ids:
            cost = [[1.0 - iou(gt["bbox"], pred["bbox"]) if gt["category_id"] == pred["category_id"] else 1.0 for pred in pred_objects] for gt in gt_objects]
            acc.update(gt_ids, pred_ids, cost)
        else:
            acc.update([], [], [])

        total_tp += len(matches)
        total_fp += max(0, len(pred_objects) - len(matches))
        total_fn += max(0, len(gt_objects) - len(matches))

        for gt_idx, pred_idx, _ in matches:
            gt = gt_objects[gt_idx]
            pred = pred_objects[pred_idx]
            pair_matches[(gt["track_id"], pred["track_id"], gt["category_id"])] += 1
            matched_gt_ids.add(gt["track_id"])
            matched_pred_ids.add(pred["track_id"])

    summary = mm.metrics.create()
    summary_df = summary.compute(acc, metrics=["motp", "mota", "idf1", "num_switches", "num_false_positives", "num_misses", "num_matches"], name="overall")
    det_a, assoc_a, hota = compute_hota(pair_matches, gt_presence, pred_presence, total_tp, total_fp, total_fn)

    print("Tracking evaluation results")
    print("---------------------------")
    print(summary_df.to_string())
    print()
    print(f"HOTA (approx): {hota:.4f}")
    print(f"Detection Accuracy (DetA): {det_a:.4f}")
    print(f"Association Accuracy (AssA): {assoc_a:.4f}")


if __name__ == "__main__":
    main()
