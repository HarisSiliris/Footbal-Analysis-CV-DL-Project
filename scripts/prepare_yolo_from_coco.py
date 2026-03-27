import os
import json
import argparse
from collections import defaultdict


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _build_category_map(categories):
    # Map COCO category ids to 0-based indices for YOLO
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    id_to_index = {}
    names = {}
    for i, cat in enumerate(sorted_cats):
        id_to_index[cat["id"]] = i
        names[i] = cat["name"]
    return id_to_index, names


def _write_data_yaml(out_path, dataset_root, names, splits_dir):
    lines = [
        f"path: {dataset_root}",
        f"train: {os.path.join(splits_dir, 'train.txt')}",
        f"val: {os.path.join(splits_dir, 'val.txt')}",
        f"test: {os.path.join(splits_dir, 'test.txt')}",
        "names:",
    ]
    for i in range(len(names)):
        lines.append(f"  {i}: {names[i]}")
    _write_lines(out_path, lines)


def _normalize_box(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n


def prepare_yolo(coco_path, output_root, splits_dir):
    coco = _load_json(coco_path)
    images = {img["id"]: img for img in coco["images"]}

    id_to_index, names = _build_category_map(coco["categories"])

    labels_dir = os.path.join(output_root, "labels")
    _ensure_dir(labels_dir)

    labels_per_image = defaultdict(list)
    for ann in coco["annotations"]:
        img = images.get(ann["image_id"])
        if img is None:
            continue
        img_w = img["width"]
        img_h = img["height"]
        x_c, y_c, w_n, h_n = _normalize_box(ann["bbox"], img_w, img_h)
        class_id = id_to_index[ann["category_id"]]
        labels_per_image[ann["image_id"]].append(
            f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
        )

    # Write labels
    for img_id, img in images.items():
        file_stem = os.path.splitext(img["file_name"])[0]
        label_path = os.path.join(labels_dir, f"{file_stem}.txt")
        lines = labels_per_image.get(img_id, [])
        _write_lines(label_path, lines)

    # Create split lists from COCO split JSONs
    _ensure_dir(splits_dir)
    for split in ["train", "val", "test"]:
        split_json = os.path.join(splits_dir, f"{split}.json")
        if not os.path.exists(split_json):
            continue
        split_coco = _load_json(split_json)
        image_lines = [
            os.path.abspath(os.path.join(output_root, "images", img["file_name"]))
            for img in split_coco["images"]
        ]
        _write_lines(os.path.join(splits_dir, f"{split}.txt"), image_lines)

    # Write data.yaml into output root
    data_yaml_path = os.path.join(output_root, "data.yaml")
    _write_data_yaml(data_yaml_path, output_root, names, "splits")

    print("YOLO prep completed.")
    print(f"Labels: {labels_dir}")
    print(f"Data YAML: {data_yaml_path}")
    print(f"Splits: {splits_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="Path to annotations.json")
    parser.add_argument("--out", required=True, help="Dataset root to store YOLO labels/data.yaml")
    parser.add_argument("--splits", required=True, help="Directory with train.json/val.json/test.json")
    args = parser.parse_args()

    prepare_yolo(args.coco, args.out, args.splits)
