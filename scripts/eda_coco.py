import os
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


def _load_coco(path):
    with open(path, "r") as f:
        return json.load(f)


def _save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_eda(coco_path, out_dir):
    coco = _load_coco(coco_path)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    class_counts = Counter()
    per_image_counts = Counter()
    area_by_class = defaultdict(list)
    aspect_by_class = defaultdict(list)
    visible_by_class = Counter()
    occluded_by_class = Counter()

    for ann in coco["annotations"]:
        cid = ann["category_id"]
        class_name = categories.get(cid, str(cid))
        class_counts[class_name] += 1
        per_image_counts[ann["image_id"]] += 1

        _, _, w, h = ann["bbox"]
        if w > 0 and h > 0:
            area_by_class[class_name].append(w * h)
            aspect_by_class[class_name].append(w / h)

        if "visible" in ann:
            if ann["visible"]:
                visible_by_class[class_name] += 1
        if "occluded" in ann:
            if ann["occluded"]:
                occluded_by_class[class_name] += 1

    # Basic stats
    widths = [img["width"] for img in images.values()]
    heights = [img["height"] for img in images.values()]

    print("\n===== EDA SUMMARY =====")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(coco['annotations'])}")
    print("Class counts:")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt}")
    print(f"Avg image size: {np.mean(widths):.2f} x {np.mean(heights):.2f}")

    # Plot: class distribution
    plt.figure()
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Class Distribution")
    plt.ylabel("Count")
    _save_plot(os.path.join(out_dir, "class_distribution.png"))

    # Plot: objects per image histogram
    plt.figure()
    plt.hist(list(per_image_counts.values()), bins=30)
    plt.title("Objects per Image")
    plt.xlabel("Objects")
    plt.ylabel("Images")
    _save_plot(os.path.join(out_dir, "objects_per_image.png"))

    # Plot: bbox area by class (log scale)
    plt.figure()
    for cls, areas in area_by_class.items():
        if areas:
            plt.hist(np.log10(np.array(areas) + 1.0), bins=30, alpha=0.5, label=cls)
    plt.title("BBox Area (log10) by Class")
    plt.xlabel("log10(area + 1)")
    plt.ylabel("Count")
    plt.legend()
    _save_plot(os.path.join(out_dir, "bbox_area_log.png"))

    # Plot: aspect ratio by class
    plt.figure()
    for cls, ratios in aspect_by_class.items():
        if ratios:
            plt.hist(ratios, bins=30, alpha=0.5, label=cls)
    plt.title("BBox Aspect Ratio by Class")
    plt.xlabel("w / h")
    plt.ylabel("Count")
    plt.legend()
    _save_plot(os.path.join(out_dir, "bbox_aspect_ratio.png"))

    # Visibility / occlusion summary if available
    if visible_by_class or occluded_by_class:
        print("\nVisibility/occlusion (if available):")
        for cls in class_counts.keys():
            vis = visible_by_class.get(cls, 0)
            occ = occluded_by_class.get(cls, 0)
            print(f"  {cls}: visible={vis}, occluded={occ}")

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="Path to annotations.json")
    parser.add_argument("--out", default="results/eda", help="Output folder for plots")
    args = parser.parse_args()

    run_eda(args.coco, args.out)
