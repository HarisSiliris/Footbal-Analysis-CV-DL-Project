import os
import random
import argparse
import json
from collections import Counter, defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# COCO Loading
# -----------------------------

def load_coco_annotations(coco_path):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    annotations_per_image = defaultdict(list)

    for ann in coco["annotations"]:
        annotations_per_image[ann["image_id"]].append(ann)

    return images, categories, annotations_per_image


# -----------------------------
# Visualization
# -----------------------------

def visualize_image(image_path, boxes, categories, save_path=None):
    image = cv2.imread(image_path)
    if image is None:
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in boxes:
        x, y, w, h = box["bbox"]
        class_name = categories[box["category_id"]]

        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            class_name,
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


# -----------------------------
# Main Analysis
# -----------------------------

def main(dataset_root, num_visualizations):

    coco_path = os.path.join(dataset_root, "annotations.json")
    images, categories, annotations_per_image = load_coco_annotations(coco_path)

    print("Scanning COCO dataset...")

    class_counts = Counter()
    total_objects = 0
    image_widths = []
    image_heights = []

    image_ids = list(images.keys())

    for image_id in image_ids:
        img_info = images[image_id]
        anns = annotations_per_image.get(image_id, [])

        total_objects += len(anns)
        image_widths.append(img_info["width"])
        image_heights.append(img_info["height"])

        for ann in anns:
            class_name = categories[ann["category_id"]]
            class_counts[class_name] += 1

    num_images = len(image_ids)

    # -----------------------------
    # Print Statistics
    # -----------------------------

    print("\n===== DATASET STATISTICS =====")
    print(f"Number of images: {num_images}")
    print(f"Total objects: {total_objects}")
    print(f"Average objects per image: {total_objects / max(num_images,1):.2f}")

    print("\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

    print("\nImage resolution:")
    print(f"Average width: {np.mean(image_widths):.2f}")
    print(f"Average height: {np.mean(image_heights):.2f}")

    print("================================\n")

    # -----------------------------
    # Save Visual Samples
    # -----------------------------

    os.makedirs("results", exist_ok=True)

    sampled_ids = random.sample(image_ids, min(num_visualizations, num_images))

    for idx, image_id in enumerate(sampled_ids):
        img_info = images[image_id]
        anns = annotations_per_image.get(image_id, [])

        # Assumes images are stored in: dataset_root/images/
        image_path = os.path.join(dataset_root, "images", img_info["file_name"])

        save_path = f"results/sample_{idx}.png"

        visualize_image(image_path, anns, categories, save_path)

    print(f"Saved {len(sampled_ids)} visualization samples in results/")


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--visualize", type=int, default=5)

    args = parser.parse_args()

    main(args.data, args.visualize)