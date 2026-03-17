import os
import random
import argparse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Utility Functions
# -----------------------------

def find_annotation_files(dataset_root):
    annotation_files = []
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file == "annotations.xml":
                annotation_files.append(os.path.join(root, file))
    return annotation_files


def parse_cvat_xml(xml_path):
    """
    Parses CVAT annotations.xml
    Returns:
        images_data: list of dicts with image info and boxes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images_data = []

    for image in root.findall("image"):
        image_name = image.get("name")
        width = int(image.get("width"))
        height = int(image.get("height"))

        boxes = []

        for box in image.findall("box"):
            label = box.get("label")
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            boxes.append({
                "label": label,
                "bbox": [xtl, ytl, xbr, ybr]
            })

        images_data.append({
            "name": image_name,
            "width": width,
            "height": height,
            "boxes": boxes
        })

    return images_data


def visualize_image(image_path, boxes, save_path=None):
    image = cv2.imread(image_path)
    if image is None:
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in boxes:
        xtl, ytl, xbr, ybr = box["bbox"]
        label = box["label"]

        cv2.rectangle(
            image,
            (int(xtl), int(ytl)),
            (int(xbr), int(ybr)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (int(xtl), int(ytl) - 5),
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

    print("Scanning dataset...")
    annotation_files = find_annotation_files(dataset_root)

    print(f"Found {len(annotation_files)} annotation files.")

    class_counts = Counter()
    total_objects = 0
    image_widths = []
    image_heights = []
    bbox_areas = defaultdict(list)

    all_images = []

    for xml_file in annotation_files:
        print(f"Processing: {xml_file}")
        images_data = parse_cvat_xml(xml_file)

        for img_data in images_data:
            total_objects += len(img_data["boxes"])
            image_widths.append(img_data["width"])
            image_heights.append(img_data["height"])
            all_images.append((xml_file, img_data))

            for box in img_data["boxes"]:
                label = box["label"]
                class_counts[label] += 1

                xtl, ytl, xbr, ybr = box["bbox"]
                area = (xbr - xtl) * (ybr - ytl)
                bbox_areas[label].append(area)

    num_images = len(all_images)

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

    print("\nAverage bounding box area per class:")
    for cls, areas in bbox_areas.items():
        print(f"{cls}: {np.mean(areas):.2f}")

    print("================================\n")

    # -----------------------------
    # Save Visual Samples
    # -----------------------------

    os.makedirs("results", exist_ok=True)

    sampled = random.sample(all_images, min(num_visualizations, num_images))

    for idx, (xml_file, img_data) in enumerate(sampled):
        # image path assumption (relative to dataset root)
        image_path = os.path.join(os.path.dirname(xml_file), "data", img_data["name"])

        save_path = f"results/sample_{idx}.png"
        visualize_image(image_path, img_data["boxes"], save_path)

    print(f"Saved {len(sampled)} visualization samples in results/")


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to Football2025 dataset root"
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=5,
        help="Number of random visualizations"
    )

    args = parser.parse_args()

    main(args.data, args.visualize)