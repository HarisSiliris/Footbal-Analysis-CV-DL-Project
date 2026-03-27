import os
import json
import cv2
import xml.etree.ElementTree as ET
from collections import defaultdict
import argparse


def _parse_box_attributes(box):
    attrs = {}
    for attr in box.findall("attribute"):
        name = attr.get("name")
        if name:
            attrs[name] = (attr.text or "").strip()
    return attrs


def convert_to_coco(dataset_root, output_path):

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Define categories (edit if needed)
    category_map = {
        "player": 1,
        "referee": 2,
        "ball": 3
    }

    for name, cid in category_map.items():
        coco["categories"].append({
            "id": cid,
            "name": name,
            "supercategory": "object"
        })

    annotation_id = 1
    image_id = 1

    for root_dir, _, files in os.walk(dataset_root):
        if "annotations.xml" not in files:
            continue

        xml_path = os.path.join(root_dir, "annotations.xml")
        print(f"Processing {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get video file (assumes one per folder)
        video_file = None
        for f in os.listdir(root_dir):
            if f.endswith(".mp4"):
                video_file = os.path.join(root_dir, f)
                break

        if video_file is None:
            continue

        cap = cv2.VideoCapture(video_file)

        frames_dict = defaultdict(list)

        # Parse tracks
        for track in root.findall("track"):
            label = track.get("label")
            track_id = track.get("id")

            for box in track.findall("box"):
                frame = int(box.get("frame"))
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))
                outside = box.get("outside")
                occluded = box.get("occluded")
                attrs = _parse_box_attributes(box)

                frames_dict[frame].append({
                    "label": label,
                    "bbox": [xtl, ytl, xbr, ybr],
                    "track_id": track_id,
                    "outside": outside,
                    "occluded": occluded,
                    "attributes": attrs
                })

        for frame_id in sorted(frames_dict.keys()):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, img = cap.read()
            if not ret:
                continue

            height, width = img.shape[:2]

            image_filename = f"{image_id:06d}.jpg"

            image_info = {
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height
            }

            coco["images"].append(image_info)

            # Save image (optional but recommended)
            images_dir = os.path.join(output_path, "images")
            os.makedirs(images_dir, exist_ok=True)
            cv2.imwrite(os.path.join(images_dir, image_filename), img)

            for obj in frames_dict[frame_id]:

                label = obj["label"]
                attrs = obj.get("attributes", {})

                # Promote referee boxes from player tracks if team=referee
                if label == "player" and attrs.get("team") == "referee":
                    label = "referee"

                if label not in category_map:
                    continue

                xtl, ytl, xbr, ybr = obj["bbox"]

                width_box = xbr - xtl
                height_box = ybr - ytl

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[label],
                    "bbox": [xtl, ytl, width_box, height_box],
                    "area": width_box * height_box,
                    "iscrowd": 0,
                    # Preserve CVAT metadata for tracking/visibility use
                    "track_id": obj.get("track_id"),
                    "visible": obj.get("outside") == "0",
                    "occluded": obj.get("occluded") == "1",
                    "attributes": attrs
                }

                coco["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

        cap.release()

    # Save JSON
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "annotations.json"), "w") as f:
        json.dump(coco, f)

    print("COCO conversion completed!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Football2025")
    parser.add_argument("--output", required=True, help="Output COCO folder")

    args = parser.parse_args()

    convert_to_coco(args.data, args.output)
