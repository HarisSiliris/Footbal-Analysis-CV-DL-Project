import os
import json
import argparse
import random


def _load_coco(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _filter_coco(coco, image_id_set):
    images = [img for img in coco["images"] if img["id"] in image_id_set]
    annotations = [ann for ann in coco["annotations"] if ann["image_id"] in image_id_set]
    return {
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def create_splits(coco_path, output_dir, train_ratio, val_ratio, test_ratio, seed):
    coco = _load_coco(coco_path)

    image_ids = [img["id"] for img in coco["images"]]
    if not image_ids:
        raise ValueError("No images found in COCO file.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    random.seed(seed)
    random.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:n_train + n_val])
    test_ids = set(image_ids[n_train + n_val:])

    os.makedirs(output_dir, exist_ok=True)

    _write_json(os.path.join(output_dir, "train.json"), _filter_coco(coco, train_ids))
    _write_json(os.path.join(output_dir, "val.json"), _filter_coco(coco, val_ids))
    _write_json(os.path.join(output_dir, "test.json"), _filter_coco(coco, test_ids))

    with open(os.path.join(output_dir, "train_ids.txt"), "w") as f:
        f.write("\n".join(str(i) for i in sorted(train_ids)))
    with open(os.path.join(output_dir, "val_ids.txt"), "w") as f:
        f.write("\n".join(str(i) for i in sorted(val_ids)))
    with open(os.path.join(output_dir, "test_ids.txt"), "w") as f:
        f.write("\n".join(str(i) for i in sorted(test_ids)))

    print("Split created:")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val:   {len(val_ids)} images")
    print(f"  Test:  {len(test_ids)} images")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="Path to annotations.json")
    parser.add_argument("--out", required=True, help="Output folder for split JSONs")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    create_splits(
        coco_path=args.coco,
        output_dir=args.out,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
