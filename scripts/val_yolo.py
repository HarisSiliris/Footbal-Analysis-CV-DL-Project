import argparse

from ultralytics import YOLO


def main(args):
    model = YOLO(args.model)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", required=True, help="Path to trained weights")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="e.g. 0 or cpu")
    parser.add_argument("--split", default="val", help="Dataset split: train/val/test")
    args = parser.parse_args()

    main(args)
