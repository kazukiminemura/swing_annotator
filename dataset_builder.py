#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


@dataclass(frozen=True)
class Anchor:
    start_frame: int
    label: str


def load_anchors(path: Path) -> list[Anchor]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("anchors JSON must be a list")

    anchors: list[Anchor] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"anchors[{i}] must be an object")
        if "start_frame" not in item or "label" not in item:
            raise ValueError(f"anchors[{i}] must include start_frame and label")
        start_frame = int(item["start_frame"])
        label = str(item["label"]).strip()
        if start_frame < 0:
            raise ValueError(f"anchors[{i}].start_frame must be >= 0")
        if not label:
            raise ValueError(f"anchors[{i}].label must not be empty")
        anchors.append(Anchor(start_frame=start_frame, label=label))

    return validate_anchors(anchors)


def validate_anchors(anchors: list[Anchor]) -> list[Anchor]:
    if not anchors:
        raise ValueError("anchors must not be empty")
    sorted_anchors = sorted(anchors, key=lambda a: a.start_frame)
    for anchor in sorted_anchors:
        if anchor.start_frame < 0:
            raise ValueError("anchor.start_frame must be >= 0")
        if not anchor.label.strip():
            raise ValueError("anchor.label must not be empty")
    for prev, cur in zip(sorted_anchors, sorted_anchors[1:]):
        if prev.start_frame == cur.start_frame:
            raise ValueError(f"duplicate start_frame: {cur.start_frame}")
    return sorted_anchors


def save_anchors(path: Path, anchors: list[Anchor]) -> None:
    clean_anchors = validate_anchors(anchors)
    serializable = [
        {"start_frame": anchor.start_frame, "label": anchor.label}
        for anchor in clean_anchors
    ]
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sanitize_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)


def resolve_label(frame_index: int, anchors: list[Anchor]) -> str | None:
    for i, anchor in enumerate(anchors):
        next_start = anchors[i + 1].start_frame if i + 1 < len(anchors) else None
        if next_start is None and frame_index >= anchor.start_frame:
            return anchor.label
        if next_start is not None and anchor.start_frame <= frame_index < next_start:
            return anchor.label
    return None


def iter_frames(cap: cv2.VideoCapture) -> Iterable[tuple[int, object]]:
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame_idx, frame
        frame_idx += 1


def parse_split_ratios(split: str) -> tuple[int, int, int]:
    chunks = [s.strip() for s in split.split(",")]
    if len(chunks) != 3:
        raise ValueError("--split must have 3 comma separated numbers (e.g. 80,10,10)")
    try:
        a, b, c = [int(x) for x in chunks]
    except ValueError as exc:
        raise ValueError("--split must contain integers") from exc
    if a < 0 or b < 0 or c < 0:
        raise ValueError("--split must not include negative values")
    if (a + b + c) <= 0:
        raise ValueError("--split total must be > 0")
    return (a, b, c)


def choose_split(key: str, split_ratios: tuple[int, int, int]) -> str:
    train, valid, test = split_ratios
    total = train + valid + test
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % total
    if bucket < train:
        return "train"
    if bucket < train + valid:
        return "valid"
    return "test"


def build_dataset(
    video_path: Path,
    anchors_path: Path,
    output_dir: Path,
    step: int,
    jpeg_quality: int,
    output_format: str = "native",
    split_ratios: tuple[int, int, int] = (80, 10, 10),
) -> Path:
    anchors = load_anchors(anchors_path)
    return build_dataset_from_anchors(
        video_path=video_path,
        anchors=anchors,
        output_dir=output_dir,
        step=step,
        jpeg_quality=jpeg_quality,
        output_format=output_format,
        split_ratios=split_ratios,
    )


def build_dataset_from_anchors(
    video_path: Path,
    anchors: list[Anchor],
    output_dir: Path,
    step: int,
    jpeg_quality: int,
    output_format: str = "native",
    split_ratios: tuple[int, int, int] = (80, 10, 10),
) -> Path:
    if output_format not in {"native", "roboflow"}:
        raise ValueError("output_format must be one of: native, roboflow")
    anchors = validate_anchors(anchors)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    video_stem = video_path.stem
    labels_used: set[str] = set()

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "label", "frame_index", "video", "split"],
        )
        writer.writeheader()
        for frame_index, frame in iter_frames(cap):
            if frame_index % step != 0:
                continue

            label = resolve_label(frame_index, anchors)
            if label is None:
                continue

            labels_used.add(label)
            safe_label = sanitize_label(label)
            file_name = f"{video_stem}_f{frame_index:06d}.jpg"
            split_name = ""

            if output_format == "native":
                class_dir = output_dir / "images" / safe_label
                image_path = class_dir / file_name
            else:
                split_name = choose_split(f"{video_path.name}:{frame_index}", split_ratios)
                class_dir = output_dir / split_name / safe_label
                image_path = class_dir / file_name

            class_dir.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(
                str(image_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not ok:
                raise RuntimeError(f"failed to write image: {image_path}")

            writer.writerow(
                {
                    "image_path": str(image_path.relative_to(output_dir)),
                    "label": label,
                    "frame_index": frame_index,
                    "video": video_path.name,
                    "split": split_name,
                }
            )

    if output_format == "roboflow":
        labels_txt = output_dir / "labels.txt"
        labels_txt.write_text(
            "\n".join(sorted(labels_used)) + "\n",
            encoding="utf-8",
        )

    cap.release()
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an image-classification dataset from a video and anchor labels."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--anchors",
        type=Path,
        required=True,
        help="JSON list: [{\"start_frame\": 0, \"label\": \"...\"}, ...]",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset directory",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality [1-100] (default: 95)",
    )
    parser.add_argument(
        "--format",
        choices=["native", "roboflow"],
        default="native",
        help="Output format. roboflow creates train/valid/test/<label>/",
    )
    parser.add_argument(
        "--split",
        default="80,10,10",
        help="Split ratio for roboflow format: train,valid,test (default: 80,10,10)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be > 0")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be between 1 and 100")
    split_ratios = parse_split_ratios(args.split)
    manifest = build_dataset(
        video_path=args.video,
        anchors_path=args.anchors,
        output_dir=args.output,
        step=args.step,
        jpeg_quality=args.jpeg_quality,
        output_format=args.format,
        split_ratios=split_ratios,
    )
    print(f"done: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
