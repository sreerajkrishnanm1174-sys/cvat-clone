"""Find objects with the deployed detector and keep only labels that exist on the task.

The detector returns boxes. Segmentation labels are traced from those boxes by SAM.
A name the task does not use is dropped, including a detector class with no matching label.
"""

from __future__ import annotations

import logging

from .geometry import bbox, iou

log = logging.getLogger(__name__)

# Detector class -> names a CVAT label may use for the same object.
SYNONYMS = {
    "person": {"person", "human", "pedestrian", "people", "man", "woman"},
    "bicycle": {"bicycle", "bike"},
    "car": {"car", "vehicle"},
    "motorcycle": {"motorcycle", "motorbike"},
    "airplane": {"airplane", "plane", "aircraft"},
    "bus": {"bus"},
    "train": {"train"},
    "truck": {"truck", "lorry"},
    "boat": {"boat", "ship"},
    "dog": {"dog"},
    "cat": {"cat"},
    "bird": {"bird"},
    "horse": {"horse"},
    "cow": {"cow"},
    "sheep": {"sheep"},
}


def label_names(label: dict) -> set[str]:
    names = {str(label.get("name") or "").strip().lower()}
    names.update(str(alias).strip().lower() for alias in label.get("detect_as") or [])
    names.discard("")
    return names


def _group(name: str) -> set[str] | None:
    key = name.strip().lower().replace("-", " ").replace("_", " ")
    if key in SYNONYMS:
        return SYNONYMS[key] | {key}
    for words in SYNONYMS.values():
        if key in words:
            return set(words)
    return None


def match_label(detected: str, labels: list[dict]) -> dict | None:
    """Return the one task label this detection belongs to, or nothing if it is unknown or ambiguous."""
    key = (detected or "").strip().lower().replace("-", " ").replace("_", " ")
    if not key:
        return None
    exact = [label for label in labels if key in label_names(label)]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        named = [label for label in exact if str(label.get("name") or "").lower() == key]
        return named[0] if len(named) == 1 else None
    group = _group(key)
    if not group:
        return None
    hits = [label for label in labels if label_names(label) & group]
    named = [label for label in hits if str(label.get("name") or "").lower() in group]
    if len(named) == 1:
        return named[0]
    if len(hits) == 1:
        return hits[0]
    return None


def is_detector_label(label: dict) -> bool:
    return any(_group(name) for name in label_names(label))


def labels_needing_vision(labels: list[dict], detected: list[dict]) -> list[dict]:
    """Ask the vision model only for labels the detector did not already locate."""
    found = {shape.get("label") for shape in detected}
    pending = []
    for label in labels:
        if label.get("type") == "tag":
            continue
        if is_detector_label(label) and label.get("name") in found:
            continue
        pending.append(label)
    return pending


def shape_box(shape: dict) -> list[float]:
    points = list(shape.get("points") or [])
    if shape.get("encoded") == "rle" and len(points) >= 4:
        points = points[-4:]
    if len(points) < 4:
        return [0, 0, 0, 0]
    kind = "rectangle" if len(points) == 4 or shape.get("encoded") == "rle" else (shape.get("type") or "polygon")
    if kind == "mask":
        kind = "rectangle"
    x1, y1, x2, y2 = bbox(points, kind)
    return [x1, y1, x2, y2]


def dedupe_shapes(shapes: list[dict], minimum: float = 0.6) -> list[dict]:
    """Drop a second shape of the same label when it covers almost the same object."""
    ordered = sorted(shapes, key=lambda shape: float(shape.get("score") or 0), reverse=True)
    kept: list[dict] = []
    for shape in ordered:
        box = shape_box(shape)
        if any(
            shape.get("label") == other.get("label") and iou(box, "rectangle", shape_box(other), "rectangle") >= minimum
            for other in kept
        ):
            continue
        kept.append(shape)
    return kept


def as_detection_box(shape: dict) -> dict:
    """Keep a segmentation shape as a box prompt. The outline comes from the mask, not from guessed points."""
    if shape.get("type") not in {"polygon", "mask"} or shape.get("encoded") == "rle":
        return shape
    box = shape_box(shape)
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return shape
    return {**shape, "points": [round(value, 2) for value in box]}


def detector_shapes(image: bytes, labels: list[dict], threshold: float = 0.25) -> list[dict]:
    """Boxes from the deployed detector, renamed to the task label and typed as that label."""
    from .config import get_settings
    from .service import call_yolo, yolo_endpoint

    settings = get_settings()
    url = yolo_endpoint(settings.yolo_function_name)
    raw = call_yolo(url, settings.yolo_function_name, image, threshold)
    shapes = []
    for item in raw:
        label = match_label(str(item.get("label") or ""), labels)
        if label is None:
            continue
        kind = label.get("type")
        if kind not in {"rectangle", "polygon", "mask"}:
            continue
        points = [float(value) for value in (item.get("points") or [])[:4]]
        if len(points) < 4:
            continue
        shapes.append(
            {
                "label": label["name"],
                "type": kind,
                "points": points,
                "attributes": {},
                "score": item.get("score"),
                "source": "auto",
            }
        )
    kept = dedupe_shapes(shapes)
    log.info("detector kept %s object(s) for this frame", len(kept))
    return kept
