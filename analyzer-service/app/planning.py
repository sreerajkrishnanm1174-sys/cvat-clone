"""Dataset profile and per-label agent plan. Only agents that exist are offered."""

from __future__ import annotations

COCO_RECTANGLE = {
    "person",
    "bicycle",
    "car",
    "motorbike",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}

# Deployed on this CVAT. SAM is an interactor: it needs clicks, it does not detect.
AVAILABLE = {
    "yolo": {
        "id": "onnx-wongkinyiu-yolov7",
        "types": {"rectangle"},
        "labels": COCO_RECTANGLE,
        "summary": "YOLO v7, already deployed, draws boxes for its known classes such as car.",
    },
    "sam": {
        "id": "pth-facebookresearch-sam-vit-h",
        "types": set(),
        "labels": set(),
        "summary": "Segment Anything is available as a click tool. It does not take a first pass by itself.",
    },
}


def profile_task(task: dict, meta: dict, jobs: list[dict], annotation_count: int | None) -> dict:
    frames = (meta or {}).get("frames") or []
    widths = [frame.get("width") for frame in frames if frame.get("width")]
    heights = [frame.get("height") for frame in frames if frame.get("height")]
    return {
        "task_id": task.get("id"),
        "name": task.get("name"),
        "size": task.get("size") or len(frames),
        "mode": task.get("mode"),
        "jobs": len(jobs),
        "frame_count": task.get("size") or len(frames),
        "widths": {"min": min(widths) if widths else None, "max": max(widths) if widths else None},
        "heights": {"min": min(heights) if heights else None, "max": max(heights) if heights else None},
        "video": task.get("mode") == "interpolation" or bool((meta or {}).get("frame_filter")),
        "existing_annotations": annotation_count,
    }


def plan_agents(labels: list[dict], allow_external: bool) -> list[dict]:
    """The agent draws on the task image. CVAT's deployed detectors are not called."""
    direct = {
        "rectangle",
        "polygon",
        "polyline",
        "points",
        "ellipse",
        "cuboid",
        "mask",
        "skeleton",
        "tag",
        "interval",
        "any",
    }
    plan = []
    for label in labels:
        name = label.get("name") or ""
        shape = label.get("type") or "any"
        if shape in direct:
            chosen = "claude_vision"
            reason = (
                "The agent reads the requirement and adds this annotation type to the project. "
                "CVAT's own models are not used."
            )
        else:
            chosen = "human"
            reason = "Draw this shape in CVAT. The agent does not call a CVAT model for it."
        plan.append({"label": name, "type": shape, "agent": chosen, "reason": reason})
    return plan
