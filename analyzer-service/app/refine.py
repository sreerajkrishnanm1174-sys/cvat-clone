"""Second pass for vision output.

The first pass asks the model for coordinates on a downscaled full image. That is good at
"what and roughly where" and weak at exact pixels. This pass fixes both shapes:

* polygon: grow a real silhouette from the image pixels (GrabCut), seeded by the model's outline.
* skeleton: zoom to the subject, locate every joint again at full resolution, then draw the
  joints on the crop and let the model check each one.

Everything degrades to the first-pass shape. Nothing here raises to the caller.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from typing import Callable

from .geometry import bbox, flat, outline_is_template, pairs, polygon_area

log = logging.getLogger(__name__)

# ask(system_prompt, content_blocks) -> model text
Ask = Callable[[str, list], str]

KEYPOINT_SYSTEM = """You locate keypoints on ONE zoomed crop. Return ONLY JSON:
{"keypoints":[{"name":"<exact name>","x":0,"y":0,"v":2}]}
Pixels of the crop as shown, origin top-left. Put each point at the CENTER of the joint or part,
not on the body edge. left/right means the subject's own left/right, not the viewer's.
v=2 visible, v=1 hidden but its position can be estimated, v=0 not in the crop (x=0,y=0).
Return one entry for every requested name, in the given order."""


def _cv():
    try:
        import cv2
        import numpy as np

        return cv2, np
    except ImportError:
        return None, None


def refine_shapes(image: bytes, shapes: list[dict], labels: list[dict], ask: Ask | None, max_skeletons: int = 8) -> list[dict]:
    cv2, np = _cv()
    if cv2 is None:
        log.warning("opencv is not installed, so outlines are not fitted to the image")
        return shapes
    img = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return shapes
    by_name = {str(item.get("name")): item for item in labels}
    out: list[dict] = []
    skeletons = 0
    for shape in shapes:
        try:
            if shape.get("type") == "polygon":
                fitted = fit_outline(img, shape.get("points") or [])
                if fitted:
                    shape = {**shape, "points": fitted}
            elif shape.get("type") == "skeleton" and ask is not None and skeletons < max_skeletons:
                skeletons += 1
                shape = refine_skeleton(img, shape, by_name.get(str(shape.get("label"))) or {}, ask)
        except Exception as exc:  # a bad refinement must never lose the first-pass shape
            log.warning("refinement skipped for %s: %s", shape.get("label"), exc)
        out.append(shape)
    return out


# ---------------------------------------------------------------- polygons


def fit_outline(img, prior: list[float]) -> list[float] | None:
    """Return a pixel-accurate outline, or None to keep the model's outline."""
    cv2, np = _cv()
    height, width = img.shape[:2]
    pts = pairs(prior)
    if len(pts) < 3:
        return None
    template = outline_is_template(prior)
    x1, y1, x2, y2 = bbox(prior, "polygon")
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None
    pad = max(10.0, 0.15 * max(bw, bh))
    cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    cx2, cy2 = int(min(width, x2 + pad)), int(min(height, y2 + pad))
    crop = img[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if ch < 8 or cw < 8:
        return None
    s = min(1.0, 640.0 / max(ch, cw))
    small = cv2.resize(crop, (max(8, round(cw * s)), max(8, round(ch * s))), interpolation=cv2.INTER_AREA) if s < 1 else crop.copy()
    sh, sw = small.shape[:2]

    local = np.array([[(x - cx1) * s, (y - cy1) * s] for x, y in pts], dtype=np.float32)
    local_i = np.round(local).astype(np.int32)
    inside = np.zeros((sh, sw), np.uint8)
    cv2.fillPoly(inside, [local_i], 1)

    mask = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    reach = max(3, int(0.08 * min(bw, bh) * s))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    mask[cv2.dilate(inside, kernel) == 1] = cv2.GC_PR_BGD
    if template:
        # The model only gave a box. Trust the middle of it, not the corners.
        bx1, by1 = int((x1 - cx1) * s), int((y1 - cy1) * s)
        bx2, by2 = int((x2 - cx1) * s), int((y2 - cy1) * s)
        mask[max(0, by1) : by2, max(0, bx1) : bx2] = cv2.GC_PR_FGD
        core = np.zeros((sh, sw), np.uint8)
        cv2.ellipse(core, ((bx1 + bx2) // 2, (by1 + by2) // 2), (max(2, (bx2 - bx1) // 5), max(2, (by2 - by1) // 5)), 0, 0, 360, 1, -1)
        mask[core == 1] = cv2.GC_FGD
    else:
        mask[inside == 1] = cv2.GC_PR_FGD
        mask[cv2.erode(inside, kernel) == 1] = cv2.GC_FGD
    if not (mask == cv2.GC_FGD).any():
        return None

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, comp, stats, _ = cv2.connectedComponentsWithStats(fg)
    if count < 2:
        return None
    # keep the component that overlaps the model's outline most
    best, best_overlap = 0, 0
    for index in range(1, count):
        overlap = int(((comp == index) & (inside == 1)).sum())
        if overlap > best_overlap:
            best, best_overlap = index, overlap
    if not best:
        return None
    fg = (comp == best).astype(np.uint8)

    area_fg, area_prior = int(fg.sum()), int(inside.sum())
    if template:
        box_area = max(1, (x2 - x1) * (y2 - y1) * s * s)
        if not (0.12 <= area_fg / box_area <= 0.97):
            return None
    else:
        union = int(((fg == 1) | (inside == 1)).sum())
        iou = int(((fg == 1) & (inside == 1)).sum()) / max(1, union)
        if iou < 0.55 or not (0.4 <= area_fg / max(1, area_prior) <= 1.8):
            return None  # GrabCut leaked or missed. The model's outline is the safer shape.

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(1.0, 0.004 * perimeter), True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    result = [((float(px) / s) + cx1, (float(py) / s) + cy1) for px, py in approx]
    result = [(min(max(px, 0.0), width), min(max(py, 0.0), height)) for px, py in result]
    out = [round(v, 2) for v in flat(result)]
    return out if polygon_area(out) > 0 else None


# ---------------------------------------------------------------- skeletons


def refine_skeleton(img, shape: dict, label: dict, ask: Ask) -> dict:
    cv2, np = _cv()
    height, width = img.shape[:2]
    elements = [dict(item) for item in shape.get("elements") or []]
    names = [str(item.get("label")) for item in elements]
    visible = [item for item in elements if not item.get("outside")]
    if len(visible) < 2:
        return shape

    xs = [float(item["points"][0]) for item in visible]
    ys = [float(item["points"][1]) for item in visible]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    hint = shape.get("hint_bbox")
    if hint:
        x1, y1, x2, y2 = min(x1, hint[0]), min(y1, hint[1]), max(x2, hint[2]), max(y2, hint[3])
    pad = max(20.0, 0.25 * max(x2 - x1, y2 - y1))
    cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    cx2, cy2 = int(min(width, x2 + pad)), int(min(height, y2 + pad))
    crop = img[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if ch < 32 or cw < 32:
        return shape
    factor = min(3.0, max(0.1, 1024.0 / max(ch, cw)))
    factor = min(factor, 1568.0 / max(ch, cw))
    sent = cv2.resize(crop, (round(cw * factor), round(ch * factor)), interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)

    edges = [list(edge) for edge in (label.get("edges") or [])]
    guess = {
        item["label"]: ((item["points"][0] - cx1) * factor, (item["points"][1] - cy1) * factor, 1 if item.get("occluded") else 2)
        for item in visible
    }

    found = _ask_keypoints(ask, sent, label.get("name") or shape.get("label"), names, edges, None)
    if found:
        guess = {name: found.get(name, guess.get(name)) for name in names if found.get(name) or guess.get(name)}
        guess = {name: value for name, value in guess.items() if value and value[2] > 0}
    checked = _ask_keypoints(ask, _overlay(sent, guess, names, edges), label.get("name") or shape.get("label"), names, edges, guess)
    if checked:
        guess = {name: value for name, value in checked.items() if value[2] > 0}
    if not found and not checked:
        return shape

    sh, sw = sent.shape[:2]
    refined = []
    for item in elements:
        name = item["label"]
        value = guess.get(name)
        if not value:
            refined.append({"label": name, "points": [0, 0], "outside": True})
            continue
        px, py, vis = value
        px, py = px / factor + cx1, py / factor + cy1
        refined.append(
            {
                "label": name,
                "points": [round(min(max(px, 0.0), width), 2), round(min(max(py, 0.0), height), 2)],
                "outside": False,
                "occluded": vis == 1,
            }
        )
    if not any(not item.get("outside") for item in refined):
        return shape
    return {**shape, "elements": refined}


def _ask_keypoints(ask: Ask, image, label: str, names: list[str], edges: list, current: dict | None) -> dict | None:
    cv2, _ = _cv()
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None
    height, width = image.shape[:2]
    if current is None:
        text = f"Image size {width}x{height}. This crop shows one {label}. Keypoints, in order: {json.dumps(names)}."
    else:
        numbered = ", ".join(f"{i + 1}={name}" for i, name in enumerate(names))
        rows = {name: [round(v[0], 1), round(v[1], 1), v[2]] for name, v in current.items()}
        text = (
            f"Image size {width}x{height}. Numbered markers were placed by an earlier pass: {numbered}. "
            f"Their coordinates: {json.dumps(rows)}. Check each marker against the name it carries. "
            "Move only a marker that is not centered on its joint, and fix any left/right swap. "
            "Return the full corrected keypoint list."
        )
    if edges:
        text += f" Connections: {json.dumps(edges)}."
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(buffer.tobytes()).decode("ascii")}},
        {"type": "text", "text": text},
    ]
    try:
        raw = ask(KEYPOINT_SYSTEM, content)
    except Exception as exc:
        log.warning("keypoint pass failed: %s", exc)
        return None
    return _parse_keypoints(raw, names, width, height)


def _parse_keypoints(text: str, names: list[str], width: int, height: int) -> dict | None:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    rows = data.get("keypoints") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    lookup = {name.lower(): name for name in names}
    out: dict = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = lookup.get(str(row.get("name") or "").lower())
        if not name:
            continue
        try:
            x, y, v = float(row.get("x", 0)), float(row.get("y", 0)), int(row.get("v", 2))
        except (TypeError, ValueError):
            continue
        if max(abs(x), abs(y)) <= 1.5 and (x or y):
            x, y = x * width, y * height
        if v <= 0 or not (-0.05 * width <= x <= 1.05 * width and -0.05 * height <= y <= 1.05 * height):
            out[name] = (0.0, 0.0, 0)
            continue
        out[name] = (min(max(x, 0.0), width), min(max(y, 0.0), height), 1 if v == 1 else 2)
    return out or None


def _overlay(image, guess: dict, names: list[str], edges: list):
    cv2, _ = _cv()
    canvas = image.copy()
    scale = max(1.0, max(canvas.shape[:2]) / 700.0)
    index = {name: i for i, name in enumerate(names)}
    for a, b in edges:
        if a in guess and b in guess:
            cv2.line(canvas, tuple(int(v) for v in guess[a][:2]), tuple(int(v) for v in guess[b][:2]), (255, 255, 255), max(1, int(scale)))
    for name, (x, y, vis) in guess.items():
        colour = (0, 200, 255) if vis == 2 else (255, 120, 0)
        centre = (int(x), int(y))
        cv2.circle(canvas, centre, max(4, int(5 * scale)), (0, 0, 0), -1)
        cv2.circle(canvas, centre, max(3, int(4 * scale)), colour, -1)
        cv2.putText(canvas, str(index[name] + 1), (centre[0] + int(6 * scale), centre[1] - int(6 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), max(2, int(3 * scale)))
        cv2.putText(canvas, str(index[name] + 1), (centre[0] + int(6 * scale), centre[1] - int(6 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), max(1, int(scale)))
    return canvas


# ---------------------------------------------------------------- CVAT mask


def polygon_to_cvat_mask(points: list[float], width: int, height: int) -> list[float] | None:
    """CVAT mask points: run lengths of the cropped mask (zeros first), then left, top, right, bottom. Unverified against a live CVAT."""
    cv2, np = _cv()
    if cv2 is None:
        return None
    x1, y1, x2, y2 = bbox(points, "polygon")
    left, top = int(max(0, math.floor(x1))), int(max(0, math.floor(y1)))
    right, bottom = int(min(width - 1, math.ceil(x2))), int(min(height - 1, math.ceil(y2)))
    if right <= left or bottom <= top:
        return None
    canvas = np.zeros((bottom - top + 1, right - left + 1), np.uint8)
    local = np.array([[round(x) - left, round(y) - top] for x, y in pairs(points)], dtype=np.int32)
    cv2.fillPoly(canvas, [local], 1)
    flat_mask = canvas.reshape(-1)
    runs: list[int] = []
    current, length = 0, 0
    for value in flat_mask:
        if int(value) == current:
            length += 1
        else:
            runs.append(length)
            current, length = int(value), 1
    runs.append(length)
    return [*runs, left, top, right, bottom]
