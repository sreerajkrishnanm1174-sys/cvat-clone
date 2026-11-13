"""Accuracy passes for vision detections.

A single pass over a downscaled photo is good at "what, and roughly where" and weak at exact
pixels. It is worst on large images with small objects and on crowded scenes. Two passes help:

* tiling: an image much larger than the model's input is also scanned in overlapping tiles at
  full resolution, and the results are merged;
* verification: each detected box is re-checked on a zoomed crop of the original pixels. The model
  tightens the box or says the crop does not hold that object, so false positives are dropped.

Everything degrades to the first-pass result. Nothing here raises to the caller.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from io import BytesIO
from typing import Callable

log = logging.getLogger(__name__)

# ask(system_prompt, content_blocks) -> model text
Ask = Callable[[str, list], str]

TILE_SIDE = 1500
TILE_OVERLAP = 200
MAX_TILES_PER_AXIS = 3
EDGE_MARGIN = 4  # px: a box this close to an inner tile edge is probably cut in half
VERIFY_PADDING = 0.30
VERIFY_MIN_PAD = 20
VERIFY_MAX_SIDE = 768
VERIFY_BATCH = 8
MAX_VERIFY = 40  # per image
MIN_BOX_OVERLAP = 0.3  # a tightened box must still be mostly the same object

VERIFY_SYSTEM = """You check object detections on zoomed crops. For each numbered crop you get a label and a proposed box in that crop's pixels (origin top-left).
Return ONLY JSON: {"results":[{"index":0,"keep":true,"box":[x1,y1,x2,y2]}]}
keep=false when the crop does not show a single object with that label at the proposed place (background, a different object, or only a fragment).
box is the tight box around that one object in the crop's pixels: its edges touch the visible object, nothing is cut off, nothing extra is included.
If you are unsure of the exact edges, repeat the proposed box. Answer every index exactly once."""


# ----------------------------------------------------------------------------- tiling


def tile_windows(
    width: int,
    height: int,
    tile: int = TILE_SIDE,
    overlap: int = TILE_OVERLAP,
    max_per_axis: int = MAX_TILES_PER_AXIS,
) -> list[tuple[int, int, int, int]]:
    """Overlapping (x1, y1, x2, y2) windows that cover the whole image."""

    def spans(size: int) -> list[tuple[int, int]]:
        if size <= tile:
            return [(0, size)]
        count = min(max_per_axis, math.ceil((size - overlap) / (tile - overlap)))
        piece = min(size, math.ceil((size + (count - 1) * overlap) / count))
        step = (size - piece) / (count - 1) if count > 1 else 0
        return [(round(i * step), round(i * step) + piece) for i in range(count)]

    return [(x1, y1, x2, y2) for y1, y2 in spans(height) for x1, x2 in spans(width)]


def needs_tiles(width: int, height: int, min_side: int) -> bool:
    return max(width, height) >= max(int(min_side), TILE_SIDE + 1)


def _offset_shape(shape: dict, dx: float, dy: float) -> dict:
    moved = dict(shape)
    points = list(shape.get("points") or [])
    moved["points"] = [round(value + (dx if index % 2 == 0 else dy), 2) for index, value in enumerate(points)]
    if shape.get("elements"):
        moved["elements"] = [
            {**element, "points": [round(v + (dx if i % 2 == 0 else dy), 2) for i, v in enumerate(element.get("points") or [])]}
            for element in shape["elements"]
        ]
    if shape.get("hint_bbox"):
        box = shape["hint_bbox"]
        moved["hint_bbox"] = [round(box[0] + dx, 2), round(box[1] + dy, 2), round(box[2] + dx, 2), round(box[3] + dy, 2)]
    return moved


def _touches_inner_edge(shape: dict, window: tuple[int, int, int, int], size: tuple[int, int]) -> bool:
    """True when the shape runs into a tile edge that is not the edge of the whole image."""
    from .detect import shape_box

    x1, y1, x2, y2 = (float(v) for v in shape_box(shape))
    wx1, wy1, wx2, wy2 = window
    width, height = size
    return (
        (wx1 > 0 and x1 <= wx1 + EDGE_MARGIN)
        or (wy1 > 0 and y1 <= wy1 + EDGE_MARGIN)
        or (wx2 < width and x2 >= wx2 - EDGE_MARGIN)
        or (wy2 < height and y2 >= wy2 - EDGE_MARGIN)
    )


def detect_tiled(
    image: bytes,
    detect_tile: Callable[[bytes], list[dict]],
    min_side: int,
) -> list[dict]:
    """Run `detect_tile(jpeg_bytes)` on each tile and return shapes in full-image pixels.

    Returns [] when the image is small enough that tiling would not help.
    """
    try:
        from PIL import Image

        picture = Image.open(BytesIO(image)).convert("RGB")
    except Exception:
        return []
    width, height = picture.size
    if not needs_tiles(width, height, min_side):
        return []
    found: list[dict] = []
    for window in tile_windows(width, height):
        x1, y1, x2, y2 = window
        buffer = BytesIO()
        picture.crop(window).save(buffer, format="JPEG", quality=90)
        try:
            shapes = detect_tile(buffer.getvalue())
        except Exception as exc:  # one bad tile must not lose the others
            log.warning("tile %s skipped: %s", window, exc)
            continue
        for shape in shapes:
            moved = _offset_shape(shape, x1, y1)
            if _touches_inner_edge(moved, window, (width, height)):
                continue
            found.append(moved)
    return found


# ----------------------------------------------------------------------------- verification


def _is_box(shape: dict) -> bool:
    return len(shape.get("points") or []) == 4 and shape.get("type") in {None, "rectangle", "polygon"}


def _overlap(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _crop_for(picture, box: list[float]):
    """JPEG crop around a box. Returns bytes, crop origin, scale sent/original, sent size."""
    width, height = picture.size
    x1, y1, x2, y2 = box
    pad_x = max(VERIFY_MIN_PAD, (x2 - x1) * VERIFY_PADDING)
    pad_y = max(VERIFY_MIN_PAD, (y2 - y1) * VERIFY_PADDING)
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x) + 1), min(height, int(y2 + pad_y) + 1)
    if right - left < 8 or bottom - top < 8:
        return None
    crop = picture.crop((left, top, right, bottom))
    scale = 1.0
    longest = max(crop.size)
    if longest > VERIFY_MAX_SIDE:
        scale = VERIFY_MAX_SIDE / longest
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue(), (left, top), scale, crop.size


def verify_boxes(image: bytes, shapes: list[dict], labels: list[dict], ask: Ask | None) -> list[dict]:
    """Tighten boxes and drop false positives by looking at each one on a zoomed crop."""
    if ask is None or not shapes:
        return shapes
    try:
        from PIL import Image

        picture = Image.open(BytesIO(image)).convert("RGB")
    except Exception:
        return shapes
    by_label = {str(item.get("name")): item for item in labels}
    candidates = [index for index, shape in enumerate(shapes) if _is_box(shape)][:MAX_VERIFY]
    if not candidates:
        return shapes
    result = list(shapes)
    removed: set[int] = set()
    for start in range(0, len(candidates), VERIFY_BATCH):
        batch = candidates[start : start + VERIFY_BATCH]
        content: list[dict] = []
        prepared: dict[int, tuple] = {}
        for slot, index in enumerate(batch):
            shape = shapes[index]
            box = [float(v) for v in shape["points"]]
            crop = _crop_for(picture, box)
            if crop is None:
                continue
            data, origin, scale, size = crop
            prepared[slot] = (index, origin, scale, size, box)
            local = [round((box[0] - origin[0]) * scale), round((box[1] - origin[1]) * scale),
                     round((box[2] - origin[0]) * scale), round((box[3] - origin[1]) * scale)]
            label = by_label.get(str(shape.get("label"))) or {}
            note = f"Crop {slot}: label '{shape.get('label')}'"
            hint = label.get("how_to_annotate") or label.get("how_to_decide")
            if hint:
                note += f" ({str(hint)[:200]})"
            note += f". Crop size {size[0]}x{size[1]}. Proposed box {local}."
            content.append({"type": "text", "text": note})
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(data).decode("ascii")}}
            )
        if not prepared:
            continue
        try:
            answer = ask(VERIFY_SYSTEM, content)
            verdicts = _parse_results(answer)
        except Exception as exc:
            log.warning("box verification skipped: %s", exc)
            continue
        for slot, (index, origin, scale, size, box) in prepared.items():
            verdict = verdicts.get(slot)
            if verdict is None:
                continue
            if verdict.get("keep") is False:
                removed.add(index)
                continue
            new = verdict.get("box")
            if not (isinstance(new, (list, tuple)) and len(new) == 4):
                continue
            try:
                lx1, ly1, lx2, ly2 = (float(v) for v in new)
            except (TypeError, ValueError):
                continue
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            fixed = [
                max(0.0, origin[0] + min(lx1, lx2) / scale),
                max(0.0, origin[1] + min(ly1, ly2) / scale),
                min(float(picture.width), origin[0] + max(lx1, lx2) / scale),
                min(float(picture.height), origin[1] + max(ly1, ly2) / scale),
            ]
            if fixed[2] - fixed[0] < 4 or fixed[3] - fixed[1] < 4 or _overlap(box, fixed) < MIN_BOX_OVERLAP:
                continue  # a different object, not a tighter fit: keep the first answer
            result[index] = {**result[index], "points": [round(v, 2) for v in fixed], "verified": True}
    return [shape for index, shape in enumerate(result) if index not in removed]


def _parse_results(text: str) -> dict[int, dict]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return {}
    found: dict[int, dict] = {}
    for item in data.get("results") or []:
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            found[item["index"]] = item
    return found
