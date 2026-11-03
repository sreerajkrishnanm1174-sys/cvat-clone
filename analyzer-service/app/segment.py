"""Turn a detected object into a CVAT polygon by tracing a SAM mask.

The vision step only identifies the object. The outline comes from the deployed
Segment Anything function: its image embedding, the same ONNX decoder the CVAT
UI already serves, then an OpenCV contour. If SAM is unreachable, the caller
keeps the previous outline.
"""

from __future__ import annotations

import base64
import logging
import os
from functools import lru_cache

import httpx

from .geometry import bbox, crosses_self, flat, pairs

log = logging.getLogger(__name__)

SAM_FUNCTION = "pth-facebookresearch-sam-vit-h"
DECODER_URL = os.getenv("SAM_DECODER_URL", "http://cvat_ui:8000/assets/decoder.onnx")
PRECISIONS = {
    "high": 0.0015,
    "balanced": 0.003,
    "optimized": 0.008,
}


def polygon_precision() -> str:
    name = (os.getenv("POLYGON_PRECISION") or "balanced").strip().lower()
    return name if name in PRECISIONS else "balanced"


def trace_polygons(image: bytes, shapes: list[dict]) -> dict[int, list[dict]]:
    """Replace polygon outlines with contours of a per-object SAM mask.

    The returned map is shape index to one or more shapes. Missing indexes
    mean SAM did not produce a usable mask.
    """
    targets = [index for index, shape in enumerate(shapes) if shape.get("type") in {"polygon", "mask"}]
    if not targets:
        return {}
    cv2, np = _cv()
    if cv2 is None:
        return {}
    image_array = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        return {}
    height, width = image_array.shape[:2]
    try:
        embedding = _image_embedding(image)
        session = _decoder()
    except Exception as exc:
        log.warning("SAM outline skipped: %s", exc)
        return {}
    traced: dict[int, list[dict]] = {}
    boxes = {index: _prompt_box(shapes[index].get("points") or [], width, height) for index in targets}
    for index in targets:
        shape = shapes[index]
        neighbors = [box for other, box in boxes.items() if other != index and box is not None]
        try:
            pieces = _trace_one(session, embedding, width, height, shape, neighbors)
        except Exception as exc:
            log.warning("SAM outline skipped for %s: %s", shape.get("label"), exc)
            pieces = []
        if pieces:
            traced[index] = pieces
    return traced


def mask_to_polygons(mask, width: int, height: int, precision: str | None = None) -> list[list[float]]:
    """Extract clipped contours. Each list is a CVAT point array."""
    cv2, np = _cv()
    if cv2 is None:
        return []
    binary = _refine_mask(mask)
    if binary is None or int(binary.sum()) < 32:
        return []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polygons = []
    if hierarchy is None:
        return []
    # Parent -1 is an outer boundary. Child contours are holes and stay inside that boundary.
    parents = hierarchy[0]
    largest = max((cv2.contourArea(item) for item in contours), default=0)
    for index, contour in enumerate(contours):
        if int(parents[index][3]) != -1:
            continue
        area = cv2.contourArea(contour)
        if area < max(32.0, largest * 0.015):
            continue
        component = np.zeros_like(binary)
        cv2.drawContours(component, [contour], -1, 1, thickness=-1)
        points = _simplify(contour, precision or polygon_precision(), width, height)
        if points and (crosses_self(points) or _artificial_diagonal(points)):
            points = None
        if points and not _covers_mask(points, component):
            points = _simplify(contour, "high", width, height)
        if (
            points
            and not crosses_self(points)
            and not _artificial_diagonal(points)
            and _covers_mask(points, component)
        ):
            polygons.append(points)
    return polygons


def _trace_one(session, embedding, width: int, height: int, shape: dict, neighbors: list | None = None) -> list[dict]:
    box = _prompt_box(shape.get("points") or [], width, height)
    if box is None:
        return []
    extras = _negative_prompts(box, neighbors or [])
    mask = _decode_mask(session, embedding, width, height, box, extras)
    if not _mask_is_body(mask):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        mask = _decode_mask(session, embedding, width, height, box, [*extras, (cx, cy, 1.0)])
    refined = _refine_mask(mask)
    if not _mask_is_body(refined):
        return []
    if shape.get("type") == "mask":
        return _masks_from_binary(shape, refined)
    polygons = mask_to_polygons(refined, width, height)
    if not _mask_followed(refined, polygons):
        return []
    return [{**shape, "type": "polygon", "points": points} for points in polygons]


def _prompt_box(points: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    if len(points) < 4:
        return None
    x1, y1, x2, y2 = bbox(points, "polygon" if len(points) >= 6 else "rectangle")
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (max(0.0, x1), max(0.0, y1), min(float(width - 1), x2), min(float(height - 1), y2))


def _image_embedding(image: bytes):
    from .service import yolo_endpoint

    np = _cv()[1]
    url = yolo_endpoint(SAM_FUNCTION)
    response = httpx.post(
        url,
        json={"image": base64.b64encode(image).decode("ascii")},
        timeout=180,
    )
    response.raise_for_status()
    blob = base64.b64decode(response.json()["blob"])
    embedding = np.frombuffer(blob, dtype=np.float32)
    if embedding.size != 1 * 256 * 64 * 64:
        raise RuntimeError("SAM returned an unexpected embedding.")
    return embedding


@lru_cache(maxsize=1)
def _decoder():
    import onnxruntime as ort

    path = os.getenv("SAM_DECODER_PATH", "")
    if not path or not os.path.isfile(path):
        path = os.path.join(os.getenv("ANALYZER_DATA_DIR", "/data"), "sam-decoder.onnx")
        if not os.path.isfile(path):
            response = httpx.get(DECODER_URL, timeout=60)
            response.raise_for_status()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(response.content)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _negative_prompts(box, neighbors: list) -> list[tuple[float, float, float]]:
    """A point in another object that falls inside this box, so SAM does not swallow the neighbor."""
    prompts = []
    for other in neighbors:
        cx = (other[0] + other[2]) / 2
        cy = (other[1] + other[3]) / 2
        if not (box[0] < cx < box[2] and box[1] < cy < box[3]):
            continue
        ix = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
        iy = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
        union = (box[2] - box[0]) * (box[3] - box[1]) + (other[2] - other[0]) * (other[3] - other[1]) - ix * iy
        if union and (ix * iy) / union >= 0.75:
            continue
        prompts.append((cx, cy, 0.0))
    return prompts[:4]


def _decode_mask(session, embedding, width: int, height: int, box, extras=None):
    np = _cv()[1]
    scale = 1024.0 / max(width, height)
    clicks = [(box[0], box[1], 2.0), (box[2], box[3], 3.0)]
    clicks.extend(extras or [])
    coords = np.array([[x * scale, y * scale] for x, y, _ in clicks], dtype=np.float32)[None, :, :]
    labels = np.array([[label for _, _, label in clicks]], dtype=np.float32)
    outputs = session.run(
        None,
        {
            "image_embeddings": embedding.reshape(1, 256, 64, 64),
            "point_coords": coords,
            "point_labels": labels,
            "orig_im_size": np.array([height, width], dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
        },
    )
    named = {output.name: value for output, value in zip(session.get_outputs(), outputs)}
    mask = np.squeeze(np.asarray(named.get("masks", outputs[0])))
    if mask.ndim != 2:
        return None
    xtl = int(np.asarray(named["xtl"]).reshape(-1)[0]) if "xtl" in named else 0
    ytl = int(np.asarray(named["ytl"]).reshape(-1)[0]) if "ytl" in named else 0
    return _place_decoder_mask(mask, xtl, ytl, width, height)


def _place_decoder_mask(mask, xtl: int, ytl: int, width: int, height: int):
    """Paste SAM's cropped mask onto the image. The crop is often one pixel larger than the box."""
    np = _cv()[1]
    crop = (np.asarray(mask) > 0).astype(np.uint8)
    if crop.ndim != 2 or not crop.any():
        return None
    canvas = np.zeros((height, width), np.uint8)
    src_y = max(0, -ytl)
    src_x = max(0, -xtl)
    dest_y = max(0, ytl)
    dest_x = max(0, xtl)
    rows = min(crop.shape[0] - src_y, height - dest_y)
    cols = min(crop.shape[1] - src_x, width - dest_x)
    if rows <= 0 or cols <= 0:
        return None
    canvas[dest_y : dest_y + rows, dest_x : dest_x + cols] = crop[src_y : src_y + rows, src_x : src_x + cols]
    return canvas if canvas.any() else None


def _refine_mask(mask):
    cv2, np = _cv()
    if cv2 is None:
        return None
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or binary.sum() < 32:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count < 2:
        return None
    largest = max(int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count))
    kept = np.zeros_like(binary)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= max(24, int(largest * 0.015)):
            kept[labels == index] = 1
    speck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, speck)
    return kept


def _mask_is_body(mask) -> bool:
    """Reject an empty mask, a filled rectangle, and a one-pixel diagonal."""
    np = _cv()[1]
    if mask is None:
        return False
    binary = np.asarray(mask)
    area = int(binary.sum())
    if area < 32:
        return False
    ys, xs = np.where(binary > 0)
    box = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    ratio = area / max(1, box)
    return 0.08 < ratio < 0.97


def _component_masks(mask):
    cv2, np = _cv()
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count < 2:
        return []
    largest = max(int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count))
    parts = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < max(32, int(largest * 0.015)):
            continue
        part = np.zeros_like(binary)
        part[labels == index] = 1
        if _mask_is_body(part):
            parts.append(part)
    return parts


def binary_to_cvat_rle(mask) -> list[float] | None:
    """CVAT mask points: run lengths of the cropped mask, then left, top, right, bottom."""
    np = _cv()[1]
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        return None
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    crop = binary[top : bottom + 1, left : right + 1].reshape(-1)
    changes = np.flatnonzero(crop[1:] != crop[:-1]) + 1
    runs = np.diff(np.concatenate(([0], changes, [crop.size]))).astype(int).tolist()
    if int(crop[0]) == 1:
        runs.insert(0, 0)
    decoded = _decode_cvat_rle([*runs, left, top, right, bottom], binary.shape[1], binary.shape[0])
    if decoded is None or int((decoded != binary).sum()) != 0:
        return None
    return [*runs, float(left), float(top), float(right), float(bottom)]


def _decode_cvat_rle(points, width: int, height: int):
    np = _cv()[1]
    if len(points) < 5:
        return None
    left, top, right, bottom = [int(value) for value in points[-4:]]
    if right < left or bottom < top or right >= width or bottom >= height:
        return None
    runs = [int(value) for value in points[:-4]]
    expected = (bottom - top + 1) * (right - left + 1)
    if sum(runs) != expected:
        return None
    flat = np.zeros(expected, np.uint8)
    cursor = 0
    for index, length in enumerate(runs):
        if index % 2 == 1:
            flat[cursor : cursor + length] = 1
        cursor += length
    canvas = np.zeros((height, width), np.uint8)
    canvas[top : bottom + 1, left : right + 1] = flat.reshape(bottom - top + 1, right - left + 1)
    return canvas


def _masks_from_binary(shape: dict, mask) -> list[dict]:
    pieces = []
    for component in _component_masks(mask):
        rle = binary_to_cvat_rle(component)
        if not rle:
            continue
        pieces.append({**shape, "type": "mask", "points": rle, "encoded": "rle"})
    return pieces


def _artificial_diagonal(points: list[float]) -> bool:
    """A 3- or 4-point outline whose longest edge is the box diagonal."""
    ring = pairs(points)
    if len(ring) < 3 or len(ring) > 4:
        return False
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    if span < 8:
        return False
    longest = 0.0
    for index, point in enumerate(ring):
        nxt = ring[(index + 1) % len(ring)]
        longest = max(longest, ((nxt[0] - point[0]) ** 2 + (nxt[1] - point[1]) ** 2) ** 0.5)
    return longest >= 0.85 * span


def _simplify(contour, precision: str, width: int, height: int) -> list[float] | None:
    cv2, _np = _cv()
    ratio = PRECISIONS.get(precision, PRECISIONS["balanced"])
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(0.8, ratio * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 8 and perimeter > 80:
        approx = cv2.approxPolyDP(contour, max(0.8, epsilon * 0.45), True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    clipped = []
    for px, py in approx:
        point = (min(max(float(px), 0.0), width - 1), min(max(float(py), 0.0), height - 1))
        if clipped and abs(point[0] - clipped[-1][0]) < 0.5 and abs(point[1] - clipped[-1][1]) < 0.5:
            continue
        clipped.append(point)
    if len(clipped) > 2 and abs(clipped[0][0] - clipped[-1][0]) < 0.5 and abs(clipped[0][1] - clipped[-1][1]) < 0.5:
        clipped.pop()
    points = [round(value, 2) for value in flat(clipped)]
    if crosses_self(points):
        dense = cv2.approxPolyDP(contour, max(0.8, epsilon * 0.35), True).reshape(-1, 2)
        points = [round(value, 2) for value in flat(
            [(min(max(float(px), 0.0), width - 1), min(max(float(py), 0.0), height - 1)) for px, py in dense]
        )]
        if crosses_self(points) or len(pairs(points)) < 3:
            return None
    return points


def _covers_mask(points: list[float], mask) -> bool:
    cv2, np = _cv()
    drawn = np.zeros_like(mask)
    ring = np.round(np.array(pairs(points), dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(drawn, [ring], 1)
    union = int(((drawn == 1) | (mask == 1)).sum())
    overlap = int(((drawn == 1) & (mask == 1)).sum())
    return union > 0 and overlap / union >= 0.82


def _mask_followed(mask, polygons: list[list[float]]) -> bool:
    if mask is None or not polygons:
        return False
    cv2, np = _cv()
    drawn = np.zeros_like(mask)
    for points in polygons:
        ring = np.round(np.array(pairs(points), dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(drawn, [ring], 1)
    union = int(((drawn == 1) | (mask == 1)).sum())
    overlap = int(((drawn == 1) & (mask == 1)).sum())
    return union > 0 and overlap / union >= 0.82


def _cv():
    try:
        import cv2
        import numpy as np

        return cv2, np
    except ImportError:
        return None, None
