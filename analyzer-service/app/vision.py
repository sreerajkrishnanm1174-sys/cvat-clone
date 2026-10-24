"""Annotate a CVAT frame with Claude. The API key is never written to logs."""

from __future__ import annotations

import json
import re

import httpx

FALLBACK_MODELS = (
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)


class VisionError(RuntimeError):
    pass


CLAUDE_SYSTEM = """You are an annotation agent. You get one image, a list of labels with annotation
rules, and general guidelines. Find every object that matches a label and return ONLY JSON:
{"objects":[{"label":"<exact label name>","bbox":[x1,y1,x2,y2],"polygon":[[x,y],...],
"keypoints":[{"name":"<keypoint name>","x":0,"y":0,"v":2}],"score":0.0-1.0,"attributes":{}}]}
Coordinates are pixels in the image you were given (origin top-left).
By label annotation_type:
- bounding_box: bbox only.
- polygon and mask: bbox plus polygon. Trace the visible boundary. Put a point wherever the
  edge changes direction. Use as many points as the shape needs, usually 12 to 40.
  A fixed 4-point or 6-point outline is wrong unless the object really has that many corners.
- points: one keypoint for each name in keypoints.
- skeleton: one keypoint for EVERY name in keypoints. v=2 when the joint is visible,
  v=1 when it is hidden but its position can be estimated, v=0 when it is not in the image
  (use x=0 and y=0). Do not skip a name.
Follow each label's include/exclude/min_size_px and how_to_decide. Do not invent objects: if a
label is not visible, return nothing for it. Prefer fewer, correct annotations.
When the prompt contains "Corrections from earlier images in this project", apply those corrections
to this image. They take precedence over your own habits, and the requirement still decides the labels."""

SPEC_TYPE = {
    "rectangle": "bounding_box",
    "polygon": "polygon",
    "mask": "mask",
    "points": "points",
    "polyline": "polyline",
    "ellipse": "ellipse",
    "skeleton": "skeleton",
    "tag": "tag",
}


def prepare_image(raw: bytes, max_side: int = 1568) -> tuple[bytes, str, float, int, int]:
    """Return JPEG bytes, media type, scale back to the original pixels, and the sent size."""
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(raw))
    image = image.convert("RGB")
    original_width, original_height = image.size
    scale = 1.0
    longest = max(original_width, original_height)
    if longest > max_side:
        scale = longest / max_side
        image = image.resize((round(original_width / scale), round(original_height / scale)))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    sent_width, sent_height = image.size
    return buffer.getvalue(), "image/jpeg", scale, sent_width, sent_height


def build_prompt(labels: list[dict], lessons: list[str], width: int, height: int) -> str:
    brief = []
    for label in labels:
        item = {
            "name": label.get("name"),
            "annotation_type": SPEC_TYPE.get(label.get("type") or "", label.get("type") or "any"),
        }
        if label.get("min_size"):
            item["min_size_px"] = label["min_size"]
        if label.get("attributes"):
            item["attributes"] = [
                {"name": attr.get("name"), "values": attr.get("values") or []}
                for attr in label["attributes"]
            ]
        if label.get("how_to_annotate"):
            item["how_to_annotate"] = label["how_to_annotate"]
        if label.get("how_to_decide"):
            item["how_to_decide"] = label["how_to_decide"]
        if label.get("keypoints"):
            item["keypoints"] = list(label["keypoints"])
        if label.get("edges"):
            item["edges"] = list(label["edges"])
        brief.append(item)
    guidelines = "Corrections from earlier images in this project. Apply the same changes here:\n" + "\n".join(
        f"- {lesson}" for lesson in lessons
    ) if lessons else "(none)"
    return f"Image size: {width}x{height} px.\nLabels:\n{json.dumps(brief, indent=1)}\nGeneral guidelines:\n{guidelines}"


def parse_shapes(text: str, labels: list[dict], scale: float, width: int, height: int) -> list[dict]:
    data = _load_json(text)
    raw_shapes = data.get("objects") or data.get("shapes") or []
    if isinstance(raw_shapes, dict):
        raw_shapes = list(raw_shapes.values())
    aliases = {}
    for label in labels:
        aliases[str(label.get("name") or "").lower()] = label
        for alias in label.get("detect_as") or []:
            aliases[str(alias).lower()] = label
    shapes = []
    for item in raw_shapes:
        if not isinstance(item, dict):
            continue
        label = aliases.get(str(item.get("label") or "").lower())
        if label is None:
            continue
        if label.get("type") == "skeleton":
            elements = _skeleton_elements(item, list(label.get("keypoints") or []), scale, width, height)
            if not any(not element.get("outside") for element in elements):
                continue
            shapes.append(
                {
                    "label": label["name"],
                    "type": "skeleton",
                    "points": [],
                    "elements": elements,
                    "attributes": item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
                    "score": _score(item),
                }
            )
            continue
        supported = {"rectangle", "polygon", "polyline", "points", "ellipse", "mask"}
        kind = label.get("type") if label.get("type") in supported else None
        if kind is None:
            continue
        raw_points = _reference_points(item, kind) or item.get("points") or []
        if raw_points and isinstance(raw_points[0], (list, tuple)):
            flat = []
            for pair in raw_points:
                flat.extend(pair[:2])
            raw_points = flat
        points = []
        for value in raw_points:
            try:
                points.append(float(value) * scale)
            except (TypeError, ValueError):
                points = []
                break
        if points and max(points) <= 1.5:
            points = [value * (width if index % 2 == 0 else height) for index, value in enumerate(points)]
        if kind == "rectangle" and len(points) < 4:
            continue
        if kind == "polygon" and len(points) < 6:
            continue
        if kind == "polyline" and len(points) < 4:
            continue
        if kind == "points" and len(points) < 2:
            continue
        if kind == "ellipse" and len(points) < 4:
            continue
        min_size = label.get("min_size") or 0
        if min_size and _too_small(points, float(min_size)):
            continue
        shapes.append(
            {
                "label": label["name"],
                "type": kind,
                "points": [round(value, 2) for value in points],
                "attributes": item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
                "score": _score(item),
            }
        )
    return shapes


def annotate_image(
    api_key: str,
    model: str,
    image: bytes,
    labels: list[dict],
    lessons: list[str],
) -> list[dict]:
    if not api_key:
        raise VisionError("No vision API key is configured.")
    payload, media_type, scale, width, height = prepare_image(image)
    prompt = build_prompt(labels, lessons, width, height)
    import base64

    encoded = base64.b64encode(payload).decode("ascii")
    body = {
        "max_tokens": 6000,
        "system": CLAUDE_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    models = []
    for name in (model, *FALLBACK_MODELS):
        if name and name not in models:
            models.append(name)
    last_error = "The vision model did not respond."
    for name in models:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={**body, "model": name},
            timeout=120,
        )
        if response.status_code == 404:
            last_error = f"The vision model {name} is not available."
            continue
        if response.status_code >= 400:
            raise VisionError(f"The vision model returned HTTP {response.status_code}.")
        text = "".join(block.get("text") or "" for block in response.json().get("content") or [] if block.get("type") == "text")
        shapes = parse_shapes(text, labels, scale, width, height)
        if _coarse_polygons(shapes):
            shapes = _refine_polygons(
                api_key, name, encoded, media_type, labels, lessons, scale, width, height, shapes,
            )
        return shapes
    raise VisionError(last_error)


def _scaled_point(x: float, y: float, scale: float, width: int, height: int) -> tuple[float, float]:
    if max(abs(x), abs(y)) <= 1.5 and (x or y):
        return x * width, y * height
    return x * scale, y * scale


def _skeleton_elements(item: dict, names: list[str], scale: float, width: int, height: int) -> list[dict]:
    raw = item.get("keypoints") or []
    named = {}
    ordered = []
    if isinstance(raw, list):
        for point in raw:
            if isinstance(point, dict) and point.get("name"):
                named[str(point["name"]).lower()] = point
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                ordered.append(point)
    if not names:
        names = [str(point.get("name")) for point in raw if isinstance(point, dict) and point.get("name")]
    elements = []
    for index, name in enumerate(names):
        point = named.get(str(name).lower())
        if point is None and index < len(ordered):
            pair = ordered[index]
            point = {"x": pair[0], "y": pair[1], "v": pair[2] if len(pair) > 2 else 2}
        if not isinstance(point, dict):
            elements.append({"label": name, "points": [0, 0], "outside": True})
            continue
        try:
            x, y = float(point.get("x", 0)), float(point.get("y", 0))
            visible = int(point.get("v", 2))
        except (TypeError, ValueError):
            elements.append({"label": name, "points": [0, 0], "outside": True})
            continue
        if visible <= 0:
            elements.append({"label": name, "points": [0, 0], "outside": True})
            continue
        x, y = _scaled_point(x, y, scale, width, height)
        elements.append({"label": name, "points": [round(x, 2), round(y, 2)], "outside": False, "occluded": visible == 1})
    return elements


def _reference_points(item: dict, kind: str) -> list:
    polygon = item.get("polygon")
    if kind == "polygon" and isinstance(polygon, list) and polygon:
        return polygon
    bbox = item.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        if kind == "polygon":
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return [x1, y1, x2, y2]
    return []


def _coarse_polygons(shapes: list[dict]) -> bool:
    return any(shape.get("type") == "polygon" and len(shape.get("points") or []) < 16 for shape in shapes)


def _vertex_count(shapes: list[dict]) -> int:
    return sum(len(shape.get("points") or []) for shape in shapes if shape.get("type") == "polygon")


def _refine_polygons(
    api_key: str,
    model: str,
    encoded: str,
    media_type: str,
    labels: list[dict],
    lessons: list[str],
    scale: float,
    width: int,
    height: int,
    shapes: list[dict],
) -> list[dict]:
    prompt = build_prompt(labels, lessons, width, height)
    prompt += (
        "\nThe previous outlines used a fixed handful of points and missed the boundary. "
        "Trace each visible edge again. Add a point wherever the boundary turns. "
        "Do not return a 4-point or 6-point template."
    )
    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 6000,
                "system": CLAUDE_SYSTEM,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            },
            timeout=120,
        )
    except httpx.HTTPError:
        return shapes
    if response.status_code >= 400:
        return shapes
    text = "".join(block.get("text") or "" for block in response.json().get("content") or [] if block.get("type") == "text")
    refined = parse_shapes(text, labels, scale, width, height)
    if refined and _vertex_count(refined) > _vertex_count(shapes):
        return refined
    return shapes


def _score(item: dict) -> float:
    try:
        return float(item.get("score", 0.8))
    except (TypeError, ValueError):
        return 0.8


GUIDE_SYSTEM = """You read an annotation requirement and decide the CVAT labels for the project.
Return ONLY JSON:
{"summary":"","workflow":"","tracking":false,"decisions":[{"name":"","type":"rectangle|polygon|polyline|points|ellipse|cuboid|mask|skeleton|tag|interval","why_this_type":"","how_to_annotate":"","how_to_decide":"","attributes":"","keypoints":[],"edges":[["joint","joint"]]}]}
A document does not have to use a Label: heading. Read skeleton properties, joint lists, and connections as one skeleton label.
keypoints are the joint names in the document's order. edges are only the connections the document names, as pairs of those names.
Do not connect a joint to the next name in the list unless the document connects them.
Create one decision for every class the document describes. Do not invent a class that is not described.
If labels were already read, keep them and add only classes the document still describes.
Set tracking true when objects must keep the same id across frames.
"""


def analyze_requirement(api_key: str, model: str, text: str, parsed: dict) -> dict:
    """Ask the model how each label should be drawn. Fall back to the requirement text itself."""
    from .spec import explain_requirement

    local = explain_requirement(parsed)
    if not api_key:
        return local
    excerpt = (text or "").strip()[:12000]
    if not excerpt and not parsed.get("labels"):
        return local
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model or FALLBACK_MODELS[0],
            "max_tokens": 2500,
            "system": GUIDE_SYSTEM,
            "messages": [
                {
                    "role": "user",
                    "content": "Labels already read from the document:\n"
                    + json.dumps(local["decisions"], indent=1)
                    + "\n\nRequirement:\n"
                    + (excerpt or "(the document text is not available)"),
                }
            ],
        },
        timeout=90,
    )
    if response.status_code >= 400:
        return local
    raw = "".join(
        block.get("text") or ""
        for block in response.json().get("content") or []
        if block.get("type") == "text"
    )
    data = _load_json(raw)
    merged = _merge_decisions(local.get("decisions") or [], data.get("decisions") or [])
    if not merged:
        return local
    data["decisions"] = merged
    data.setdefault("summary", local["summary"])
    data.setdefault("workflow", local["workflow"])
    if "tracking" not in data:
        data["tracking"] = local.get("tracking", False)
    return data


def _merge_decisions(local: list, found: list) -> list:
    """Keep labels already read, and add classes the document describes in another form."""
    by_name = {item["name"]: dict(item) for item in local if item.get("name")}
    order = list(by_name)
    for item in found:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        name = str(item["name"]).strip()
        base = by_name.get(name, {"name": name, "keypoints": []})
        for key, value in item.items():
            if value not in (None, "", []):
                base[key] = value
        base["name"] = name
        if not isinstance(base.get("keypoints"), list):
            base["keypoints"] = []
        if name not in by_name:
            order.append(name)
        by_name[name] = base
    return [by_name[name] for name in order]


def _load_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"shapes": []}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"shapes": []}
    return data if isinstance(data, dict) else {"shapes": []}


def _too_small(points: list[float], min_size: float) -> bool:
    xs = points[0::2]
    ys = points[1::2]
    if not xs or not ys:
        return False
    return (max(xs) - min(xs)) < min_size and (max(ys) - min(ys)) < min_size
