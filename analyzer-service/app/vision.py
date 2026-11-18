"""Annotate a CVAT frame with Claude. The API key is never written to logs."""

from __future__ import annotations

import base64
import json
import math
import re

import httpx

from . import accuracy
from .geometry import outline_is_template, simplify_polygon

FALLBACK_MODELS = (
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
)


def _vision_error_text(response) -> str:
    text = ""
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            text = str(error.get("message") or "")
    except Exception:
        text = ""
    if not text:
        text = (response.text or "")[:300]
    return " ".join(text.split())[:300]


def _vision_model_missing(response) -> bool:
    message = _vision_error_text(response).lower()
    return "model" in message and any(
        phrase in message for phrase in ("not found", "does not exist", "not available", "invalid model")
    )


class VisionError(RuntimeError):
    pass


CLAUDE_SYSTEM = """You are an annotation agent. You get one image, a list of labels with annotation
rules, and general guidelines. Small reference crops may come first, each with a caption saying which
label the annotator drew. The LAST image in the message is the only one to annotate; coordinates belong
to it alone. Find every object that matches a label and return ONLY JSON:
{"objects":[{"label":"<exact label name>","bbox":[x1,y1,x2,y2],"oriented_box":[cx,cy,width,height,angle],"polygon":[[x,y],...],
"keypoints":[{"name":"<keypoint name>","x":0,"y":0,"v":2}],"score":0.0-1.0,"attributes":{}}]}
Coordinates are pixels in the image you were given (origin top-left).
Each label may list attributes. For every object, fill every attribute of its label in "attributes",
as {"<attribute name>": "<value>"}. Judge each value from what you can see. checkbox: "true" or "false".
select and radio: exactly one of the listed values. number: a number between min and max. text: a short
phrase. If you cannot tell from the image, use the attribute's default. Never invent an attribute name.
Attributes describe the labeled object; they are never separate objects, labels, or boxes. Return one
object entry and one geometry per physical object, with all of that label's attribute values attached
to the same entry in its "attributes" field.
"attribute_habits" says what the annotators in this project chose before. The reference crops show what each chosen
value looks like. Follow what the image shows. Only when it does not show the value, use the usual choice.
By label annotation_type:
- bounding_box: bbox only.
- polygon and mask: bbox only. Tight box around the visible object. The silhouette is traced from the
  image afterwards, so do not invent polygon points and do not turn the box into a triangle.
- points: bbox of the subject plus one keypoint for each name in keypoints.
- skeleton: bbox of the whole subject plus one keypoint for EVERY name in keypoints. Place each point at
  the center of the joint or part, not on the body edge. left/right is the subject's own left/right, not
  the viewer's. v=2 when the joint is visible, v=1 when it is hidden but its position can be estimated,
  v=0 when it is not in the image (use x=0 and y=0). Do not skip a name, and do not guess a position for a
  joint that is outside the frame.
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


def build_prompt(labels: list[dict], lessons: list[str], width: int, height: int, rotated_boxes: bool = False) -> str:
    brief = []
    for label in labels:
        item = {
            "name": label.get("name"),
            "annotation_type": SPEC_TYPE.get(label.get("type") or "", label.get("type") or "any"),
        }
        if label.get("min_size"):
            item["min_size_px"] = label["min_size"]
        if label.get("attributes"):
            from .attributes import describe

            item["attributes"] = [describe(attr) for attr in label["attributes"]]
            if label.get("learned_attributes"):
                item["attribute_habits"] = label["learned_attributes"]
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
    rotated_guidance = (
        "For every rectangle label, return oriented_box=[cx,cy,width,height,angle] in image pixels. "
        "The center and width/height describe the tight rectangle BEFORE rotation. Angle is degrees clockwise, "
        "from 0 inclusive to 360 exclusive; use 0 only when the object's long edges align with the image axes. "
        "Estimate the object's actual orientation, even when it is strongly tilted. Do not use the axis-aligned "
        "bbox as the oriented_box. Do not return oriented_box for polygons or other shape types.\n"
        if rotated_boxes else ""
    )
    return f"Image size: {width}x{height} px.\n{rotated_guidance}Labels:\n{json.dumps(brief, indent=1)}\nGeneral guidelines:\n{guidelines}"


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
                    "hint_bbox": _scaled_bbox(item, scale, width, height),
                    "attributes": item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
                    "score": _score(item),
                }
            )
            continue
        supported = {"rectangle", "polygon", "polyline", "points", "ellipse", "mask"}
        kind = label.get("type") if label.get("type") in supported else None
        if kind is None:
            continue
        raw_points = _keypoint_pairs(item) if kind == "points" else []
        raw_points = raw_points or _reference_points(item, kind) or item.get("points") or []
        if kind == "rectangle" and len(raw_points) < 4:
            oriented_box = item.get("oriented_box") or item.get("rotated_bbox")
            if isinstance(oriented_box, (list, tuple)) and len(oriented_box) >= 5:
                try:
                    cx, cy, box_width, box_height = (float(value) for value in oriented_box[:4])
                    raw_points = [
                        cx - box_width / 2, cy - box_height / 2,
                        cx + box_width / 2, cy + box_height / 2,
                    ]
                except (TypeError, ValueError):
                    raw_points = []
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
        if kind == "polygon" and len(points) >= 6:
            points = simplify_polygon(points)
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
        parsed = {
                "label": label["name"],
                "type": kind,
                "points": [round(value, 2) for value in points],
                "attributes": item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
                "score": _score(item),
            }
        if kind == "rectangle":
            try:
                oriented_box = item.get("oriented_box") or item.get("rotated_bbox")
                if isinstance(oriented_box, (list, tuple)) and len(oriented_box) >= 5:
                    cx, cy, box_width, box_height, rotation = (float(value) for value in oriented_box[:5])
                    if not all(math.isfinite(value) for value in (cx, cy, box_width, box_height, rotation)):
                        raise ValueError("The oriented box contains a non-finite value")
                    box_width, box_height = abs(box_width), abs(box_height)
                    if max(abs(cx), abs(cy), abs(box_width), abs(box_height)) <= 1.5:
                        cx *= width
                        box_width *= width
                        cy *= height
                        box_height *= height
                    parsed["points"] = [
                        round((cx - box_width / 2) * scale, 2),
                        round((cy - box_height / 2) * scale, 2),
                        round((cx + box_width / 2) * scale, 2),
                        round((cy + box_height / 2) * scale, 2),
                    ]
                else:
                    rotation = float(item.get("rotation") or 0)
                if math.isfinite(rotation):
                    # CVAT's REST serializer only accepts angles in [0, 360].
                    parsed["rotation"] = rotation % 360
            except (TypeError, ValueError):
                pass
        shapes.append(parsed)
    return shapes


def annotate_image(
    api_key: str,
    model: str,
    image: bytes,
    labels: list[dict],
    lessons: list[str],
    refine: bool = True,
    exemplars: list[dict] | None = None,
    rotated_boxes: bool = False,
) -> list[dict]:
    from .detect import as_detection_box, dedupe_shapes, detector_shapes, labels_needing_vision

    detected: list[dict] = []
    try:
        detected = detector_shapes(image, labels)
    except Exception as exc:
        log_detect(exc)
    pending = labels_needing_vision(labels, detected) if detected else list(labels)
    if rotated_boxes:
        # Axis-aligned detector results need a vision pass to estimate their angle.
        pending_names = {item.get("name") for item in pending}
        pending.extend(
            item for item in labels
            if item.get("type") == "rectangle" and item.get("name") not in pending_names
        )
    vision: list[dict] = []
    if pending:
        if not api_key and not detected:
            raise VisionError("No vision API key is configured.")
        if api_key:
            vision = _ask_vision(api_key, model, image, pending, lessons, refine=False, exemplars=exemplars, rotated_boxes=rotated_boxes)
            tuning = _accuracy_settings()
            if tuning["tiling"]:
                vision = vision + accuracy.detect_tiled(
                    image,
                    lambda tile: _ask_vision(api_key, model, tile, pending, lessons, refine=False, exemplars=exemplars, rotated_boxes=rotated_boxes),
                    tuning["tile_min_side"],
                )
            if refine:
                vision = [as_detection_box(shape) for shape in vision]
            if tuning["verify"]:
                vision = accuracy.verify_boxes(image, vision, pending, _make_ask(api_key, model))
            if rotated_boxes:
                from .geometry import iou

                for detection in detected:
                    if detection.get("type") != "rectangle":
                        continue
                    candidates = [
                        shape for shape in vision
                        if shape.get("label") == detection.get("label")
                        and shape.get("type") == "rectangle"
                        and shape.get("rotation") is not None
                    ]
                    if candidates:
                        match = max(
                            candidates,
                            key=lambda shape: iou(detection["points"], "rectangle", shape["points"], "rectangle"),
                        )
                        if iou(detection["points"], "rectangle", match["points"], "rectangle") >= 0.25:
                            detection["rotation"] = match["rotation"]
    shapes = dedupe_shapes(detected + vision)
    if not refine or not shapes:
        return shapes
    from .refine import refine_shapes

    def ask(system: str, content: list) -> str:
        reply = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 3000,
                "system": system,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=120,
        )
        if reply.status_code >= 400:
            detail = _vision_error_text(reply)
            raise VisionError(detail or f"The vision model returned HTTP {reply.status_code}.")
        return "".join(block.get("text") or "" for block in reply.json().get("content") or [] if block.get("type") == "text")

    traced = refine_shapes(image, shapes, labels, ask if api_key else None)
    return dedupe_shapes(traced)


def _accuracy_settings() -> dict:
    try:
        from .config import get_settings

        settings = get_settings()
        return {
            "tiling": bool(settings.annotation_tiling),
            "tile_min_side": int(settings.annotation_tile_min_side),
            "verify": bool(settings.annotation_verify),
        }
    except Exception:
        return {"tiling": True, "tile_min_side": 2000, "verify": True}


def _make_ask(api_key: str, model: str):
    """ask(system, content) -> text, trying the configured model and then the fallbacks."""
    state = {"name": None}

    def ask(system: str, content: list) -> str:
        names = [state["name"]] if state["name"] else []
        for name in (model, *FALLBACK_MODELS):
            if name and name not in names:
                names.append(name)
        last = "The vision model did not respond."
        for name in names:
            reply = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={"model": name, "max_tokens": 3000, "system": system, "messages": [{"role": "user", "content": content}]},
                timeout=120,
            )
            if reply.status_code == 404 or _vision_model_missing(reply):
                last = f"The vision model {name} is not available."
                continue
            if reply.status_code >= 400:
                raise VisionError(_vision_error_text(reply) or f"The vision model returned HTTP {reply.status_code}.")
            state["name"] = name
            return "".join(b.get("text") or "" for b in reply.json().get("content") or [] if b.get("type") == "text")
        raise VisionError(last)

    return ask


def log_detect(exc: Exception) -> None:
    import logging

    logging.getLogger(__name__).warning("detector skipped: %s", exc)


def _ask_vision(
    api_key: str,
    model: str,
    image: bytes,
    labels: list[dict],
    lessons: list[str],
    refine: bool = True,
    exemplars: list[dict] | None = None,
    rotated_boxes: bool = False,
) -> list[dict]:
    payload, media_type, scale, width, height = prepare_image(image)
    prompt = build_prompt(labels, lessons, width, height, rotated_boxes)
    import base64

    encoded = base64.b64encode(payload).decode("ascii")
    content = _example_blocks(exemplars, {item.get("name") for item in labels})
    content += [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}},
        {"type": "text", "text": prompt},
    ]
    body = {
        "max_tokens": 6000,
        "system": CLAUDE_SYSTEM,
        "messages": [{"role": "user", "content": content}],
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
        if response.status_code == 404 or _vision_model_missing(response):
            last_error = f"The vision model {name} is not available."
            continue
        if response.status_code >= 400:
            detail = _vision_error_text(response)
            raise VisionError(detail or f"The vision model returned HTTP {response.status_code}.")
        text = "".join(block.get("text") or "" for block in response.json().get("content") or [] if block.get("type") == "text")
        shapes = parse_shapes(text, labels, scale, width, height)
        if refine and _coarse_polygons(shapes):
            shapes = _refine_polygons(
                api_key, name, encoded, media_type, labels, lessons, scale, width, height, shapes,
            )
        if refine and shapes:
            from .refine import refine_shapes

            def ask(system: str, content: list) -> str:
                reply = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={"model": name, "max_tokens": 3000, "system": system, "messages": [{"role": "user", "content": content}]},
                    timeout=120,
                )
                if reply.status_code >= 400:
                    detail = _vision_error_text(reply)
                    raise VisionError(detail or f"The vision model returned HTTP {reply.status_code}.")
                return "".join(b.get("text") or "" for b in reply.json().get("content") or [] if b.get("type") == "text")

            shapes = refine_shapes(image, shapes, labels, ask)
        return shapes
    raise VisionError(last_error)


def _example_blocks(exemplars: list[dict] | None, label_names: set) -> list[dict]:
    """Crops of what this project's annotators drew, placed before the image to annotate."""
    import base64

    from .labelmemory import exemplar_caption

    usable = [item for item in exemplars or [] if item.get("crop") and item.get("label") in label_names]
    if not usable:
        return []
    blocks: list[dict] = [
        {
            "type": "text",
            "text": (
                "Reference examples follow. Annotators in this project drew each of these objects by hand. "
                "Use them to learn what each label means here and how they draw it. "
                "They are not part of the image to annotate, so never output coordinates for them."
            ),
        }
    ]
    for item in usable:
        blocks.append({"type": "text", "text": exemplar_caption(item)})
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(item["crop"]).decode("ascii"),
                },
            }
        )
    blocks.append({"type": "text", "text": "Now annotate the following image."})
    return blocks


def _scaled_point(x: float, y: float, scale: float, width: int, height: int) -> tuple[float, float]:
    if max(abs(x), abs(y)) <= 1.5 and (x or y):
        return x * width * scale, y * height * scale
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


def _scaled_bbox(item: dict, scale: float, width: int, height: int) -> list[float] | None:
    box = item.get("bbox")
    if not (isinstance(box, list) and len(box) >= 4):
        return None
    try:
        values = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None
    if max(abs(v) for v in values) <= 1.5:
        return [values[0] * width * scale, values[1] * height * scale, values[2] * width * scale, values[3] * height * scale]
    return [v * scale for v in values]


def _keypoint_pairs(item: dict) -> list:
    """A points label is its keypoints. The bbox corners are not points of the object."""
    pairs_out = []
    for point in item.get("keypoints") or []:
        if not isinstance(point, dict):
            continue
        try:
            if int(point.get("v", 2)) <= 0:
                continue
            pairs_out.append([float(point["x"]), float(point["y"])])
        except (KeyError, TypeError, ValueError):
            continue
    return pairs_out


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
    return any(
        shape.get("type") == "polygon"
        and (
            len(shape.get("points") or []) < 16
            or outline_is_template(shape.get("points") or [])
        )
        for shape in shapes
    )


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
        "\nThe previous outline used one fixed point count and did not follow the silhouette. "
        "Trace the visible boundary again. Add a point only where the edge turns, and leave straight edges with just their corners. "
        "Two objects must not get the same point count unless their outlines really match."
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
{"summary":"","workflow":"","tracking":false,"decisions":[{"name":"","type":"rectangle|polygon|polyline|points|ellipse|cuboid|mask|skeleton|tag|interval","why_this_type":"","how_to_annotate":"","how_to_decide":"","attributes":[{"name":"","input_type":"select|radio|checkbox|number|text","values":[],"default":"","mutable":true}],"keypoints":[],"edges":[["joint","joint"]]}]}
A document does not have to use a Label: heading. Read skeleton properties, joint lists, and connections as one skeleton label.
keypoints are the joint names in the document's order. edges are only the connections the document names, as pairs of those names.
Do not connect a joint to the next name in the list unless the document connects them.
Create one decision for every class the document describes. Do not invent a class that is not described.
If labels were already read, keep them and add only classes the document still describes.
When existing CVAT project labels are supplied, map requirement classes to those exact names only when the document clearly refers to them. Do not create duplicate aliases. If mapping is unclear, do not guess.
Set tracking true when objects must keep the same id across frames.
attributes: inspect all supplied text and every attached page/image, including tables, screenshots, scans, diagrams,
captions, and notes. List every attribute the document asks for on that class, also when it is only described in a sentence
(for example "record the colour of each mirror" is an attribute named colour). Use select or radio with values when the
document lists the choices, checkbox for yes/no, number for a measure (values are [min, max, step]), and text for free entry.
Copy specified defaults and values exactly. Leave defaults empty when none is specified. Do not infer a value or attribute.
mutable is false when the value stays the same for the whole object on a video. Do not invent an attribute the document does not ask for.
"""


def analyze_requirement(
    api_key: str,
    model: str,
    text: str,
    parsed: dict,
    images: list[dict] | None = None,
    existing_labels: list[dict] | None = None,
) -> dict:
    """Extract label rules from all document text and embedded visual requirement pages."""
    from .spec import explain_requirement

    local = explain_requirement(parsed)
    if not api_key:
        if text or images:
            local["extraction_warning"] = "Visual requirement analysis was skipped because no vision API key is configured."
        return local
    document_text = (text or "").strip()
    if not document_text and not images and not parsed.get("labels"):
        return local
    content = [{
        "type": "text",
        "text": "Labels already read from the document:\n"
        + json.dumps(local["decisions"], indent=1)
        + "\n\nExisting CVAT project labels (use these exact names for matching classes; do not create aliases):\n"
        + json.dumps([
            {"name": item.get("name"), "type": item.get("type"), "attributes": [
                {"name": attr.get("name"), "input_type": attr.get("input_type"), "values": attr.get("values"),
                 "default_value": attr.get("default_value")}
                for attr in item.get("attributes") or []
            ]}
            for item in existing_labels or []
            if item.get("name") and not item.get("parent_id") and not item.get("has_parent")
        ], indent=1)
        + "\n\nComplete requirement text (read every section):\n"
        + (document_text or "(no selectable text; inspect the attached pages/images)"),
    }]
    for image in images or []:
        raw = image.get("data") or b""
        if raw:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.get("media_type") or "image/jpeg",
                    "data": base64.b64encode(raw).decode("ascii"),
                },
            })
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
            "messages": [{"role": "user", "content": content}],
        },
        timeout=90,
    )
    if response.status_code >= 400:
        message = " ".join((response.text or "").split())[:800]
        raise VisionError(f"Requirement analysis returned HTTP {response.status_code}: {message}")
    raw = "".join(
        block.get("text") or ""
        for block in response.json().get("content") or []
        if block.get("type") == "text"
    )
    data = _load_json(raw)
    decisions = data.get("decisions")
    if not isinstance(decisions, list):
        # Accept a direct list/object response as a recoverable model format
        # variation; the document is data, and all labels still pass normal type
        # validation in spec.apply_guide before they can be applied.
        decisions = data.get("labels") if isinstance(data.get("labels"), list) else []
    cleaned = []
    for decision in decisions:
        if not isinstance(decision, dict) or not str(decision.get("name") or "").strip():
            continue
        normalized = dict(decision)
        raw_type = str(normalized.get("type") or normalized.get("annotation_type") or "").strip().lower()
        normalized["type"] = {
            "bounding_box": "rectangle", "bounding box": "rectangle", "bbox": "rectangle",
            "box": "rectangle", "2d bounding box": "rectangle",
        }.get(raw_type, raw_type)
        if not isinstance(normalized.get("attributes"), list):
            normalized["attributes"] = []
        cleaned.append(normalized)
    if not cleaned and local.get("decisions"):
        local["extraction_warning"] = "The requirement guide response had no usable label decisions; using locally extracted document data."
        return local
    merged = _merge_decisions(local.get("decisions") or [], cleaned)
    if not merged:
        # Keep deterministic document parsing results when the guide response is
        # malformed or incomplete; expose the cause to the UI rather than erasing
        # the locally recovered labels and attributes.
        if local.get("decisions"):
            local["extraction_warning"] = "The requirement guide response had no usable label decisions; using locally extracted document data."
            return local
        raise VisionError("Requirement analysis returned no usable label decisions.")
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
            if key == "attributes" and isinstance(value, list):
                # keep the attributes already read from the document, and add the ones described in words
                seen = {str(a.get("name") or "").casefold() for a in base.get("attributes") or [] if isinstance(a, dict)}
                base["attributes"] = [a for a in base.get("attributes") or [] if isinstance(a, dict)] + [
                    a for a in value if isinstance(a, dict) and str(a.get("name") or "").casefold() not in seen
                ]
            elif value not in (None, "", []):
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
