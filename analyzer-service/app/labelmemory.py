"""Label memory: learn what each label means from the annotator's own work.

Used when no requirement document exists, and alongside one when it does.
The labels come from the CVAT project. When people annotate by hand, this module keeps:

* small image crops of objects the annotator drew, per label (visual examples), and
* a plain-language note per label: which shape they draw, how big, how many per image.

The next auto-annotation pass sends the crops and notes to the vision agent, so it
annotates new data the way this project's annotators did. A requirement document,
when there is one, stays authoritative: notes only fill what the document leaves empty.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .db import LabelExemplar
from .geometry import bbox, box_size, median

log = logging.getLogger(__name__)

# Most crops kept per label. Older hand drawings are replaced by newer corrections.
KEEP_PER_LABEL = 12
# Crops sent to the vision agent per label, and over all labels, on one image.
SEND_PER_LABEL = 3
# A label with attributes may send more, so each attribute value has an example to learn from.
SEND_PER_LABEL_ATTRIBUTES = 6
SEND_TOTAL = 12
# A label note is written once the annotators have drawn this much of it.
NOTE_OBJECTS = 3
# Ignore boxes smaller than this (pixels). They make useless examples.
MIN_CROP_SIDE = 8
CROP_MAX_SIDE = 256
CROP_PADDING = 0.12

LEARNABLE_ROLES = {"human", "correction"}
# A label whose CVAT type is "any" takes the shape the annotators actually draw.
RESOLVABLE_TYPES = {"rectangle", "polygon", "polyline", "points", "ellipse"}


# ----------------------------------------------------------------------------- geometry


def shape_box(shape: dict) -> list[float] | None:
    """Pixel box (x1, y1, x2, y2) around any shape, or None when it has no usable geometry."""
    kind = shape.get("type") or "rectangle"
    points = [float(value) for value in (shape.get("points") or [])]
    if kind == "mask" and len(points) >= 4:
        # A CVAT mask is run lengths followed by its box.
        x1, y1, x2, y2 = points[-4:]
    elif kind == "skeleton":
        joints = [
            item.get("points") or []
            for item in shape.get("elements") or []
            if not item.get("outside") and len(item.get("points") or []) >= 2
        ]
        if not joints:
            return None
        xs = [float(item[0]) for item in joints]
        ys = [float(item[1]) for item in joints]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        if not points:
            return None
        x1, y1, x2, y2 = bbox(points, kind)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def make_crop(image: bytes, box: list[float]) -> tuple[bytes, int, int, int, int] | None:
    """JPEG crop around a box with some context. Returns bytes, crop size, source size."""
    if not image:
        return None
    try:
        from PIL import Image

        picture = Image.open(BytesIO(image)).convert("RGB")
    except Exception:  # not an image, or an unreadable one
        return None
    width, height = picture.size
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * CROP_PADDING
    pad_y = (y2 - y1) * CROP_PADDING
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x) + 1)
    bottom = min(height, int(y2 + pad_y) + 1)
    if right - left < MIN_CROP_SIDE or bottom - top < MIN_CROP_SIDE:
        return None
    crop = picture.crop((left, top, right, bottom))
    longest = max(crop.size)
    if longest > CROP_MAX_SIDE:
        factor = CROP_MAX_SIDE / longest
        crop = crop.resize((max(1, round(crop.width * factor)), max(1, round(crop.height * factor))))
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue(), crop.width, crop.height, width, height


# ----------------------------------------------------------------------------- capture


def _shape_key(frame: int, shape: dict) -> str:
    """Stable per object: CVAT id when there is one, otherwise a hash of the drawing."""
    if shape.get("id") is not None:
        return f"{frame}:id{shape['id']}:{shape.get('track_id') or ''}"
    raw = json.dumps([shape.get("label"), shape.get("type"), shape.get("points")], sort_keys=True)
    return f"{frame}:h{hashlib.md5(raw.encode()).hexdigest()[:16]}"


def _points_signature(shape: dict) -> str:
    return json.dumps([shape.get("type"), [round(float(v), 1) for v in shape.get("points") or []]])


def capture_frame(
    session: Session,
    client,
    project_id: int,
    task_id: int,
    job_id: int,
    frame: int,
    items: list[dict],
    known_labels: set[str] | None = None,
) -> int:
    """Keep crops of what the annotator drew on one frame. Returns how many were stored.

    `items` are the classified frame samples from learning.classify_frame.
    Only hand-drawn shapes and hand-corrected shapes are learned. Shapes the agent drew
    and the annotator left alone are not, because nobody confirmed them.
    """
    wanted = []
    for item in items or []:
        if item.get("role") not in LEARNABLE_ROLES:
            continue
        shape = item.get("human") or {}
        label = shape.get("label")
        if not label or (known_labels is not None and label not in known_labels):
            continue
        box = shape_box(shape)
        if box is None or min(box[2] - box[0], box[3] - box[1]) < MIN_CROP_SIDE:
            continue
        wanted.append((item["role"], shape, box))
    existing = {
        row.shape_key: row
        for row in session.scalars(
            select(LabelExemplar).where(
                LabelExemplar.cvat_project_id == project_id,
                LabelExemplar.job_id == job_id,
                LabelExemplar.frame == frame,
            )
        ).all()
    }
    # A shape the annotator removed later is no longer a trusted example.
    current = {_shape_key(frame, shape) for _, shape, _ in wanted}
    for key, row in list(existing.items()):
        if key not in current:
            session.delete(row)
            del existing[key]
    if not wanted:
        return 0

    changed = []
    for role, shape, box in wanted:
        key = _shape_key(frame, shape)
        row = existing.get(key)
        if row is not None and row.signature == _points_signature(shape) and row.label == shape["label"]:
            continue  # already learned exactly as drawn: do not fetch the frame again
        changed.append((role, shape, box, key, row))
    if not changed:
        return 0

    try:
        image = client.get_frame(task_id, frame)
    except Exception as exc:  # CVAT busy or frame unreadable: learn it next sync
        log.warning("label memory: frame %s of task %s skipped: %s", frame, task_id, exc)
        return 0

    stored = 0
    for role, shape, box, key, row in changed:
        crop = make_crop(image, box)
        if crop is None:
            continue
        data, crop_w, crop_h, image_w, image_h = crop
        values = {
            "label": shape["label"],
            "shape_type": shape.get("type") or "rectangle",
            "points_json": list(shape.get("points") or []),
            "attributes_json": dict(shape.get("attributes") or {}),
            "role": role,
            "box_json": box,
            "image_w": image_w,
            "image_h": image_h,
            "crop": data,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "signature": _points_signature(shape),
        }
        if row is not None:
            for name, value in values.items():
                setattr(row, name, value)
            row.created_at = datetime.now(timezone.utc)
        else:
            session.add(
                LabelExemplar(
                    cvat_project_id=project_id,
                    job_id=job_id,
                    frame=frame,
                    shape_key=key,
                    task_id=task_id,
                    **values,
                )
            )
        stored += 1
    if stored:
        session.flush()
        for label in {shape["label"] for _, shape, *_ in changed}:
            _trim(session, project_id, label)
    return stored


def _trim(session: Session, project_id: int, label: str) -> None:
    """Keep the newest KEEP_PER_LABEL crops, preferring corrections over first drawings."""
    rows = session.scalars(
        select(LabelExemplar)
        .where(LabelExemplar.cvat_project_id == project_id, LabelExemplar.label == label)
        .order_by(LabelExemplar.id.desc())
    ).all()
    if len(rows) <= KEEP_PER_LABEL:
        return
    # The newest example of every attribute value is protected: it is the only picture of what that value looks like.
    protected: list[LabelExemplar] = []
    seen: set = set()
    for row in rows:
        pairs = _value_pairs(row)
        if pairs - seen:
            protected.append(row)
            seen |= pairs
    keep = {row.id for row in protected[: KEEP_PER_LABEL * 2]}
    ranked = sorted(rows, key=lambda row: (row.role == "correction", row.id), reverse=True)
    for row in ranked:
        if len(keep) >= KEEP_PER_LABEL:
            break
        keep.add(row.id)
    for row in rows:
        if row.id not in keep:
            session.delete(row)


# ----------------------------------------------------------------------------- notes


def derive_label_notes(samples: list[dict]) -> dict[str, dict]:
    """What the annotators did, per label, from stored frame samples. One drawing is not a habit."""
    objects: dict[str, list[dict]] = defaultdict(list)
    per_frame: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        frame = sample.get("frame_key")
        for item in sample.get("items") or []:
            if item.get("role") not in LEARNABLE_ROLES:
                continue
            shape = item.get("human") or {}
            label = shape.get("label")
            if not label:
                continue
            objects[label].append(shape)
            per_frame[label][frame] += 1
    notes: dict[str, dict] = {}
    for label, shapes in objects.items():
        if len(shapes) < NOTE_OBJECTS:
            continue
        kinds = Counter(shape.get("type") or "rectangle" for shape in shapes)
        kind, kind_count = kinds.most_common(1)[0]
        widths, heights, vertices = [], [], []
        attr_values: dict[str, list[str]] = defaultdict(list)
        for shape in shapes:
            box = shape_box(shape)
            if box:
                widths.append(box[2] - box[0])
                heights.append(box[3] - box[1])
            if shape.get("type") == "polygon":
                vertices.append(len(shape.get("points") or []) // 2)
            for name, value in (shape.get("attributes") or {}).items():
                attr_values[str(name)].append(str(value))
        counts = list(per_frame[label].values())
        common_attrs = {}
        attribute_stats: dict[str, dict[str, int]] = {}
        for name, values in attr_values.items():
            tally = Counter(values)
            attribute_stats[name] = dict(tally.most_common())
            value, count = tally.most_common(1)[0]
            if count >= max(2, int(len(values) * 0.6)):
                common_attrs[name] = value
        note = {
            "objects": len(shapes),
            "images": len(counts),
            "type": kind,
            "type_share": round(kind_count / len(shapes), 2),
            "per_image": int(round(median([float(v) for v in counts]))) if counts else 0,
            "width": int(round(median(widths))) if widths else 0,
            "height": int(round(median(heights))) if heights else 0,
            "points": int(round(median([float(v) for v in vertices]))) if vertices else 0,
            "attributes": common_attrs,
            "attribute_stats": attribute_stats,
            "attribute_text": _attribute_text(attribute_stats),
        }
        note["text"] = _note_text(label, note)
        notes[label] = note
    return notes


def _note_text(label: str, note: dict) -> str:
    parts = [f"In this project, annotators draw '{label}' as a {note['type']}"]
    if note["type_share"] < 0.8:
        parts[0] += " (most of the time)"
    if note["width"] and note["height"]:
        parts.append(f"typically about {note['width']}x{note['height']} px")
    if note["per_image"]:
        parts.append(f"usually {note['per_image']} per image where it appears")
    if note["type"] == "polygon" and note["points"]:
        parts.append(f"outlines use around {note['points']} points")
    text = ", ".join(parts) + "."
    if note.get("attribute_text"):
        text += " Attributes: " + note["attribute_text"]
    return text


def _attribute_text(stats: dict[str, dict[str, int]]) -> str:
    """How annotators in this project set each attribute, from what they actually chose."""
    lines = []
    for name, tally in stats.items():
        total = sum(tally.values())
        top, count = next(iter(tally.items()))
        if len(tally) == 1 or count / total >= 0.8:
            lines.append(f"{name} is usually '{top}' ({count} of {total})")
        else:
            lines.append(f"{name} varies: " + ", ".join(f"'{v}' {n}" for v, n in tally.items()))
    return "; ".join(lines) + "." if lines else ""


def apply_notes(spec_labels: list[dict], notes: dict[str, dict]) -> list[str]:
    """Fill what the requirement left empty. Returns the labels that were filled from notes."""
    filled = []
    for label in spec_labels:
        note = notes.get(label.get("name") or "")
        if not note:
            continue
        used = False
        if note.get("attribute_text"):
            # attribute habits are useful next to a requirement too: the document says what exists, the habits say what is chosen
            label["learned_attributes"] = note["attribute_text"]
        if not label.get("how_to_annotate"):
            label["how_to_annotate"] = note["text"]
            used = True
        if (label.get("type") or "any") == "any" and note["type"] in RESOLVABLE_TYPES:
            label["type"] = note["type"]
            used = True
        elif (label.get("type") or "any") == "any" and note["type"] == "mask":
            label["type"] = "polygon"
            used = True
        if used:
            filled.append(label["name"])
    return filled


# ----------------------------------------------------------------------------- exemplars


def select_exemplars(
    session: Session,
    project_id: int,
    label_names: list[str],
    per_label: int = SEND_PER_LABEL,
    total: int = SEND_TOTAL,
) -> list[dict]:
    """Pick a few varied crops per label: the newest, then the ones most unlike it in size."""
    chosen: list[dict] = []
    per_budget = max(1, min(per_label, total // max(1, len(label_names))))
    for name in label_names:
        rows = session.scalars(
            select(LabelExemplar)
            .where(LabelExemplar.cvat_project_id == project_id, LabelExemplar.label == name)
            .order_by(LabelExemplar.id.desc())
        ).all()
        picked: list[LabelExemplar] = []
        remaining = list(rows)
        covered: set = set()
        has_values = any(row.attributes_json for row in rows)
        limit = max(per_budget, min(SEND_PER_LABEL_ATTRIBUTES, total // max(1, len(label_names)))) if has_values else per_budget
        while remaining and len(picked) < limit:
            # prefer an example that shows an attribute value not shown yet, then one unlike the others in size
            best = max(
                remaining,
                key=lambda row: (
                    len(_value_pairs(row) - covered),
                    min((_size_gap(row, other) for other in picked), default=0.0),
                ),
            )
            if len(picked) >= per_budget and not (_value_pairs(best) - covered):
                break
            picked.append(best)
            covered |= _value_pairs(best)
            remaining.remove(best)
        for row in picked:
            chosen.append(_exemplar_dict(row))
    return chosen[:total]


def _value_pairs(row: LabelExemplar) -> set:
    return {(str(name), str(value)) for name, value in (row.attributes_json or {}).items()}


def _size_gap(a: LabelExemplar, b: LabelExemplar) -> float:
    import math

    area_a = max(1.0, (a.box_json[2] - a.box_json[0]) * (a.box_json[3] - a.box_json[1]))
    area_b = max(1.0, (b.box_json[2] - b.box_json[0]) * (b.box_json[3] - b.box_json[1]))
    return abs(math.log(area_a / area_b))


def _exemplar_dict(row: LabelExemplar) -> dict:
    width, height = box_size(row.box_json, "rectangle")
    return {
        "label": row.label,
        "shape_type": row.shape_type,
        "attributes": row.attributes_json or {},
        "size": [int(width), int(height)],
        "points": len(row.points_json or []) // 2 if row.shape_type == "polygon" else 0,
        "crop": row.crop,
    }


def exemplar_caption(item: dict) -> str:
    text = f"Example of label '{item['label']}', drawn by the annotator as a {item['shape_type']}"
    if item.get("size"):
        text += f" about {item['size'][0]}x{item['size'][1]} px in the original image"
    if item.get("attributes"):
        text += ", attributes " + ", ".join(f"{k}={v}" for k, v in item["attributes"].items())
    return text + "."


# ----------------------------------------------------------------------------- report / reset


def memory_report(session: Session, project_id: int, label_names: list[str], notes: dict[str, dict]) -> dict:
    rows = session.scalars(select(LabelExemplar).where(LabelExemplar.cvat_project_id == project_id)).all()
    counts = Counter(row.label for row in rows)
    labels = []
    for name in label_names:
        note = notes.get(name)
        labels.append(
            {
                "label": name,
                "examples_kept": counts.get(name, 0),
                "learned": bool(note) or counts.get(name, 0) > 0,
                "note": note["text"] if note else None,
                "objects_seen": note["objects"] if note else 0,
                "images_seen": note["images"] if note else 0,
                "draws_as": note["type"] if note else None,
                "attributes": note["attribute_stats"] if note else {},
            }
        )
    return {
        "labels": labels,
        "total_examples": len(rows),
        "outside_project_labels": sorted(set(counts) - set(label_names)),
    }


def reset_memory(session: Session, project_id: int, label: str | None = None) -> int:
    query = delete(LabelExemplar).where(LabelExemplar.cvat_project_id == project_id)
    if label:
        query = query.where(LabelExemplar.label == label)
    return int(session.execute(query).rowcount or 0)
