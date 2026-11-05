"""Orchestration. CVAT calls stay in the adapter."""

from __future__ import annotations

import base64
import json
import logging
import math
import os
from collections import defaultdict

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import learning, planning
from .db import AuditLog, Prediction, ProfileVersion, ProjectState, Run, Sample, SpecVersion, WebhookEvent
from .geometry import (
    apply_box_delta,
    bbox,
    box_size,
    covers_same_object,
    crosses_self,
    iou,
    scale_outline,
    simplify_polygon,
)

log = logging.getLogger(__name__)


def get_project(session: Session, project_id: int, organization: str = "") -> ProjectState:
    state = session.get(ProjectState, project_id)
    if state is None:
        state = ProjectState(cvat_project_id=project_id, organization=organization or "")
        session.add(state)
        session.flush()
    return state


def save_spec(session: Session, state: ProjectState, document_name: str, parsed: dict) -> ProjectState:
    state.spec_version += 1
    state.spec_json = parsed
    session.add(
        SpecVersion(
            cvat_project_id=state.cvat_project_id,
            version=state.spec_version,
            document_name=document_name,
            spec_json=parsed,
        )
    )
    _audit(session, state.cvat_project_id, "spec", f"version {state.spec_version}")
    return state


def _attribute_body(attr: dict, existing_id: int | None = None) -> dict:
    body = {
        "name": attr["name"],
        "mutable": True,
        "input_type": attr.get("input_type") or "text",
        "default_value": str(attr.get("default") or attr.get("default_value") or ""),
        "values": list(attr.get("values") or []),
    }
    if existing_id:
        body["id"] = existing_id
    return body


def _label_create_body(label: dict) -> dict:
    from .spec import HUMAN_KEYPOINTS, skeleton_svg

    body = {
        "name": label["name"],
        "type": label["type"],
        "attributes": [_attribute_body(attr) for attr in label.get("attributes") or []],
    }
    if label.get("type") == "skeleton":
        names = [str(item).strip() for item in (label.get("keypoints") or HUMAN_KEYPOINTS) if str(item).strip()]
        body["sublabels"] = [{"name": name, "type": "points"} for name in names]
        body["svg"] = skeleton_svg(names, label.get("edges") or [])
    return body


def skeleton_ids_to_replace(spec: dict, existing: list[dict]) -> list[int]:
    """Existing skeleton labels whose joints or bones do not match the requirement."""
    import re

    from .spec import skeleton_svg

    by_name = {
        item.get("name"): item
        for item in existing
        if not item.get("parent_id") and not item.get("has_parent")
    }
    replace = []
    for label in spec.get("labels") or []:
        if label.get("type") != "skeleton":
            continue
        current = by_name.get(label.get("name"))
        if not current or current.get("id") is None:
            continue
        wanted = [str(name).strip() for name in (label.get("keypoints") or []) if str(name).strip()]
        children = list(current.get("sublabels") or [])
        if not children:
            children = [item for item in existing if item.get("parent_id") == current.get("id")]
            children.sort(key=lambda item: item.get("id") or 0)
        found = [item.get("name") for item in children]
        expected = skeleton_svg(wanted, label.get("edges") or [])
        expected_edges = set(re.findall(r'data-node-from="(\d+)"[^>]*data-node-to="(\d+)"', expected))
        current_edges = set(re.findall(r'data-node-from="(\d+)"[^>]*data-node-to="(\d+)"', current.get("svg") or ""))
        if found != wanted or current_edges != expected_edges:
            replace.append(int(current["id"]))
    return replace


def label_patch(diff: dict, existing: list[dict], confirm: bool) -> list[dict]:
    """Build a project label patch. Omitted labels are left untouched. Nothing is deleted."""
    existing_by_name = {item["name"]: item for item in existing}
    payload = []
    for label in diff["create"]:
        payload.append(_label_create_body(label))
    for change in diff["update"]:
        current = existing_by_name[change["name"]]
        current_attrs = {attr["name"]: attr for attr in current.get("attributes") or []}
        incoming = {attr["name"]: attr for attr in change.get("attributes") or []}
        attributes = []
        for name, attr in current_attrs.items():
            if name in incoming:
                attributes.append(_attribute_body(incoming.pop(name), attr.get("id")))
            else:
                attributes.append(_attribute_body(attr, attr.get("id")))
        for attr in incoming.values():
            attributes.append(_attribute_body(attr))
        item = {"id": change["id"], "name": change["name"], "attributes": attributes}
        blocked = next((entry for entry in diff["blocked"] if entry["name"] == change["name"]), None)
        if blocked and confirm and blocked.get("change", {}).get("to_type"):
            item["type"] = blocked["change"]["to_type"]
        payload.append(item)
    if confirm:
        for entry in diff["blocked"]:
            change = entry.get("change")
            if not change or any(item.get("id") == change.get("id") for item in payload):
                continue
            payload.append({"id": change["id"], "name": change["name"], "type": change.get("to_type")})
    return payload


def rebuild_profile(session: Session, state: ProjectState) -> dict:
    rows = session.scalars(select(Sample).where(Sample.cvat_project_id == state.cvat_project_id)).all()
    samples = [
        {"frame_key": f"{row.job_id}:{row.frame}", "items": row.items_json}
        for row in rows
    ]
    profile = learning.derive_profile(samples, state.spec_json or {})
    if profile != (state.profile_json or {"rules": []}):
        state.profile_version += 1
        state.profile_json = profile
        session.add(
            ProfileVersion(
                cvat_project_id=state.cvat_project_id,
                version=state.profile_version,
                profile_json=profile,
            )
        )
    state.finished_frames = len({(row.job_id, row.frame) for row in rows})
    return profile


def record_frame(
    session: Session,
    state: ProjectState,
    *,
    task_id: int,
    job_id: int,
    frame: int,
    event_key: str,
    snapshot: list[dict],
    final: list[dict],
) -> list[dict]:
    items = learning.classify_frame(snapshot, final)
    existing = session.scalar(
        select(Sample).where(
            Sample.cvat_project_id == state.cvat_project_id,
            Sample.job_id == job_id,
            Sample.frame == frame,
            Sample.event_key == event_key,
        )
    )
    if existing is None:
        session.add(
            Sample(
                cvat_project_id=state.cvat_project_id,
                task_id=task_id,
                job_id=job_id,
                frame=frame,
                event_key=event_key,
                items_json=items,
            )
        )
    else:
        existing.items_json = items
    if state.learn_enabled:
        rebuild_profile(session, state)
    return items


def latest_snapshot(session: Session, project_id: int, job_id: int, frame: int) -> list[dict]:
    row = session.scalar(
        select(Prediction)
        .where(
            Prediction.cvat_project_id == project_id,
            Prediction.job_id == job_id,
            Prediction.frame == frame,
        )
        .order_by(Prediction.id.desc())
    )
    return list(row.shapes_json) if row else []


def remember_event(session: Session, dedupe_key: str, payload: dict) -> bool:
    if session.scalar(select(WebhookEvent).where(WebhookEvent.dedupe_key == dedupe_key)):
        return False
    session.add(WebhookEvent(dedupe_key=dedupe_key, payload_json=payload, status="received"))
    return True


def apply_profile_to_shape(shape: dict, profile: dict | None, label_type: str) -> dict | None:
    label = shape.get("label")
    score = shape.get("score")
    for rule in learning.active_rules(profile, "confidence_cutoff", label):
        if score is not None and float(score) < float(rule["payload"]["cutoff"]):
            return None
    updated = {
        "label": label,
        "type": shape.get("type"),
        "points": list(shape.get("points") or []),
        "attributes": dict(shape.get("attributes") or {}),
        "score": score,
        "source": "auto",
    }
    if shape.get("elements"):
        updated["elements"] = shape["elements"]
    if shape.get("encoded"):
        updated["encoded"] = shape["encoded"]
    for rule in learning.active_rules(profile, "label_confusion", label):
        updated["label"] = rule["payload"]["to"]
        label = updated["label"]
    if updated["type"] == "rectangle":
        offsets = learning.active_rules(profile, "box_offset", label)
        if offsets:
            updated["points"] = apply_box_delta(updated["points"], offsets[-1]["payload"])
    if updated["type"] == "polygon":
        scales = learning.active_rules(profile, "outline_scale", label)
        if scales:
            updated["points"] = scale_outline(updated["points"], float(scales[-1]["payload"]["scale"]))
        point_rules = learning.active_rules(profile, "polygon_points", label)
        if point_rules:
            updated["points"] = simplify_polygon(updated["points"])
    for rule in learning.active_rules(profile, "attribute_default", updated["label"]):
        name = rule["payload"]["name"]
        updated["attributes"].setdefault(name, rule["payload"]["value"])
    if label_type not in {"any", updated["type"]}:
        if label_type == "polygon" and updated["type"] == "rectangle" and len(updated["points"]) >= 4:
            x1, y1, x2, y2 = updated["points"][:4]
            updated["type"] = "polygon"
            updated["points"] = [x1, y1, x2, y1, x2, y2, x1, y2]
        else:
            return None
    return updated


def _measured_geometry(shape: dict) -> tuple[list, str]:
    if shape.get("encoded") == "rle":
        return list(shape.get("points") or [])[-4:], "rectangle"
    return list(shape.get("points") or []), shape.get("type") or "rectangle"


def review_reasons(shape: dict, profile: dict | None, label_spec: dict | None, weak_track: bool = False) -> list[str]:
    reasons = []
    label = shape.get("label")
    spec_label = label_spec or {}
    allowed = {
        attr["name"]: attr for attr in spec_label.get("attributes") or []
    }
    for name, value in (shape.get("attributes") or {}).items():
        attr = allowed.get(name)
        if attr and attr.get("values") and str(value) not in {str(item) for item in attr["values"]}:
            reasons.append(f"{name} is not one of the allowed values")
    if any(rule.get("status") == "conflict" and rule.get("kind") == "label_confusion" and rule.get("label") == label for rule in (profile or {}).get("rules") or []):
        reasons.append(f"{label} looks like another label, so a person should check it")
    for rule in learning.active_rules(profile, "size_range", label):
        measure_points, measure_type = _measured_geometry(shape)
        width, height = box_size(measure_points, measure_type)
        payload = rule["payload"]
        if width < payload["min_width"] * 0.5 or width > payload["max_width"] * 1.5:
            reasons.append(f"This {label} is outside the usual size")
        elif height < payload["min_height"] * 0.5 or height > payload["max_height"] * 1.5:
            reasons.append(f"This {label} is outside the usual size")
    if shape.get("review"):
        reasons.append(str(shape["review"]))
    if shape.get("type") == "polygon" and crosses_self(shape.get("points") or []):
        reasons.append("The outline crosses itself")
    if weak_track:
        reasons.append("The track link to the previous frame is weak and needs a look")
    score = shape.get("score")
    if score is not None and float(score) < 0.25:
        reasons.append("The guess is weak, so start review here")
    return reasons


def associate_by_iou(
    frames: dict[int, list[dict]],
    minimum: float = 0.3,
    gap: int = 45,
    gate: float = 2.0,
    high_confidence: float = 0.55,
) -> dict[int, list[dict]]:
    """Keep one id per object for the whole sequence.

    A detection is assigned to an existing track, including one that was missed
    for a few frames, before a new id is allowed. The match uses the predicted
    position, not only overlap with the previous box, so a fast move stays on
    the same id. The shape type and label never change on that id.
    """
    active: list[dict] = []
    linked: dict[int, list[dict]] = {}
    next_track = 1
    for frame in sorted(frames):
        incoming = list(frames[frame])
        parts = [shape for shape in incoming if shape.get("primary") is False]
        shapes = _collapse_frame([shape for shape in incoming if shape.get("primary") is not False])
        scores = [
            [_track_score(prior, shape, frame, gate) for prior in active]
            for shape in shapes
        ]
        chosen = _assign_tracks(scores, minimum) if active and shapes else []
        assignment = {shape_index: (prior_index, score) for shape_index, prior_index, score in chosen}
        current = []
        for shape_index, shape in enumerate(shapes):
            item = dict(shape)
            if shape_index in assignment:
                prior_index, score = assignment[shape_index]
                prior = active[prior_index]
                detected_label = item.get("label")
                item["track_id"] = prior["track_id"]
                item["label"] = prior.get("track_label") or prior.get("label")
                item["track_label"] = item["label"]
                item["weak_track"] = score < high_confidence
                item["track_confidence"] = round(score, 3)
                missed = frame - int(prior.get("frame") or frame)
                item["track_status"] = "recovered" if missed > 1 else "updated"
                if _geometry_unreliable(prior, shape, score):
                    item["outside"] = True
                    item["review"] = "This frame drifted off the object, so it was not drawn. Check it."
                    item["points"] = list(prior.get("points") or item.get("points"))
                    item["track_status"] = "drift"
                if not item.get("attributes") and prior.get("attributes"):
                    item["attributes"] = dict(prior["attributes"])
                _remember_track(item, prior, frame)
                if detected_label and detected_label != item["label"]:
                    log.info(
                        "track frame=%s id=%s label=%s detection_label=%s confidence=%.3f status=label-kept",
                        frame,
                        item["track_id"],
                        item["label"],
                        detected_label,
                        score,
                    )
                log.log(
                    logging.INFO if item["track_status"] in {"recovered", "drift"} else logging.DEBUG,
                    "track frame=%s id=%s label=%s detection=%s confidence=%.3f status=%s",
                    frame,
                    item["track_id"],
                    item.get("label"),
                    item.get("score"),
                    score,
                    item["track_status"],
                )
            elif any(_same_object(item, other) or _center_inside(item, other) for other in current):
                log.debug(
                    "track frame=%s label=%s detection=%s status=duplicate",
                    frame,
                    item.get("label"),
                    item.get("score"),
                )
                continue
            else:
                item["track_id"] = next_track
                next_track += 1
                item["track_label"] = item.get("label")
                item["weak_track"] = False
                item["track_confidence"] = 1.0
                item["track_status"] = "new"
                _remember_track(item, None, frame)
                log.info(
                    "track frame=%s id=%s label=%s detection=%s confidence=1.000 status=new",
                    frame,
                    item["track_id"],
                    item.get("label"),
                    item.get("score"),
                )
            item["group_id"] = int(item["track_id"])
            item["_seen"] = frame
            current.append(item)
        current.extend(_attach_parts(parts, current))
        seen = {item["track_id"] for item in current if item.get("primary") is not False}
        active = [
            prior
            for prior in active
            if prior.get("track_id") not in seen and frame - int(prior.get("frame") or frame) <= gap
        ]
        active.extend({**item, "frame": frame} for item in current if item.get("primary") is not False)
        linked[frame] = current
    return linked


def _attach_parts(parts: list[dict], primaries: list[dict]) -> list[dict]:
    """Parts from one detection join that object's group. They do not become new objects."""
    parents = {}
    for item in primaries:
        if item.get("instance"):
            parents.setdefault(item["instance"], item)
    attached = []
    for part in parts:
        parent = parents.get(part.get("instance"))
        if parent is None:
            log.info("group label=%s status=unmatched-part", part.get("label"))
            continue
        parent_id = int(parent["track_id"])
        sibling = dict(part)
        sibling["label"] = parent.get("label")
        sibling["track_label"] = parent.get("label")
        sibling["group_id"] = parent_id
        sibling["track_id"] = parent_id * 1000 + int(part.get("part_index") or 1)
        sibling["track_status"] = parent.get("track_status")
        sibling["source"] = parent.get("source") or sibling.get("source")
        if parent.get("attributes") and not sibling.get("attributes"):
            sibling["attributes"] = dict(parent["attributes"])
        log.info(
            "group id=%s track=%s label=%s part=%s status=assigned",
            parent_id,
            sibling["track_id"],
            sibling.get("label"),
            sibling.get("part_index"),
        )
        attached.append(sibling)
    return attached


def _remember_track(item: dict, prior: dict | None, frame: int) -> None:
    source = prior if item.get("outside") and prior else item
    box = _shape_box(source)
    history = list((prior or {}).get("_history") or [])
    history.append((frame, box))
    item["_history"] = history[-8:]
    item["_box"] = box
    item["_velocity"] = _velocity(item["_history"])
    item["_kalman"] = _kalman_update(prior, box, frame)
    item["track_label"] = item.get("track_label") or (prior or {}).get("track_label") or item.get("label")


def _velocity(history: list[tuple[int, list[float]]]) -> tuple[float, float]:
    if len(history) < 2:
        return (0.0, 0.0)
    frame_a, box_a = history[-2]
    frame_b, box_b = history[-1]
    step = max(1, frame_b - frame_a)
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    return ((bx - ax) / step, (by - ay) / step)


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _predict_box(prior: dict, frame: int) -> list[float]:
    """Constant-velocity prediction. The saved shape still comes from the detection."""
    box = list(prior.get("_box") or _shape_box(prior))
    last = int(prior.get("frame") or frame)
    step = min(8, max(0, frame - last))
    state = prior.get("_kalman")
    if state:
        cx, cy, vx, vy = state
    else:
        vx, vy = prior.get("_velocity") or (0.0, 0.0)
        cx, cy = _center(box)
    width = box[2] - box[0]
    height = box[3] - box[1]
    px = cx + vx * step
    py = cy + vy * step
    return [px - width / 2, py - height / 2, px + width / 2, py + height / 2]


def _kalman_update(prior: dict | None, box: list[float], frame: int) -> tuple[float, float, float, float]:
    cx, cy = _center(box)
    previous = (prior or {}).get("_kalman")
    if not previous or prior is None:
        vx, vy = _velocity((prior or {}).get("_history") or [])
        return (cx, cy, vx, vy)
    px, py, pvx, pvy = previous
    step = max(1, frame - int(prior.get("frame") or frame))
    vx = (0.6 * pvx) + (0.4 * (cx - px) / step)
    vy = (0.6 * pvy) + (0.4 * (cy - py) / step)
    return (cx, cy, vx, vy)


def _class_relation(left: str | None, right: str | None) -> str:
    """same, related (car/truck), or different. A different class is not the same object."""
    first = (left or "").strip().lower()
    second = (right or "").strip().lower()
    if not first or not second or first == second:
        return "same"
    from .detect import _group

    left_group = _group(first)
    right_group = _group(second)
    if left_group and right_group and left_group == right_group:
        return "same"
    vehicles = {"car", "truck", "bus", "vehicle", "van", "lorry"}
    if first in vehicles and second in vehicles:
        return "related"
    return "different"


def _geometry_unreliable(prior: dict, shape: dict, score: float) -> bool:
    """A low-confidence box that changes size sharply is not written onto the track."""
    current = _box_area(_shape_box(shape))
    if current <= 1:
        return True
    previous = _box_area(_shape_box(prior))
    if previous <= 1:
        return False
    ratio = current / previous
    if 0.35 <= ratio <= 2.5:
        return False
    return float(shape.get("score") or 0) < 0.5 or score < 0.55


def _assign_tracks(scores: list[list[float]], minimum: float) -> list[tuple[int, int, float]]:
    """One detection per track. The assignment maximizes the match scores."""
    if not scores or not scores[0]:
        return []
    rows = len(scores)
    cols = len(scores[0])
    size = max(rows, cols)
    cost = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(rows):
        for col in range(cols):
            cost[row][col] = -float(scores[row][col])
    chosen = []
    for row, col in _min_assignment(cost):
        if row < rows and col < cols and scores[row][col] >= minimum:
            chosen.append((row, col, scores[row][col]))
    return chosen


def _min_assignment(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Hungarian assignment. Returns (row, column) pairs that minimize cost."""
    size = len(cost)
    inf = float("inf")
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    pair = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        pair[0] = row
        column = 0
        minimum = [inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current = pair[column]
            delta = inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                value = cost[current - 1][candidate - 1] - u[current] - v[candidate]
                if value < minimum[candidate]:
                    minimum[candidate] = value
                    way[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for index in range(size + 1):
                if used[index]:
                    u[pair[index]] += delta
                    v[index] -= delta
                else:
                    minimum[index] -= delta
            column = next_column
            if pair[column] == 0:
                break
        while True:
            previous = way[column]
            pair[column] = pair[previous]
            column = previous
            if column == 0:
                break
    return [(pair[column] - 1, column - 1) for column in range(1, size + 1) if pair[column]]


def _center_inside(shape: dict, other: dict) -> bool:
    """A second outline of an object already tracked on this frame."""
    if _class_relation(shape.get("label"), other.get("label")) == "different":
        return False
    if (shape.get("type") or "rectangle") != (other.get("type") or "rectangle"):
        return False
    box = _shape_box(shape)
    cx, cy = _center(box)
    outer = _shape_box(other)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _collapse_frame(shapes: list[dict]) -> list[dict]:
    ordered = sorted(shapes, key=lambda shape: float(shape.get("score") or 0), reverse=True)
    kept: list[dict] = []
    for shape in ordered:
        if any(_same_object(shape, other) for other in kept):
            continue
        kept.append(shape)
    return kept


def _same_object(shape: dict, other: dict) -> bool:
    if _class_relation(shape.get("label"), other.get("label")) == "different":
        return False
    if (shape.get("type") or "rectangle") != (other.get("type") or "rectangle"):
        return False
    return covers_same_object(_shape_box(shape), _shape_box(other))


def _shape_box(shape: dict) -> list[float]:
    points, kind = _measured_geometry(shape)
    x1, y1, x2, y2 = bbox(points, "rectangle" if shape.get("encoded") == "rle" else kind)
    return [x1, y1, x2, y2]


def _track_score(prior: dict, shape: dict, frame: int = 0, gate: float = 2.0) -> float:
    relation = _class_relation(prior.get("track_label") or prior.get("label"), shape.get("label"))
    if relation == "different" or (prior.get("type") or "rectangle") != (shape.get("type") or "rectangle"):
        return 0.0
    if frame - int(prior.get("frame") or frame) < 0:
        return 0.0
    kind = shape.get("type") or "rectangle"
    last_box = list(prior.get("_box") or _shape_box(prior))
    predicted = _predict_box(prior, frame)
    current = _shape_box(shape)
    if kind == "points":
        score = max(_points_score(prior, shape), _points_score(_shifted_prior(prior, frame), shape))
    elif kind == "polyline":
        score = max(_polyline_score(prior, shape), _polyline_score(_shifted_prior(prior, frame), shape))
    else:
        score = max(_box_affinity(last_box, current, gate), _box_affinity(predicted, current, gate))
    if relation == "related" and score < 0.45:
        return 0.0
    return score * (0.9 if relation == "related" else 1.0)


def _shifted_prior(prior: dict, frame: int) -> dict:
    box = list(prior.get("_box") or _shape_box(prior))
    predicted = _predict_box(prior, frame)
    dx = _center(predicted)[0] - _center(box)[0]
    dy = _center(predicted)[1] - _center(box)[1]
    points = list(prior.get("points") or [])
    shifted = [value + (dx if index % 2 == 0 else dy) for index, value in enumerate(points)]
    return {**prior, "points": shifted}


def _box_affinity(previous: list[float], current: list[float], gate: float) -> float:
    """How likely these two boxes are the same object, even when they barely overlap."""
    overlap = iou(previous, "rectangle", current, "rectangle")
    px, py = _center(previous)
    cx, cy = _center(current)
    distance = math.hypot(cx - px, cy - py)
    pw = max(1.0, previous[2] - previous[0])
    ph = max(1.0, previous[3] - previous[1])
    cw = max(1.0, current[2] - current[0])
    ch = max(1.0, current[3] - current[1])
    diagonal = max(20.0, math.hypot(max(pw, cw), max(ph, ch)))
    if distance > diagonal * gate:
        return 0.0
    center = 1.0 - distance / (diagonal * gate)
    similarity = (min(pw, cw) / max(pw, cw)) * (min(ph, ch) / max(ph, ch))
    return (0.5 * overlap) + (0.3 * center) + (0.2 * similarity)


def _motion_score(a_points: list, a_type: str, b_points: list, b_type: str) -> float:
    """Kept so older callers still compare two boxes. New tracks use _box_affinity."""
    ax1, ay1, ax2, ay2 = bbox(a_points, a_type)
    bx1, by1, bx2, by2 = bbox(b_points, b_type)
    return _box_affinity([ax1, ay1, ax2, ay2], [bx1, by1, bx2, by2], 2.0)


def _points_score(prior: dict, shape: dict) -> float:
    from .geometry import pairs

    before = pairs(prior.get("points") or [])
    after = pairs(shape.get("points") or [])
    if not before or len(before) != len(after):
        return 0.0
    span = max(
        max(point[0] for point in before) - min(point[0] for point in before),
        max(point[1] for point in before) - min(point[1] for point in before),
        20.0,
    )
    shift = sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(before, after)) / len(before)
    return max(0.0, 1.0 - shift / (span * 2))


def _polyline_score(prior: dict, shape: dict) -> float:
    from .geometry import pairs

    before = pairs(prior.get("points") or [])
    after = pairs(shape.get("points") or [])
    if len(before) < 2 or len(after) < 2:
        return 0.0
    prior_points, prior_type = _measured_geometry(prior)
    item_points, item_type = _measured_geometry(shape)
    ends = 0.0
    for start, finish in ((before[0], after[0]), (before[-1], after[-1])):
        ends += ((start[0] - finish[0]) ** 2 + (start[1] - finish[1]) ** 2) ** 0.5
    return max(0.0, iou(prior_points, prior_type, item_points, item_type) - min(0.4, ends / 400.0))


def stabilize_tracks(frames: dict[int, list[dict]], frame_size: tuple[int, int] = (0, 0)) -> dict[int, list[dict]]:
    """Drop a keyframe that jumps off the object or sits outside the picture.

    The track id stays on the frames that are still reliable. A sudden jump in
    the last frames is not written as a visible shape.
    """
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for frame, shapes in frames.items():
        for shape in shapes:
            grouped.setdefault(int(shape.get("track_id") or 0), []).append((frame, shape))
    rejected: set[tuple[int, int]] = set()
    width, height = frame_size
    for track_id, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        steps: list[float] = []
        previous = None
        for frame, shape in rows:
            center = _center(_shape_box(shape))
            if previous is not None:
                steps.append(math.hypot(center[0] - previous[0], center[1] - previous[1]))
            previous = center
        typical = sorted(steps)[len(steps) // 2] if steps else 0.0
        previous = None
        for frame, shape in rows:
            box = _shape_box(shape)
            center = _center(box)
            if width > 1 and height > 1 and _outside_frame(box, width, height):
                shape["outside"] = True
                shape["review"] = "The shape is outside the picture, so it was not drawn."
                rejected.add((frame, track_id))
                previous = None
                continue
            if previous is not None:
                jump = math.hypot(center[0] - previous[0], center[1] - previous[1])
                diagonal = max(20.0, math.hypot(max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])))
                limit = max(48.0, typical * 3, diagonal * 0.8)
                confidence = float(shape.get("track_confidence") or 1)
                if jump > limit and confidence < 0.55:
                    shape["outside"] = True
                    shape["review"] = "This frame drifted off the object, so it was not drawn. Check it."
                    rejected.add((frame, track_id))
                    continue
            previous = center
    if not rejected:
        return frames
    return frames


def recover_drifted(frames: dict[int, list[dict]], frame_image, trace) -> dict[int, list[dict]]:
    """Redraw a drifted keyframe from the last good box. The tracking id stays."""
    for frame, shapes in frames.items():
        for shape in shapes:
            if "drifted" not in str(shape.get("review") or ""):
                continue
            prompt = _previous_box(frames, shape.get("track_id"), frame)
            if prompt is None:
                continue
            try:
                image = frame_image(frame)
                asked = dict(shape)
                asked["points"] = prompt
                if asked.get("type") not in {"polygon", "mask"}:
                    asked["type"] = "mask"
                pieces = (trace(image, [asked]) or {}).get(0) or []
            except Exception:
                continue
            if not pieces:
                continue
            piece = pieces[0]
            if not _recovered_piece(shape, piece, prompt):
                continue
            if shape.get("type") == "rectangle":
                box = _shape_box(piece)
                shape["points"] = [round(value, 2) for value in box]
            else:
                shape["points"] = piece.get("points") or shape.get("points")
                if piece.get("encoded"):
                    shape["encoded"] = piece["encoded"]
            shape["outside"] = False
            shape.pop("review", None)
            shape["track_status"] = "recovered"
    return frames


def _previous_box(frames: dict[int, list[dict]], track_id, frame: int) -> list[float] | None:
    earlier = []
    for seen, shapes in frames.items():
        if seen >= frame:
            continue
        for shape in shapes:
            if shape.get("track_id") == track_id and not shape.get("outside"):
                earlier.append((seen, _shape_box(shape)))
    if not earlier:
        return None
    earlier.sort(key=lambda item: item[0])
    return earlier[-1][1]


def _recovered_piece(shape: dict, piece: dict, prompt: list[float]) -> bool:
    box = _shape_box(piece)
    if _box_area(box) <= 0:
        return False
    return iou(prompt, "rectangle", box, "rectangle") >= 0.15


def _outside_frame(box: list[float], width: int, height: int) -> bool:
    area = _box_area(box)
    if area <= 0:
        return True
    return _inside_area(box, width, height) / area < 0.35


def carry_track_corrections(annotations: dict, snapshots: dict[int, list[dict]]) -> dict | None:
    """Move later keyframes of a track by the same correction, without copying the polygon.

    A later keyframe the annotator already changed is left alone. Manual tracks are left alone.
    """
    changed = False
    tracks = []
    for track in annotations.get("tracks") or []:
        if track.get("source") == "manual":
            tracks.append(track)
            continue
        updated, did = _carry_one_track(track, snapshots)
        changed = changed or did
        tracks.append(updated)
    if not changed:
        return None
    return {**annotations, "tracks": tracks}


def _carry_one_track(track: dict, snapshots: dict[int, list[dict]]) -> tuple[dict, bool]:
    shapes = [dict(shape) for shape in track.get("shapes") or []]
    votes: dict[int, int] = defaultdict(int)
    for shape in shapes:
        if shape.get("outside"):
            continue
        pred = _snapshot_match(snapshots.get(int(shape.get("frame") or 0)) or [], shape)
        if pred is not None and pred.get("track_id") is not None:
            votes[int(pred["track_id"])] += 1
    if not votes:
        return track, False
    track_key = max(votes, key=votes.get)
    delta = None
    corrected_frame = -1
    for shape in shapes:
        if shape.get("outside"):
            continue
        pred = _snapshot_by_track(snapshots.get(int(shape.get("frame") or 0)) or [], track_key)
        if pred is None or not _geometry_changed(pred, shape):
            continue
        if int(shape.get("frame") or 0) >= corrected_frame:
            corrected_frame = int(shape.get("frame") or 0)
            delta = _center_delta(pred, shape)
    if delta is None or corrected_frame < 0 or (abs(delta[0]) < 1.5 and abs(delta[1]) < 1.5):
        return track, False
    did = False
    updated = []
    for shape in shapes:
        frame = int(shape.get("frame") or 0)
        if shape.get("outside") or frame <= corrected_frame:
            updated.append(shape)
            continue
        pred = _snapshot_by_track(snapshots.get(frame) or [], track_key)
        if pred is None or _geometry_changed(pred, shape):
            updated.append(shape)
            continue
        moved = dict(shape)
        moved["points"] = _shift_annotation(shape.get("type") or "rectangle", list(shape.get("points") or []), delta)
        updated.append(moved)
        did = True
    if not did:
        return track, False
    return {**track, "shapes": updated}, True


def _geometry_changed(agent: dict, human: dict) -> bool:
    kind = agent.get("type") or human.get("type") or "rectangle"
    return learning.shape_changed(
        {"label": agent.get("label"), "type": kind, "points": agent.get("points") or [], "attributes": {}},
        {"label": agent.get("label"), "type": human.get("type") or kind, "points": human.get("points") or [], "attributes": {}},
    )


def _shape_view(shape: dict) -> dict:
    return {"label": shape.get("label"), "type": shape.get("type"), "points": shape.get("points") or [], "attributes": {}}


def _snapshot_match(snapshot: list[dict], shape: dict) -> dict | None:
    best = None
    best_score = 0.3
    current = _shape_view(shape)
    for pred in snapshot:
        score = iou(*learning._compare_geometry(pred), *learning._compare_geometry(current))
        if score > best_score:
            best_score = score
            best = pred
    return best


def _snapshot_by_track(snapshot: list[dict], track_key: int) -> dict | None:
    for pred in snapshot:
        if pred.get("track_id") is not None and int(pred["track_id"]) == track_key:
            return pred
    return None


def _center_delta(agent: dict, human: dict) -> tuple[float, float]:
    agent_box = _shape_box(agent) if agent.get("type") != "mask" else _payload_box(agent)
    human_box = _payload_box(human) if human.get("type") else _shape_box(human)
    ax, ay = _center(agent_box)
    hx, hy = _center(human_box)
    return (hx - ax, hy - ay)


def _shift_annotation(kind: str, points: list[float], delta: tuple[float, float]) -> list[float]:
    if kind == "mask" and len(points) >= 4:
        shifted = list(points)
        shifted[-4] = float(shifted[-4]) + delta[0]
        shifted[-3] = float(shifted[-3]) + delta[1]
        shifted[-2] = float(shifted[-2]) + delta[0]
        shifted[-1] = float(shifted[-1]) + delta[1]
        return shifted
    return [float(value) + (delta[0] if index % 2 == 0 else delta[1]) for index, value in enumerate(points)]


def to_cvat_tracks(
    frames: dict[int, list[dict]],
    label_ids: dict[str, dict],
    end_frame: int | None = None,
    frame_size: tuple[int, int] | None = None,
) -> list[dict]:
    """Native CVAT tracks. Each track keeps the geometry of its shapes, including masks and polygons."""
    grouped: dict[int, list[dict]] = {}
    for frame in sorted(frames):
        for shape in frames[frame]:
            payload = to_cvat_shape(shape, label_ids, frame)
            if payload is None or shape.get("track_id") is None:
                continue
            grouped.setdefault(int(shape["track_id"]), []).append(payload)
    tracks = []
    for shapes in grouped.values():
        shapes = sorted(shapes, key=lambda item: int(item["frame"]))
        width, height = frame_size or (0, 0)
        _mark_left_the_frame(shapes, int(width), int(height))
        keyed = []
        previous = None
        for shape in shapes:
            if previous is not None and int(shape["frame"]) > int(previous["frame"]) + 1:
                hidden = dict(previous)
                hidden["frame"] = int(previous["frame"]) + 1
                hidden["outside"] = True
                hidden["occluded"] = True
                keyed.append(_track_shape(hidden))
            keyed.append(_track_shape(shape))
            previous = shape
        last = shapes[-1]
        if end_frame is not None and int(last["frame"]) < int(end_frame):
            hidden = dict(last)
            hidden["frame"] = int(last["frame"]) + 1
            hidden["outside"] = True
            hidden["occluded"] = True
            keyed.append(_track_shape(hidden))
        first = shapes[0]
        for shape in keyed:
            shape["label_id"] = first["label_id"]
        tracks.append(
            {
                "label_id": first["label_id"],
                "frame": int(first["frame"]),
                "group": int(first.get("group") or 0),
                "source": "auto",
                "shapes": keyed,
                "attributes": first.get("attributes") or [],
            }
        )
    return tracks


def _mark_left_the_frame(shapes: list[dict], width: int, height: int) -> None:
    """Mark a track keyframe outside once the object has left the picture.

    CVAT keeps drawing a track until a keyframe has outside set. A box that is
    clipped by the frame edge and much smaller than the same object was inside
    the frame is that keyframe. A box that merely touches the edge stays visible.
    """
    if width <= 1 or height <= 1:
        return
    boxes = [_payload_box(shape) for shape in shapes]
    full = 0.0
    for box in boxes:
        area = _box_area(box)
        if area <= 0 or _touches_border(box, width, height):
            continue
        if _inside_area(box, width, height) > 0.9 * area:
            full = max(full, area)
    for shape, box in zip(shapes, boxes):
        area = _box_area(box)
        if area <= 0:
            continue
        inside = _inside_area(box, width, height)
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        center_out = not (0 <= cx <= width and 0 <= cy <= height)
        mostly_out = inside / area < 0.35
        clipped_away = bool(full) and _touches_border(box, width, height) and area < 0.6 * full
        if center_out or mostly_out or clipped_away:
            shape["outside"] = True
            shape["occluded"] = False


def _payload_box(shape: dict) -> list[float]:
    points = list(shape.get("points") or [])
    kind = shape.get("type") or "rectangle"
    if kind == "mask" and len(points) >= 4:
        points = points[-4:]
        kind = "rectangle"
    x1, y1, x2, y2 = bbox(points, kind)
    return [x1, y1, x2, y2]


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inside_area(box: list[float], width: int, height: int) -> float:
    ix1, iy1 = max(box[0], 0), max(box[1], 0)
    ix2, iy2 = min(box[2], width), min(box[3], height)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _touches_border(box: list[float], width: int, height: int) -> bool:
    return box[0] <= 1 or box[1] <= 1 or box[2] >= width - 1 or box[3] >= height - 1


def _track_shape(shape: dict) -> dict:
    payload = {
        "type": shape["type"],
        "frame": int(shape["frame"]),
        "points": shape["points"],
        "occluded": bool(shape.get("occluded")),
        "outside": bool(shape.get("outside")),
        "z_order": 0,
        "attributes": [],
    }
    if shape["type"] in {"rectangle", "ellipse"}:
        payload["rotation"] = float(shape.get("rotation") or 0)
    return payload


def yolo_endpoint(function_name: str) -> str:
    """Find the published YOLO function. CVAT calls it on a host port, not the Nuclio dashboard."""
    import socket

    if not os.path.exists("/var/run/docker.sock"):
        raise RuntimeError("YOLO is deployed, but this service cannot see its port.")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect("/var/run/docker.sock")
        request = b"GET /containers/json HTTP/1.0\r\nHost: localhost\r\n\r\n"
        sock.sendall(request)
        data = b""
        while chunk := sock.recv(65536):
            data += chunk
    finally:
        sock.close()
    body = data.split(b"\r\n\r\n", 1)[1]
    for item in json.loads(body):
        names = " ".join(item.get("Names") or [])
        if function_name not in names:
            continue
        for port in item.get("Ports") or []:
            if port.get("PrivatePort") == 8080 and port.get("PublicPort"):
                return f"http://host.docker.internal:{port['PublicPort']}"
    raise RuntimeError(f"The detector {function_name} is not running.")


def call_yolo(invoke_url: str, function_name: str, image: bytes, threshold: float) -> list[dict]:
    payload = {"image": base64.b64encode(image).decode("ascii"), "threshold": threshold}
    direct = invoke_url.rstrip("/")
    if direct.endswith("8070") or direct.endswith("8080"):
        response = httpx.post(
            direct + "/api/function_invocations",
            headers={
                "x-nuclio-function-name": function_name,
                "x-nuclio-function-namespace": "nuclio",
                "x-nuclio-path": "/",
            },
            json=payload,
            timeout=120,
        )
    else:
        response = httpx.post(direct, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    shapes = []
    for item in data:
        shapes.append(
            {
                "label": item.get("label"),
                "type": item.get("type") or "rectangle",
                "points": [float(value) for value in item.get("points") or []],
                "attributes": {},
                "score": float(item["confidence"]) if item.get("confidence") is not None else None,
            }
        )
    return shapes


def shapes_for_labels(
    raw: list[dict],
    labels: list[dict],
    profile: dict | None,
) -> list[dict]:
    from .detect import match_label

    by_name = {label["name"].lower(): label for label in labels}
    aliases = {}
    for label in labels:
        for alias in label.get("detect_as") or []:
            aliases[str(alias).lower()] = label
    kept = []
    for shape in raw:
        detected = (shape.get("label") or "").lower()
        label = by_name.get(detected) or aliases.get(detected) or match_label(detected, labels)
        if label is None:
            continue
        adapted = apply_profile_to_shape(
            {**shape, "label": label["name"]},
            profile,
            label.get("type") or "any",
        )
        if adapted:
            kept.append(adapted)
    return kept


def to_cvat_shape(shape: dict, label_ids: dict[str, dict], frame: int) -> dict | None:
    label = label_ids.get(shape["label"])
    if label is None:
        return None
    attributes = []
    specs = {attr["name"]: attr for attr in label.get("attributes") or []}
    for name, value in (shape.get("attributes") or {}).items():
        if name in specs:
            attributes.append({"spec_id": specs[name]["id"], "value": str(value)})
    payload = {
        "type": shape["type"],
        "frame": frame,
        "label_id": label["id"],
        "points": shape["points"],
        "occluded": bool(shape.get("occluded")),
        "outside": bool(shape.get("outside")),
        "z_order": 0,
        "rotation": 0,
        "source": "auto",
        "attributes": attributes,
        "group": int(shape.get("group_id") or 0),
    }
    if shape["type"] not in {"rectangle", "ellipse"}:
        payload.pop("rotation", None)
    if shape["type"] == "polygon":
        from .geometry import outline_is_template
        from .segment import _artificial_diagonal

        points = shape.get("points") or []
        if len(points) < 6 or outline_is_template(points) or _artificial_diagonal(points) or crosses_self(points):
            return None
    if shape["type"] == "mask":
        # Only a SAM mask already stored as CVAT run lengths is saved.
        # Filling the detection box here draws the diagonal across the person.
        if shape.get("encoded") != "rle" or len(shape.get("points") or []) < 5:
            return None
        payload["points"] = [float(value) for value in shape["points"]]
    if shape["type"] == "skeleton":
        sublabels = {item["name"]: item for item in label.get("sublabels") or []}
        elements = []
        for element in shape.get("elements") or []:
            sublabel = sublabels.get(element.get("label"))
            if sublabel is None:
                continue
            outside = bool(element.get("outside"))
            elements.append(
                {
                    "type": "points",
                    "frame": frame,
                    "label_id": sublabel["id"],
                    "points": [0, 0] if outside else list(element.get("points") or [0, 0])[:2],
                    "occluded": bool(element.get("occluded")),
                    "outside": outside,
                    "z_order": 0,
                    "source": "auto",
                    "attributes": [],
                    "group": 0,
                }
            )
        if not elements:
            return None
        payload["points"] = []
        payload["elements"] = elements
    return payload


def group_frames(annotations: dict, labels_by_id: dict[int, dict]) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = defaultdict(list)
    for shape in annotations.get("shapes") or []:
        label = labels_by_id.get(shape.get("label_id"), {})
        frames[int(shape.get("frame") or 0)].append(_from_cvat(shape, label, labels_by_id))
    for track in annotations.get("tracks") or []:
        label = labels_by_id.get(track.get("label_id"), {})
        for shape in track.get("shapes") or []:
            if shape.get("outside"):
                continue
            item = _from_cvat(shape, label, labels_by_id)
            item["source"] = track.get("source") or item.get("source")
            item["track_id"] = track.get("id")
            frames[int(shape.get("frame") or 0)].append(item)
    return frames


def _from_cvat(shape: dict, label: dict, labels_by_id: dict | None = None) -> dict:
    attributes = {}
    by_id = {attr.get("id"): attr.get("name") for attr in label.get("attributes") or []}
    for attr in shape.get("attributes") or []:
        name = attr.get("name") or by_id.get(attr.get("spec_id"))
        if name:
            attributes[name] = attr.get("value")
    elements = []
    for element in shape.get("elements") or []:
        joint = (labels_by_id or {}).get(element.get("label_id"), {})
        elements.append(
            {
                "name": joint.get("name") or "",
                "points": element.get("points") or [],
                "outside": bool(element.get("outside")),
                "occluded": bool(element.get("occluded")),
            }
        )
    result = {
        "label": label.get("name"),
        "type": shape.get("type") or label.get("type"),
        "points": shape.get("points") or [],
        "attributes": attributes,
        "source": shape.get("source") or "manual",
        "score": shape.get("score"),
        "group_id": int(shape.get("group") or 0),
    }
    if shape.get("id") is not None:
        result["id"] = shape.get("id")
    if elements:
        result["elements"] = elements
    return result


def _audit(session: Session, project_id: int | None, action: str, detail: str) -> None:
    session.add(AuditLog(cvat_project_id=project_id, action=action, detail=detail[:500]))


def rollback_profile(session: Session, state: ProjectState, version: int) -> dict:
    row = session.scalar(
        select(ProfileVersion).where(
            ProfileVersion.cvat_project_id == state.cvat_project_id,
            ProfileVersion.version == version,
        )
    )
    if row is None:
        raise KeyError(version)
    state.profile_json = row.profile_json
    state.profile_version = version
    return row.profile_json


def dataset_plan(task: dict, meta: dict, jobs: list[dict], labels: list[dict], allow_external: bool, annotation_count: int | None) -> dict:
    return {
        "dataset": planning.profile_task(task, meta, jobs, annotation_count),
        "agents": planning.plan_agents(labels, allow_external),
    }
