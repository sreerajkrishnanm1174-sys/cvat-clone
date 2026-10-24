"""Orchestration. CVAT calls stay in the adapter."""

from __future__ import annotations

import base64
import json
import os
from collections import defaultdict

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import learning, planning
from .db import AuditLog, Prediction, ProfileVersion, ProjectState, Run, Sample, SpecVersion, WebhookEvent
from .geometry import (
    apply_box_delta,
    box_size,
    crosses_self,
    iou,
    resample_polygon,
    scale_outline,
)


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
            updated["points"] = resample_polygon(updated["points"], int(point_rules[-1]["payload"]["points"]))
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
        width, height = box_size(shape.get("points") or [], shape.get("type") or "rectangle")
        payload = rule["payload"]
        if width < payload["min_width"] * 0.5 or width > payload["max_width"] * 1.5:
            reasons.append(f"This {label} is outside the usual size")
        elif height < payload["min_height"] * 0.5 or height > payload["max_height"] * 1.5:
            reasons.append(f"This {label} is outside the usual size")
    if shape.get("type") == "polygon" and crosses_self(shape.get("points") or []):
        reasons.append("The outline crosses itself")
    if weak_track:
        reasons.append("The track link to the previous frame is weak and needs a look")
    score = shape.get("score")
    if score is not None and float(score) < 0.25:
        reasons.append("The guess is weak, so start review here")
    return reasons


def associate_by_iou(frames: dict[int, list[dict]], minimum: float = 0.3) -> dict[int, list[dict]]:
    """Link shapes across frames. Every link is a review hint, not an identity."""
    previous: list[dict] | None = None
    linked: dict[int, list[dict]] = {}
    next_track = 1
    for frame in sorted(frames):
        current = []
        for shape in frames[frame]:
            item = dict(shape)
            match = None
            best = minimum
            if previous:
                for prior in previous:
                    score = iou(
                        prior.get("points") or [],
                        prior.get("type") or "rectangle",
                        item.get("points") or [],
                        item.get("type") or "rectangle",
                    )
                    if score > best and prior.get("label") == item.get("label"):
                        best = score
                        match = prior
            if match:
                item["track_id"] = match["track_id"]
                item["weak_track"] = best < 0.5
            else:
                item["track_id"] = next_track
                next_track += 1
                item["weak_track"] = False
            current.append(item)
        linked[frame] = current
        previous = current
    return linked


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
    by_name = {label["name"].lower(): label for label in labels}
    aliases = {}
    for label in labels:
        for alias in label.get("detect_as") or []:
            aliases[str(alias).lower()] = label
    kept = []
    for shape in raw:
        detected = (shape.get("label") or "").lower()
        label = by_name.get(detected) or aliases.get(detected)
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
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0,
        "source": "auto",
        "attributes": attributes,
        "group": shape.get("track_id") or 0,
    }
    if shape["type"] not in {"rectangle", "ellipse"}:
        payload.pop("rotation", None)
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
    }
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
