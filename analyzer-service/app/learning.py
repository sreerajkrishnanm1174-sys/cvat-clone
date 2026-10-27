"""Project learning. Client requirements stay authoritative."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from .geometry import (
    box_delta,
    box_size,
    iou,
    mean_spacing,
    median,
    polygon_area,
)

HUMAN_OBJECTS = 6
HUMAN_IMAGES = 3
CORRECTION_OBJECTS = 3
CORRECTION_IMAGES = 2

MODE_TEXT = {
    "human_first": "You annotate, the agent learns",
    "auto_first": "The agent annotates first, you correct",
    "mixed": "You mix hand annotation and automatic annotation",
}


def shape_changed(agent: dict, human: dict) -> bool:
    if agent.get("label") != human.get("label"):
        return True
    if agent.get("type") != human.get("type"):
        return True
    if _attrs(agent) != _attrs(human):
        return True
    ap, hp = agent.get("points") or [], human.get("points") or []
    if len(ap) != len(hp):
        return True
    if any(abs(float(a) - float(b)) > 1.0 for a, b in zip(ap, hp)):
        return True
    return _elements_changed(agent.get("elements") or [], human.get("elements") or [])


def _elements_changed(agent: list[dict], human: list[dict]) -> bool:
    by_name = {item.get("name"): item for item in human}
    if {item.get("name") for item in agent} != set(by_name):
        return True
    for item in agent:
        other = by_name.get(item.get("name")) or {}
        if bool(item.get("outside")) != bool(other.get("outside")):
            return True
        left = item.get("points") or []
        right = other.get("points") or []
        if any(abs(float(a) - float(b)) > 1.0 for a, b in zip(left, right)):
            return True
    return False


def _attrs(shape: dict) -> dict:
    raw = shape.get("attributes") or {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {str(item.get("name")): str(item.get("value")) for item in raw if item.get("name")}


def classify_frame(snapshot: list[dict], final: list[dict]) -> list[dict]:
    """Split a finished frame into human, correction, untouched, or imported samples."""
    used: set[int] = set()
    samples: list[dict] = []
    for pred in snapshot:
        best = None
        best_iou = 0.3
        for index, shape in enumerate(final):
            if index in used or shape.get("source") == "file":
                continue
            score = iou(
                pred.get("points") or [],
                pred.get("type") or "rectangle",
                shape.get("points") or [],
                shape.get("type") or "rectangle",
            )
            if score > best_iou:
                best_iou = score
                best = index
        if best is None:
            samples.append({"role": "deleted", "agent": pred, "human": None})
            continue
        used.add(best)
        human = final[best]
        role = "correction" if shape_changed(pred, human) else "untouched"
        samples.append({"role": role, "agent": pred, "human": human, "iou": best_iou})
    for index, shape in enumerate(final):
        if index in used:
            continue
        source = shape.get("source") or "manual"
        if source == "file":
            samples.append({"role": "imported", "agent": None, "human": shape})
        elif source == "auto":
            samples.append({"role": "untouched", "agent": shape, "human": shape})
        else:
            samples.append({"role": "human", "agent": None, "human": shape})
    return samples


def _images_objects(rows: list[dict]) -> tuple[int, int]:
    images = {row["image"] for row in rows}
    return len(rows), len(images)


def _enough(rows: list[dict], objects: int, images: int) -> bool:
    count, image_count = _images_objects(rows)
    return count >= objects and image_count >= images


def _rule(kind: str, label: str, source: str, rows: list[dict], payload: dict, status: str) -> dict:
    count, images = _images_objects(rows)
    return {
        "kind": kind,
        "label": label,
        "source": source,
        "scope": "project",
        "evidence": {"objects": count, "images": images},
        "confidence": min(1.0, count / 10),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "status": status,
    }


def _spec_labels(spec: dict | None) -> dict[str, dict]:
    return {item["name"]: item for item in (spec or {}).get("labels") or []}


def derive_profile(samples: list[dict], spec: dict | None) -> dict:
    """Turn stored frame samples into rules. One drawing never becomes a rule."""
    spec_labels = _spec_labels(spec)
    human_by_label: dict[str, list[dict]] = defaultdict(list)
    corrections: dict[str, list[dict]] = defaultdict(list)
    deleted: dict[str, list[dict]] = defaultdict(list)
    kept_scores: dict[str, list[dict]] = defaultdict(list)

    for sample in samples:
        for item in sample.get("items") or []:
            role = item.get("role")
            image = sample.get("frame_key")
            human = item.get("human") or {}
            agent = item.get("agent") or {}
            if role == "human" and human:
                human_by_label[human.get("label") or ""].append({"image": image, "shape": human})
            elif role == "correction" and human and agent:
                label = agent.get("label") or human.get("label") or ""
                corrections[label].append({"image": image, "agent": agent, "human": human})
                if human.get("label"):
                    human_by_label[human["label"]].append({"image": image, "shape": human})
            elif role == "deleted" and agent:
                deleted[agent.get("label") or ""].append({"image": image, "agent": agent})
            elif role == "untouched" and agent:
                kept_scores[agent.get("label") or ""].append({"image": image, "agent": agent})

    rules: list[dict] = []
    for label, rows in human_by_label.items():
        if not label or not _enough(rows, HUMAN_OBJECTS, HUMAN_IMAGES):
            continue
        widths, heights, points, spacing = [], [], [], []
        attr_values: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            shape = row["shape"]
            width, height = box_size(shape.get("points") or [], shape.get("type") or "rectangle")
            widths.append(width)
            heights.append(height)
            if shape.get("type") == "polygon":
                count = len(shape.get("points") or []) // 2
                points.append(count)
                spacing.append(mean_spacing(shape.get("points") or []))
            for name, value in _attrs(shape).items():
                attr_values[name].append(value)
        rules.append(
            _rule(
                "size_range",
                label,
                "human",
                rows,
                {
                    "min_width": min(widths),
                    "max_width": max(widths),
                    "min_height": min(heights),
                    "max_height": max(heights),
                },
                "active",
            )
        )
        spec_min = (spec_labels.get(label) or {}).get("min_size")
        if spec_min is not None and min(widths) < float(spec_min):
            rules[-1]["status"] = "conflict"
            rules[-1]["payload"]["spec_min_size"] = spec_min
            rules[-1]["payload"]["note"] = "The requirement's minimum size stays in force."
        if points and _enough(
            [row for row in rows if row["shape"].get("type") == "polygon"],
            HUMAN_OBJECTS,
            HUMAN_IMAGES,
        ):
            rules.append(
                _rule(
                    "polygon_points",
                    label,
                    "human",
                    [row for row in rows if row["shape"].get("type") == "polygon"],
                    {"points": int(round(median(points))), "spacing": median(spacing)},
                    "active",
                )
            )
        for name, values in attr_values.items():
            if not _enough(rows, HUMAN_OBJECTS, HUMAN_IMAGES):
                continue
            common = max(set(values), key=values.count)
            if values.count(common) < HUMAN_OBJECTS:
                continue
            spec_attr = {
                attr["name"]: attr
                for attr in (spec_labels.get(label) or {}).get("attributes") or []
            }.get(name)
            status = "active"
            if spec_attr and spec_attr.get("default") not in (None, "", common):
                status = "conflict"
            rules.append(
                _rule(
                    "attribute_default",
                    label,
                    "human",
                    rows,
                    {"name": name, "value": common},
                    status,
                )
            )

    for label, rows in corrections.items():
        if not _enough(rows, CORRECTION_OBJECTS, CORRECTION_IMAGES):
            continue
        box_rows = [
            row
            for row in rows
            if row["agent"].get("type") == "rectangle" and row["human"].get("type") == "rectangle"
        ]
        if _enough(box_rows, CORRECTION_OBJECTS, CORRECTION_IMAGES):
            deltas = [box_delta(row["agent"]["points"], row["human"]["points"]) for row in box_rows]
            rules.append(
                _rule(
                    "box_offset",
                    label,
                    "correction",
                    box_rows,
                    {key: median([item[key] for item in deltas]) for key in deltas[0]},
                    "active",
                )
            )
        poly_rows = [
            row
            for row in rows
            if row["agent"].get("type") == "polygon" and row["human"].get("type") == "polygon"
        ]
        if _enough(poly_rows, CORRECTION_OBJECTS, CORRECTION_IMAGES):
            scales = []
            counts = []
            for row in poly_rows:
                agent_area = polygon_area(row["agent"].get("points") or [])
                human_area = polygon_area(row["human"].get("points") or [])
                if agent_area > 0:
                    scales.append((human_area / agent_area) ** 0.5)
                counts.append(len(row["human"].get("points") or []) // 2)
            if scales:
                rules.append(
                    _rule(
                        "outline_scale",
                        label,
                        "correction",
                        poly_rows,
                        {"scale": median(scales)},
                        "active",
                    )
                )
            rules.append(
                _rule(
                    "polygon_points",
                    label,
                    "correction",
                    poly_rows,
                    {"points": int(round(median(counts)))},
                    "active",
                )
            )
        relabels = [row for row in rows if row["agent"].get("label") != row["human"].get("label")]
        if _enough(relabels, CORRECTION_OBJECTS, CORRECTION_IMAGES):
            target = max(
                {row["human"].get("label") for row in relabels},
                key=lambda name: sum(row["human"].get("label") == name for row in relabels),
            )
            both_in_spec = label in spec_labels and target in spec_labels
            rules.append(
                _rule(
                    "label_confusion",
                    label,
                    "correction",
                    relabels,
                    {"to": target},
                    "conflict" if both_in_spec else "active",
                )
            )

    for label, rows in deleted.items():
        rejected = rows
        kept = kept_scores.get(label) or []
        if not _enough(rejected, CORRECTION_OBJECTS, CORRECTION_IMAGES):
            continue
        rejected_scores = [
            float(row["agent"]["score"]) for row in rejected if row["agent"].get("score") is not None
        ]
        kept_vals = [float(row["agent"]["score"]) for row in kept if row["agent"].get("score") is not None]
        if not rejected_scores:
            continue
        cutoff = max(rejected_scores)
        if kept_vals:
            cutoff = (max(rejected_scores) + min(kept_vals)) / 2
        rules.append(
            _rule(
                "confidence_cutoff",
                label,
                "correction",
                rejected,
                {"cutoff": cutoff},
                "active",
            )
        )

    return {"rules": rules}


def active_rules(profile: dict | None, kind: str, label: str) -> list[dict]:
    return [
        rule
        for rule in (profile or {}).get("rules") or []
        if rule.get("status") == "active" and rule.get("kind") == kind and rule.get("label") == label
    ]


def explain_rule(rule: dict) -> str:
    label = rule.get("label") or "this label"
    payload = rule.get("payload") or {}
    kind = rule.get("kind")
    evidence = rule.get("evidence") or {}
    prefix = f"Learned from {evidence.get('objects', 0)} objects on {evidence.get('images', 0)} images. "
    if rule.get("status") == "conflict":
        prefix = "Not applied, because the requirement disagrees. " + prefix
    if kind == "box_offset":
        tighter = payload.get("d_xbr", 0) - payload.get("d_xtl", 0) < 0
        fit = "tighter" if tighter else "looser"
        return prefix + f"Your boxes for {label} are {fit} than the agent's, so it now draws them that way."
    if kind == "polygon_points":
        return prefix + (
            f"On {label}, keep a point where the outline turns and drop points that sit on a straight edge. "
            "Do not reuse one point count for every object."
        )
    if kind == "outline_scale":
        scale = float(payload.get("scale") or 1)
        fit = "tighter" if scale < 1 else "looser"
        return prefix + f"Your {label} outlines are {fit} than the agent's, so it now adjusts them."
    if kind == "size_range":
        return prefix + f"A {label} is usually between the sizes you have drawn. Objects outside that are flagged."
    if kind == "attribute_default":
        return prefix + f"You usually set {payload.get('name')} to {payload.get('value')} on {label}."
    if kind == "confidence_cutoff":
        return prefix + f"The agent hides weak {label} guesses below {float(payload.get('cutoff') or 0):.2f}."
    if kind == "label_confusion":
        return prefix + f"The agent often confuses {label} with {payload.get('to')}."
    return prefix + kind


def _lesson_ready(samples: list[dict], item: dict) -> bool:
    """One drawing is remembered, but it is not taught until the evidence threshold is met."""
    label = ((item.get("human") or {}).get("label") or (item.get("agent") or {}).get("label") or "")
    role = item.get("role")
    need_objects = CORRECTION_OBJECTS if role == "correction" else HUMAN_OBJECTS
    need_images = CORRECTION_IMAGES if role == "correction" else HUMAN_IMAGES
    count = 0
    images = set()
    for sample in samples:
        for other in sample.get("items") or []:
            other_label = ((other.get("human") or {}).get("label") or (other.get("agent") or {}).get("label") or "")
            if other.get("role") == role and other_label == label:
                count += 1
                if sample.get("frame_key"):
                    images.add(sample.get("frame_key"))
    return count >= need_objects and len(images) >= need_images


def prompt_lessons(profile: dict | None, samples: list[dict]) -> list[str]:
    """Plain corrections the vision agent must follow on the next images."""
    lines = []
    seen = set()
    for sample in samples:
        for item in sample.get("items") or []:
            if not _lesson_ready(samples, item):
                continue
            sentence = _correction_sentence(item)
            if sentence and sentence not in seen:
                seen.add(sentence)
                lines.append(sentence)
    for rule in (profile or {}).get("rules") or []:
        if rule.get("status") != "active":
            continue
        sentence = explain_rule(rule)
        if sentence not in seen:
            seen.add(sentence)
            lines.append(sentence)
    return lines[:16]


def _correction_sentence(item: dict) -> str | None:
    role = item.get("role")
    agent = item.get("agent") or {}
    human = item.get("human") or {}
    if role == "deleted" and agent.get("label"):
        return f"Do not draw a {agent.get('label')} when it looks like one the annotator deleted."
    if role == "human" and human.get("label"):
        return f"Add a {human.get('label')} when the annotator had to draw one that was missing."
    if role != "correction":
        return None
    if agent.get("label") and human.get("label") and agent.get("label") != human.get("label"):
        return f"Do not label that object {agent.get('label')}. The annotator changed it to {human.get('label')}."
    moved = _moved_joints(agent.get("elements") or [], human.get("elements") or [])
    if moved and human.get("label"):
        return f"On {human.get('label')}, move these joints the way the annotator did: {', '.join(moved[:8])}."
    if (
        human.get("label")
        and agent.get("type") == "rectangle"
        and human.get("type") == "rectangle"
        and len(agent.get("points") or []) >= 4
        and len(human.get("points") or []) >= 4
    ):
        delta = box_delta(agent["points"], human["points"])
        return (
            f"For {human.get('label')} boxes, apply the annotator's correction: "
            f"left {delta['d_xtl']:+.0f}px, top {delta['d_ytl']:+.0f}px, "
            f"right {delta['d_xbr']:+.0f}px, bottom {delta['d_ybr']:+.0f}px."
        )
    if human.get("type") and agent.get("type") and human.get("type") != agent.get("type"):
        return f"Draw {human.get('label')} as a {human.get('type')}."
    if _attrs(agent) != _attrs(human) and human.get("label"):
        return f"On {human.get('label')}, use the attribute values the annotator corrected."
    if human.get("label") and human.get("type") == "polygon":
        return f"Redraw {human.get('label')} tightly around the visible boundary, the way the annotator corrected it."
    if human.get("label"):
        return f"Fit the {human.get('label')} box the way the annotator corrected it."
    return None


def _moved_joints(agent: list[dict], human: list[dict]) -> list[str]:
    by_name = {item.get("name"): item for item in agent}
    moved = []
    for item in human:
        name = item.get("name")
        previous = by_name.get(name)
        if not name or not previous:
            continue
        if _elements_changed([previous], [item]):
            moved.append(str(name))
    return moved


def explain_profile(profile: dict | None, frame_count: int, mode: str) -> dict:
    rules = (profile or {}).get("rules") or []
    return {
        "mode": mode,
        "mode_text": MODE_TEXT.get(mode, MODE_TEXT["mixed"]),
        "finished_frames": frame_count,
        "summary": f"The agent has learned from {frame_count} finished frames.",
        "habits": [explain_rule(rule) for rule in rules],
        "unresolved": [
            explain_rule(rule) for rule in rules if rule.get("status") != "active"
        ],
    }
