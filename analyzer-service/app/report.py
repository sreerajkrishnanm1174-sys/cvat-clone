"""Project requirement steps and the annotations currently on its tasks."""

from __future__ import annotations


def requirement_steps(spec: dict | None) -> list[dict]:
    parsed = spec or {}
    labels = [item for item in parsed.get("labels") or [] if item.get("name")]
    decisions = {
        item.get("name"): item
        for item in ((parsed.get("guide") or {}).get("decisions") or [])
        if item.get("name")
    }
    summary = ((parsed.get("guide") or {}).get("summary") or "").strip()
    steps = [
        {
            "step": 1,
            "title": "Requirement",
            "detail": summary or "Use the labels and drawing method from this project's requirement.",
        }
    ]
    for label in labels:
        decision = decisions.get(label["name"]) or {}
        parts = [
            decision.get("how_to_annotate") or "",
            decision.get("how_to_decide") or "",
        ]
        if label.get("keypoints"):
            parts.append("Keypoints: " + ", ".join(str(name) for name in label["keypoints"][:12]))
        if label.get("edges"):
            parts.append("Bones: " + ", ".join(f"{pair[0]}-{pair[1]}" for pair in label["edges"][:12]))
        steps.append(
            {
                "step": len(steps) + 1,
                "title": f"Annotate {label['name']} ({label.get('type') or 'any'})",
                "detail": " ".join(part.strip() for part in parts if part and str(part).strip()),
            }
        )
    output_format = str(parsed.get("output_format") or "CVAT")
    if output_format.lower() not in {"", "cvat"}:
        steps.append(
            {
                "step": len(steps) + 1,
                "title": "Export",
                "detail": f"When the dataset export finishes, also write the annotations as {output_format}.",
            }
        )
    return steps


def extra_label_deletions(spec: dict | None, existing: list[dict]) -> list[dict]:
    """Labels that are duplicated or are not in the requirement."""
    wanted = []
    for item in (spec or {}).get("labels") or []:
        name = item.get("name")
        if name and name not in wanted:
            wanted.append(name)
    wanted_names = set(wanted)
    kept: set[str] = set()
    removed_ids: set[int] = set()
    deletions = []
    for item in existing:
        if item.get("parent_id") or item.get("has_parent"):
            continue
        name = item.get("name") or ""
        label_id = item.get("id")
        if label_id is None:
            continue
        if name in wanted_names and name not in kept:
            kept.add(name)
            continue
        removed_ids.add(int(label_id))
        deletions.append({"id": int(label_id), "deleted": True, "name": name})
    for item in existing:
        parent = item.get("parent_id")
        label_id = item.get("id")
        if parent in removed_ids and label_id is not None:
            deletions.append({"id": int(label_id), "deleted": True, "name": item.get("name") or ""})
    return deletions


def project_report(client, project_id: int, spec: dict | None) -> dict:
    parsed = spec or {}
    labels = client.get_labels(project_id)
    top_labels = [
        {"id": item.get("id"), "name": item.get("name"), "type": item.get("type")}
        for item in labels
        if not item.get("parent_id") and not item.get("has_parent")
    ]
    annotations = []
    for task in client.list_tasks(project_id)[:30]:
        task_id = int(task["id"])
        shapes = 0
        for job in client.get_jobs(task_id=task_id):
            current = client.get_annotations("jobs", int(job["id"]))
            shapes += len(current.get("shapes") or [])
        annotations.append(
            {
                "id": task_id,
                "name": task.get("name") or f"Task {task_id}",
                "images": int(task.get("size") or 0),
                "shapes": shapes,
            }
        )
    return {
        "project_id": project_id,
        "output_format": parsed.get("output_format") or "CVAT",
        "steps": requirement_steps(parsed),
        "requirement_labels": [
            {"name": item.get("name"), "type": item.get("type")}
            for item in parsed.get("labels") or []
            if item.get("name")
        ],
        "cvat_labels": top_labels,
        "extra_labels": [item["name"] for item in extra_label_deletions(parsed, labels) if not _child_name(item, labels)],
        "annotations": annotations,
    }


def _child_name(deletion: dict, labels: list[dict]) -> bool:
    match = next((item for item in labels if item.get("id") == deletion.get("id")), None)
    return bool(match and (match.get("parent_id") or match.get("has_parent")))
