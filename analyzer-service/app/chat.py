"""Dashboard chat agent. It uses the signed-in user's CVAT token and never calls CVAT's models."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx

from . import report, service, spec
from .config import Settings
from .cvat import CvatClient, CvatError
from .db import Run
from .service import get_project

TOOLS = [
    {
        "name": "list_projects",
        "description": "List the CVAT projects this user can see.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_project",
        "description": "Create a CVAT project for this user.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List tasks. Pass project_id to limit the list to one project.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}},
        },
    },
    {
        "name": "create_task",
        "description": "Create a task. If the user attached images in this message, they are uploaded into the task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "project_id": {"type": "integer"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "apply_requirement",
        "description": "Save the requirement document attached to this message onto a project and create its labels.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}},
            "required": ["project_id"],
        },
    },
    {
        "name": "annotate_task",
        "description": "Annotate every image in a task from the project requirement. This can take several minutes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "integer"},
                "task_id": {"type": "integer"},
            },
            "required": ["project_id", "task_id"],
        },
    },
    {
        "name": "project_details",
        "description": "Read one project's requirement steps, labels, and the annotations currently on its tasks.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}},
            "required": ["project_id"],
        },
    },
    {
        "name": "fix_labels",
        "description": "Remove duplicate labels and labels that are not in the requirement, then apply the requirement labels.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}},
            "required": ["project_id"],
        },
    },
    {
        "name": "update_requirement",
        "description": "Change the saved requirement and the CVAT labels. Use this when the user wants labels removed, kept, renamed, retyped, or added, even if those labels are already in the requirement.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "integer"},
                "remove_labels": {"type": "array", "items": {"type": "string"}},
                "keep_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "If set, every other label is removed from the requirement and from CVAT.",
                },
                "rename": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                        "required": ["from", "to"],
                    },
                },
                "add_labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                        "required": ["name", "type"],
                    },
                },
                "type_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                        "required": ["name", "type"],
                    },
                },
            },
            "required": ["project_id"],
        },
    },
    {
        "name": "analyze_requirement",
        "description": "Read the project's saved requirement again and apply its labels, skeleton joints, and bones.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}},
            "required": ["project_id"],
        },
    },
    {
        "name": "rename_project",
        "description": "Rename a CVAT project.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}, "name": {"type": "string"}},
            "required": ["project_id", "name"],
        },
    },
    {
        "name": "delete_project",
        "description": "Delete a CVAT project. Use only when the user asked to delete that project.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}},
            "required": ["project_id"],
        },
    },
    {
        "name": "rename_task",
        "description": "Rename a task in a project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "integer"},
                "task_id": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["project_id", "task_id", "name"],
        },
    },
    {
        "name": "delete_task",
        "description": "Delete a task. Use only when the user asked to delete that task.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}, "task_id": {"type": "integer"}},
            "required": ["project_id", "task_id"],
        },
    },
    {
        "name": "add_images",
        "description": "Add the image files attached to this message after the images already in the task. Existing annotations stay on the current images.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}, "task_id": {"type": "integer"}},
            "required": ["project_id", "task_id"],
        },
    },
    {
        "name": "replace_images",
        "description": "Replace the task images with the image files attached to this message. The previous images and their annotations are removed.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}, "task_id": {"type": "integer"}},
            "required": ["project_id", "task_id"],
        },
    },
    {
        "name": "learn_corrections",
        "description": "Learn from the annotation corrections already saved in this project so the next task uses them.",
        "input_schema": {
            "type": "object",
            "properties": {"project_id": {"type": "integer"}},
            "required": ["project_id"],
        },
    },
    {
        "name": "clear_annotations",
        "description": "Remove the shapes on one task so they can be annotated again. Use this when the user says the annotations are wrong.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "integer"},
                "task_id": {"type": "integer"},
            },
            "required": ["project_id", "task_id"],
        },
    },
]

AGENT_SYSTEM = """You are the Requirement Analyzer for the CVAT project the user has open.
Use the tools. Do not invent ids. CVAT's YOLO and SAM models are not used.
The user message can include the current project id and task id. Use those when the user does not name another project or task.
You can create, rename, and delete projects and tasks. You can change the requirement, add or remove labels, add or replace images, clear annotations, annotate, and learn from corrections.
When the user asks about the project, call project_details and answer with the requirement steps and what is annotated.
When the user says something is wrong, or asks to change labels or the requirement, do it in this turn:
- labels that are not in the requirement, or duplicates: call fix_labels
- remove, keep, rename, or add labels, even when they are already in the requirement: call update_requirement, which changes the saved requirement and CVAT
- wrong shapes on a task: call clear_annotations, then annotate_task
- a new requirement file is attached: call apply_requirement
- image files are attached: call add_images on the current task, unless the user asked to replace the images or create a new task
Do the change. Do not tell the user to upload a revised document or to click another button.
Delete a project or task only when the user asked to delete it.
Reply in short plain sentences. Do not use markdown, asterisks, or bullet characters.
After you create or replace a project or task, include its id.
"""


def run_tool(name: str, arguments: dict, client: CvatClient, session, settings: Settings, upload: dict) -> str:
    try:
        if name == "list_projects":
            rows = client.list_projects()
            return json.dumps([{"id": row.get("id"), "name": row.get("name")} for row in rows])
        if name == "create_project":
            created = client.create_project(str(arguments.get("name") or "").strip())
            return json.dumps({"id": created.get("id"), "name": created.get("name"), "path": f"/projects/{created.get('id')}"})
        if name == "list_tasks":
            project_id = arguments.get("project_id")
            rows = client.list_tasks(int(project_id) if project_id else None)
            return json.dumps([
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "project_id": row.get("project_id"),
                    "size": row.get("size"),
                }
                for row in rows
            ])
        if name == "create_task":
            project_id = arguments.get("project_id")
            created = client.create_task(
                str(arguments.get("name") or "").strip(),
                int(project_id) if project_id else None,
            )
            images = upload.get("images") or []
            if images:
                client.upload_task_images(int(created["id"]), images)
            return json.dumps({
                "id": created.get("id"),
                "name": created.get("name"),
                "project_id": created.get("project_id"),
                "images": len(images),
                "path": f"/tasks/{created.get('id')}",
            })
        if name == "apply_requirement":
            parsed = upload.get("spec")
            if not parsed:
                return "No requirement document was attached to this message."
            project_id = int(arguments["project_id"])
            state = get_project(session, project_id)
            from .main import _with_guide

            parsed = _with_guide(parsed, upload.get("text") or "", settings)
            service.save_spec(session, state, upload.get("document_name") or "requirement", parsed)
            labels = client.get_labels(project_id)
            diff = spec.diff_labels(parsed, labels)
            patch = service.label_patch(diff, labels, False)
            if patch:
                client.patch_project_labels(project_id, patch)
            return json.dumps({
                "project_id": project_id,
                "applied": len(patch),
                "labels": [item["name"] for item in parsed.get("labels") or []],
                "questions": parsed.get("questions") or [],
            })
        if name == "project_details":
            project_id = int(arguments["project_id"])
            state = get_project(session, project_id)
            return json.dumps(report.project_report(client, project_id, state.spec_json or {}))
        if name == "fix_labels":
            project_id = int(arguments["project_id"])
            state = get_project(session, project_id)
            parsed = state.spec_json or {}
            if not parsed.get("labels"):
                return "This project has no requirement yet. Attach the requirement document first."
            existing = client.get_labels(project_id)
            removed = report.extra_label_deletions(parsed, existing)
            removed_ids = {item["id"] for item in removed}
            kept = [item for item in existing if item.get("id") not in removed_ids]
            diff = spec.diff_labels(parsed, kept)
            patch = [{key: value for key, value in item.items() if key != "name"} for item in removed]
            patch.extend(service.label_patch(diff, kept, True))
            if patch:
                client.patch_project_labels(project_id, patch)
            return json.dumps({
                "project_id": project_id,
                "removed": [item["name"] for item in removed if item.get("name")],
                "applied": len(patch),
            })
        if name == "clear_annotations":
            project_id = int(arguments["project_id"])
            task_id = int(arguments["task_id"])
            task = client.get_task(task_id)
            if task.get("project_id") != project_id:
                return "That task is not in this project."
            cleared = 0
            for job in client.get_jobs(task_id=task_id):
                existing = client.get_annotations("jobs", int(job["id"]))
                cleared += len(existing.get("shapes") or [])
                client.put_annotations(
                    "jobs",
                    int(job["id"]),
                    {
                        "version": existing.get("version") or 0,
                        "tags": existing.get("tags") or [],
                        "shapes": [],
                        "tracks": [],
                    },
                )
            return json.dumps({"task_id": task_id, "cleared_shapes": cleared})
        if name == "update_requirement":
            project_id = int(arguments["project_id"])
            state = get_project(session, project_id)
            updated = revise_spec(
                state.spec_json or {},
                remove=arguments.get("remove_labels"),
                keep=arguments.get("keep_labels"),
                rename=arguments.get("rename"),
                add=arguments.get("add_labels"),
                type_changes=arguments.get("type_changes"),
            )
            if not (state.spec_json or {}).get("labels") and not updated.get("labels"):
                return "This project has no requirement yet. Attach the requirement document first."
            _rename_cvat_labels(client, project_id, arguments.get("rename") or [])
            service.save_spec(session, state, "requirement", updated)
            applied = _sync_labels(client, project_id, updated)
            return json.dumps({
                "project_id": project_id,
                "labels": [item.get("name") for item in updated.get("labels") or []],
                "applied": applied,
            })
        if name == "analyze_requirement":
            project_id = int(arguments["project_id"])
            state = get_project(session, project_id)
            text, document_name = _saved_requirement_text(settings, project_id)
            parsed = spec.extract_spec(text) if text else dict(state.spec_json or {})
            if not parsed.get("labels") and not text:
                return "This project has no requirement yet. Attach the requirement document first."
            from .main import _with_guide

            parsed = _with_guide(parsed, text, settings)
            service.save_spec(session, state, document_name, parsed)
            applied = _sync_labels(client, project_id, parsed)
            return json.dumps({
                "project_id": project_id,
                "applied": applied,
                "labels": [item.get("name") for item in parsed.get("labels") or []],
            })
        if name == "rename_project":
            project_id = int(arguments["project_id"])
            name = str(arguments.get("name") or "").strip()
            if not name:
                return "A project name is required."
            renamed = client.patch_project(project_id, {"name": name})
            return json.dumps({"id": renamed.get("id"), "name": renamed.get("name")})
        if name == "delete_project":
            project_id = int(arguments["project_id"])
            client.delete_project(project_id)
            return json.dumps({"deleted_project_id": project_id})
        if name == "rename_task":
            project_id = int(arguments["project_id"])
            task_id = int(arguments["task_id"])
            _task_in_project(client, project_id, task_id)
            renamed = client.patch_task(task_id, {"name": str(arguments.get("name") or "").strip()})
            return json.dumps({"id": renamed.get("id"), "name": renamed.get("name"), "project_id": project_id})
        if name == "delete_task":
            project_id = int(arguments["project_id"])
            task_id = int(arguments["task_id"])
            _task_in_project(client, project_id, task_id)
            client.delete_task(task_id)
            return json.dumps({"deleted_task_id": task_id, "project_id": project_id})
        if name in {"add_images", "replace_images"}:
            project_id = int(arguments["project_id"])
            task_id = int(arguments["task_id"])
            images = upload.get("images") or []
            if not images:
                return "Attach the image files in this message, then ask again."
            updated = _change_task_images(client, project_id, task_id, images, name == "add_images")
            return json.dumps(updated)
        if name == "learn_corrections":
            project_id = int(arguments["project_id"])
            state = get_project(session, project_id)
            state.learn_enabled = True
            from .main import learn_finished_jobs

            learned = learn_finished_jobs(session, client, state, settings)
            return json.dumps({
                "project_id": project_id,
                "samples": learned,
                "rules": len((state.profile_json or {}).get("rules") or []),
            })
        if name == "annotate_task":
            project_id = int(arguments["project_id"])
            task_id = int(arguments["task_id"])
            state = get_project(session, project_id)
            state.annotation_mode = "auto_first"
            state.learn_enabled = True
            from .main import run_annotation

            run = Run(cvat_project_id=project_id, task_id=task_id, status="running", mode="auto_first")
            session.add(run)
            session.flush()
            summary = run_annotation(session, client, state, task_id, run, settings)
            return json.dumps({"task_id": task_id, "written": summary.get("written"), "message": summary.get("message")})
    except (CvatError, KeyError, ValueError, TypeError) as exc:
        return f"That action failed: {exc}"[:400]
    return "Unknown action."


def reply(settings: Settings, client: CvatClient, session, messages: list[dict], upload: dict) -> str:
    if not settings.anthropic_api_key:
        raise RuntimeError("The agent needs an API key before it can chat.")
    conversation = []
    for message in messages[-20:]:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            conversation.append({"role": role, "content": content})
    if not conversation or conversation[-1]["role"] != "user":
        raise RuntimeError("Send a message to the agent.")
    note = _upload_note(upload)
    if note:
        conversation[-1]["content"] += "\n" + note
    models = []
    for name in (settings.anthropic_model, "claude-sonnet-5-5", "claude-sonnet-5"):
        if name and name not in models:
            models.append(name)
    last_error = "The agent did not respond."
    for model in models:
        text, error = _complete(settings.anthropic_api_key, model, conversation, client, session, settings, upload)
        if text:
            return text
        last_error = error
        if "not available" not in error:
            break
    raise RuntimeError(last_error)


def _plain(text: str) -> str:
    cleaned = text.replace("**", "").replace("`", "")
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            stripped = stripped[2:]
        lines.append(stripped)
    return "\n".join(line for line in lines if line).strip()


def _upload_note(upload: dict) -> str:
    parts = []
    if upload.get("spec"):
        labels = ", ".join(item["name"] for item in upload["spec"].get("labels") or []) or "no labels yet"
        parts.append(f"A requirement document named {upload.get('document_name')} is attached. Labels found: {labels}.")
    images = upload.get("images") or []
    if images:
        parts.append(
            f"{len(images)} image file(s) are attached. Add them to the current task, replace that task's images, or create a task with them."
        )
    return " ".join(parts)


def _complete(api_key, model, conversation, client, session, settings, upload) -> tuple[str, str]:
    working = list(conversation)
    for _ in range(12):
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1500,
                "system": AGENT_SYSTEM,
                "tools": TOOLS,
                "messages": working,
            },
            timeout=180,
        )
        if response.status_code == 404:
            return "", f"The model {model} is not available."
        if response.status_code >= 400:
            return "", f"The agent returned HTTP {response.status_code}."
        data = response.json()
        blocks = data.get("content") or []
        working.append({"role": "assistant", "content": blocks})
        tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
        if not tool_uses:
            text = "".join(block.get("text") or "" for block in blocks if block.get("type") == "text").strip()
            return _plain(text) or "Done.", ""
        results = []
        for block in tool_uses:
            output = run_tool(block.get("name") or "", block.get("input") or {}, client, session, settings, upload)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.get("id"),
                "content": output,
            })
        working.append({"role": "user", "content": results})
    return "I stopped after several actions. Tell me the next step.", ""


def read_upload(files: list[tuple[str, bytes]], data_dir: Path) -> dict:
    upload = {"images": [], "spec": None, "document_name": ""}
    for name, content in files:
        suffix = Path(name).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".zip"}:
            upload["images"].append((Path(name).name, content))
            continue
        target = data_dir / "chat" / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        try:
            upload["text"] = spec.read_document(target)
            upload["spec"] = spec.extract_spec(upload["text"])
            upload["document_name"] = Path(name).name
        except ValueError:
            continue
    return upload


def _label_key(name: str) -> str:
    return str(name or "").strip().casefold()


def revise_spec(
    parsed: dict | None,
    remove: list | None = None,
    keep: list | None = None,
    rename: list | None = None,
    add: list | None = None,
    type_changes: list | None = None,
) -> dict:
    """Apply a chat instruction to the saved requirement labels."""
    updated = copy.deepcopy(parsed or {})
    labels = list(updated.get("labels") or [])
    for item in rename or []:
        source = _label_key(item.get("from"))
        target = str(item.get("to") or "").strip()
        if not source or not target:
            continue
        for label in labels:
            if _label_key(label.get("name")) == source:
                label["name"] = target
    for item in type_changes or []:
        for label in labels:
            if _label_key(label.get("name")) == _label_key(item.get("name")) and item.get("type"):
                label["type"] = item["type"]
    if keep:
        wanted = {_label_key(name) for name in keep}
        labels = [label for label in labels if _label_key(label.get("name")) in wanted]
    if remove:
        blocked = {_label_key(name) for name in remove}
        labels = [label for label in labels if _label_key(label.get("name")) not in blocked]
    names = {_label_key(label.get("name")) for label in labels}
    for item in add or []:
        name = str(item.get("name") or "").strip()
        if not name or _label_key(name) in names:
            continue
        labels.append({"name": name, "type": item.get("type") or "rectangle", "attributes": []})
        names.add(_label_key(name))
    updated["labels"] = labels
    return updated


def _rename_cvat_labels(client: CvatClient, project_id: int, rename: list) -> None:
    if not rename:
        return
    existing = client.get_labels(project_id)
    patch = []
    for item in rename:
        source = _label_key(item.get("from"))
        target = str(item.get("to") or "").strip()
        if not source or not target:
            continue
        for label in existing:
            if label.get("parent_id") or label.get("has_parent"):
                continue
            if _label_key(label.get("name")) == source and label.get("id") is not None:
                patch.append({"id": label["id"], "name": target})
    if patch:
        client.patch_project_labels(project_id, patch)


def _sync_labels(client: CvatClient, project_id: int, parsed: dict) -> int:
    existing = client.get_labels(project_id)
    replace_ids = set(service.skeleton_ids_to_replace(parsed, existing))
    removed_ids = {item["id"] for item in report.extra_label_deletions(parsed, existing)}
    removed_ids.update(replace_ids)
    for item in existing:
        if item.get("parent_id") in replace_ids and item.get("id") is not None:
            removed_ids.add(int(item["id"]))
    kept = [item for item in existing if item.get("id") not in removed_ids]
    patch = [{"id": label_id, "deleted": True} for label_id in removed_ids]
    patch.extend(service.label_patch(spec.diff_labels(parsed, kept), kept, True))
    if patch:
        client.patch_project_labels(project_id, patch)
    return len(patch)


def _task_in_project(client: CvatClient, project_id: int, task_id: int) -> dict:
    task = client.get_task(task_id)
    if task.get("project_id") != project_id:
        raise ValueError("That task is not in this project.")
    return task


def _saved_requirement_text(settings: Settings, project_id: int) -> tuple[str, str]:
    folder = Path(settings.analyzer_data_dir) / "uploads" / str(project_id)
    if not folder.exists():
        return "", "requirement"
    files = [path for path in folder.iterdir() if path.is_file()]
    if not files:
        return "", "requirement"
    latest = max(files, key=lambda path: path.stat().st_mtime)
    try:
        return spec.read_document(latest), latest.name
    except ValueError:
        return "", latest.name


def _unique_name(name: str, used: set[str]) -> str:
    clean = Path(str(name or "image.jpg")).name or "image.jpg"
    if clean not in used:
        used.add(clean)
        return clean
    stem = Path(clean).stem
    ext = Path(clean).suffix
    index = 2
    while True:
        candidate = f"{stem}_{index}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def _strip_ids(value):
    if isinstance(value, list):
        return [_strip_ids(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_ids(item) for key, item in value.items() if key != "id"}
    return value


def _remap_annotations(annotations: dict, old_names: list[str], frames: list[dict]) -> dict:
    by_name = {}
    for index, frame in enumerate(frames or []):
        name = Path(str((frame or {}).get("name") or "")).name
        if name:
            by_name[name] = index
    payload = _strip_ids(annotations or {})

    def map_frame(frame):
        try:
            name = old_names[int(frame)]
        except (TypeError, ValueError, IndexError):
            return frame
        return by_name.get(name, frame)

    for shape in payload.get("shapes") or []:
        shape["frame"] = map_frame(shape.get("frame"))
    for tag in payload.get("tags") or []:
        tag["frame"] = map_frame(tag.get("frame"))
    for track in payload.get("tracks") or []:
        for shape in track.get("shapes") or []:
            shape["frame"] = map_frame(shape.get("frame"))
    payload["version"] = 0
    return payload


def _read_task_images(client: CvatClient, task: dict) -> tuple[list[tuple[str, bytes]], list[str]]:
    meta = client.get_task_meta(int(task["id"]))
    frames = meta.get("frames") or []
    files = []
    names = []
    used: set[str] = set()
    for index in range(int(task.get("size") or 0)):
        frame = frames[index] if index < len(frames) else {}
        name = _unique_name(frame.get("name") or f"frame_{index}.jpg", used)
        files.append((name, client.get_frame(int(task["id"]), index)))
        names.append(name)
    return files, names


def _change_task_images(client: CvatClient, project_id: int, task_id: int, images: list, adding: bool) -> dict:
    task = _task_in_project(client, project_id, task_id)
    if task.get("mode") == "interpolation":
        raise ValueError("Add and replace work on image tasks. This task is a video.")
    if not int(task.get("size") or 0):
        client.upload_task_images(task_id, images)
        updated = client.get_task(task_id)
        return {"task_id": task_id, "size": updated.get("size"), "replaced_task": False}
    annotations = None
    current_files: list[tuple[str, bytes]] = []
    old_names: list[str] = []
    if adding:
        annotations = client.get_annotations("tasks", task_id)
        current_files, old_names = _read_task_images(client, task)
    used = set(old_names)
    incoming = [(_unique_name(name, used), content) for name, content in images]
    created = client.create_task(task.get("name") or "task", project_id)
    new_id = int(created["id"])
    try:
        client.upload_task_images(new_id, current_files + incoming)
        has_annotations = bool(
            annotations and (annotations.get("shapes") or annotations.get("tracks") or annotations.get("tags"))
        )
        if has_annotations:
            meta = client.get_task_meta(new_id)
            client.put_annotations("tasks", new_id, _remap_annotations(annotations, old_names, meta.get("frames") or []))
    except Exception:
        try:
            client.delete_task(new_id)
        except CvatError:
            pass
        raise
    client.delete_task(task_id)
    updated = client.get_task(new_id)
    return {
        "task_id": new_id,
        "previous_task_id": task_id,
        "size": updated.get("size"),
        "replaced_task": True,
        "added": adding,
    }
