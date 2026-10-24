from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .cvat import CvatClient, CvatError

from .db import Prediction, Run, Sample, init_db, make_engine, make_session_factory
from .learning import MODE_TEXT, explain_profile, prompt_lessons
from . import exporting, planning, report, service, spec
from .vision import annotate_image
from . import chat as agent_chat
from .service import get_project


class AnswerIn(BaseModel):
    question_id: str
    answer: str


class ApplyIn(BaseModel):
    confirm_type_changes: bool = False


class LearningSettingsIn(BaseModel):
    annotation_mode: str | None = None
    learn_enabled: bool | None = None
    compare_with_agent: bool | None = None
    allow_external_models: bool | None = None


class AnnotateIn(BaseModel):
    task_id: int
    mode: str | None = None


class RollbackIn(BaseModel):
    version: int = Field(ge=1)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    Path(settings.analyzer_data_dir).mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.database_url)
    init_db(engine)
    sessions = make_session_factory(engine)
    app = FastAPI(title="Requirement Analyzer", version="1.0.0", root_path=settings.analyzer_root_path)
    app.state.settings = settings
    app.state.sessions = sessions
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def db() -> Session:
        session = sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def identity(
        authorization: str | None = Header(default=None),
        x_organization: str | None = Header(default=None),
    ) -> dict:
        if not authorization or " " not in authorization:
            raise HTTPException(status_code=401, detail="CVAT credentials are required")
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() not in {"bearer", "token"} or not token:
            raise HTTPException(status_code=401, detail="CVAT credentials are required")
        return {
            "scheme": "Bearer" if scheme.lower() == "bearer" else "Token",
            "token": token,
            "organization": x_organization or "",
        }

    def cvat_client(who: dict = Depends(identity)) -> CvatClient:
        client = CvatClient(settings.cvat_base_url, who["token"], who["scheme"], who["organization"] or None)
        try:
            yield client
        finally:
            client.close()

    def open_project(project_id: int, client: CvatClient, session: Session, organization: str) -> tuple[dict, object]:
        try:
            project = client.get_project(project_id)
        except CvatError as exc:
            raise HTTPException(status_code=exc.status, detail="CVAT refused this project") from exc
        return project, get_project(session, project_id, organization)

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")

    @app.post("/chat")
    def chat_agent(
        message: str = Form(""),
        history: str = Form("[]"),
        project_id: str = Form(""),
        task_id: str = Form(""),
        files: list[UploadFile] | None = File(default=None),
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        try:
            previous = json.loads(history or "[]")
        except json.JSONDecodeError:
            previous = []
        if not isinstance(previous, list):
            previous = []
        uploaded = []
        for item in files or []:
            uploaded.append((item.filename or "file", item.file.read()))
        bundle = agent_chat.read_upload(uploaded, Path(settings.analyzer_data_dir))
        transcript = [item for item in previous if isinstance(item, dict)]
        context = []
        if str(project_id or "").strip().isdigit():
            context.append(f"Current project id: {int(project_id)}.")
        if str(task_id or "").strip().isdigit():
            context.append(f"Current task id: {int(task_id)}.")
        content = message.strip()
        if context:
            content = (content + "\n" + " ".join(context)).strip()
        transcript.append({"role": "user", "content": content})
        try:
            text = agent_chat.reply(settings, client, session, transcript, bundle)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"reply": text}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict:
        return {"name": "requirement-analyzer", "version": "1.0.0"}

    @app.post("/projects/{project_id}/requirements")
    def upload_requirements(
        project_id: int,
        file: UploadFile = File(...),
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        destination = Path(settings.analyzer_data_dir) / "uploads" / str(project_id)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / Path(file.filename or "requirement.txt").name
        target.write_bytes(file.file.read())
        try:
            text = spec.read_document(target)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        parsed = _with_guide(spec.extract_spec(text), text, settings)
        service.save_spec(session, state, target.name, parsed)
        applied = _apply_spec(client, project_id, parsed)
        return {"spec": parsed, "version": state.spec_version, "applied": applied}

    @app.post("/projects/{project_id}/requirements/analyze")
    def analyze_requirement(
        project_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        folder = Path(settings.analyzer_data_dir) / "uploads" / str(project_id)
        text = ""
        document_name = "requirement"
        if folder.exists():
            files = [path for path in folder.iterdir() if path.is_file()]
            if files:
                latest = max(files, key=lambda path: path.stat().st_mtime)
                document_name = latest.name
                try:
                    text = spec.read_document(latest)
                except ValueError:
                    text = ""
        parsed = spec.extract_spec(text) if text else dict(state.spec_json or {})
        if not parsed.get("labels") and not text:
            raise HTTPException(status_code=400, detail="Upload a requirement before asking the agent to analyze it.")
        parsed = _with_guide(parsed, text, settings)
        service.save_spec(session, state, document_name, parsed)
        applied = _apply_spec(client, project_id, parsed)
        return {"spec": parsed, "version": state.spec_version, "applied": applied}

    @app.get("/projects/{project_id}/brief")
    def project_brief(
        project_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        return report.project_report(client, project_id, state.spec_json or {})

    @app.get("/projects/{project_id}/spec")
    def get_spec(
        project_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        return {"version": state.spec_version, "spec": state.spec_json or {}}

    @app.post("/projects/{project_id}/spec/answers")
    def answer(
        project_id: int,
        body: AnswerIn,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        try:
            updated = spec.apply_answer(state.spec_json or {}, body.question_id, body.answer)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        service.save_spec(session, state, "answer", updated)
        return {"spec": updated, "version": state.spec_version}

    @app.post("/projects/{project_id}/labels/preview")
    def preview_labels(
        project_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        existing = client.get_labels(project_id)
        return spec.diff_labels(state.spec_json or {}, existing)

    @app.post("/projects/{project_id}/labels/apply")
    def apply_labels(
        project_id: int,
        body: ApplyIn,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        existing = client.get_labels(project_id)
        diff = spec.diff_labels(state.spec_json or {}, existing)
        patch = service.label_patch(diff, existing, body.confirm_type_changes)
        if patch:
            try:
                client.patch_project_labels(project_id, patch)
            except CvatError as exc:
                raise HTTPException(status_code=exc.status, detail="CVAT refused the label update") from exc
        return {"applied": len(patch), "diff": spec.diff_labels(state.spec_json or {}, client.get_labels(project_id))}

    @app.get("/projects/{project_id}/plan")
    def plan(
        project_id: int,
        task_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        task = client.get_task(task_id)
        if task.get("project_id") not in {project_id, None}:
            raise HTTPException(status_code=404, detail="That task is not in this project")
        labels = (state.spec_json or {}).get("labels") or [
            {"name": item["name"], "type": item.get("type")} for item in client.get_labels(project_id)
        ]
        meta = client.get_task_meta(task_id)
        jobs = client.get_jobs(task_id=task_id)
        return service.dataset_plan(task, meta, jobs, labels, state.allow_external_models, None)

    @app.get("/projects/{project_id}/learning")
    def learning_view(
        project_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        explained = explain_profile(state.profile_json, state.finished_frames, state.annotation_mode)
        explained["learn_enabled"] = state.learn_enabled
        explained["compare_with_agent"] = state.compare_with_agent
        explained["allow_external_models"] = state.allow_external_models
        explained["profile_version"] = state.profile_version
        explained["rules"] = (state.profile_json or {}).get("rules") or []
        return explained

    @app.post("/projects/{project_id}/learning/settings")
    def learning_settings(
        project_id: int,
        body: LearningSettingsIn,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        if body.annotation_mode:
            if body.annotation_mode not in MODE_TEXT:
                raise HTTPException(status_code=400, detail="Unknown annotation mode")
            state.annotation_mode = body.annotation_mode
        if body.learn_enabled is not None:
            state.learn_enabled = body.learn_enabled
        if body.compare_with_agent is not None:
            state.compare_with_agent = body.compare_with_agent
        if body.allow_external_models is not None:
            state.allow_external_models = body.allow_external_models
        _ensure_webhook(client, state, settings)
        return {
            "annotation_mode": state.annotation_mode,
            "mode_text": MODE_TEXT[state.annotation_mode],
            "learn_enabled": state.learn_enabled,
            "compare_with_agent": state.compare_with_agent,
            "allow_external_models": state.allow_external_models,
            "webhook_id": state.webhook_id,
        }

    @app.post("/projects/{project_id}/learning/reset")
    def reset_learning(
        project_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        session.query(Sample).filter(Sample.cvat_project_id == project_id).delete()
        session.query(Prediction).filter(Prediction.cvat_project_id == project_id).delete()
        state.profile_json = {"rules": []}
        state.profile_version = 0
        state.finished_frames = 0
        return {"forgotten": True, "summary": "The agent has learned from 0 finished frames."}

    @app.post("/projects/{project_id}/learning/rollback")
    def rollback(
        project_id: int,
        body: RollbackIn,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        try:
            profile = service.rollback_profile(session, state, body.version)
        except KeyError:
            raise HTTPException(status_code=404, detail="That profile version does not exist") from None
        return {"profile_version": state.profile_version, "rules": profile.get("rules") or []}

    @app.post("/projects/{project_id}/annotate")
    def annotate(
        project_id: int,
        body: AnnotateIn,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        if body.mode:
            if body.mode not in MODE_TEXT:
                raise HTTPException(status_code=400, detail="Unknown annotation mode")
            state.annotation_mode = body.mode
        run = Run(
            cvat_project_id=project_id,
            task_id=body.task_id,
            status="running",
            mode=state.annotation_mode,
            profile_version=state.profile_version if state.learn_enabled else 0,
        )
        session.add(run)
        session.flush()
        try:
            summary = run_annotation(session, client, state, body.task_id, run, settings)
        except CvatError as exc:
            run.status = "failed"
            run.message = "CVAT refused the annotation write"
            raise HTTPException(status_code=exc.status, detail=run.message) from exc
        return {"run_id": run.id, "status": run.status, "message": run.message, **summary}

    @app.get("/runs/{run_id}")
    def run_status(run_id: int, _: dict = Depends(identity), session: Session = Depends(db)) -> dict:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "id": run.id,
            "status": run.status,
            "progress": run.progress,
            "total": run.total,
            "message": run.message,
        }

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: int, _: dict = Depends(identity), session: Session = Depends(db)) -> dict:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        run.cancel_requested = True
        return {"cancel_requested": True}

    @app.post("/projects/{project_id}/sync")
    def sync_project(
        project_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        learned = learn_finished_jobs(session, client, state, settings)
        return {"samples": learned, "finished_frames": state.finished_frames}

    @app.post("/projects/{project_id}/jobs/{job_id}/learn")
    def learn_saved_job(
        project_id: int,
        job_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        _, state = open_project(project_id, client, session, who["organization"])
        if not state.learn_enabled:
            return {"samples": 0, "skipped": True, "finished_frames": state.finished_frames}
        try:
            job = client.get_job(job_id)
        except CvatError as exc:
            raise HTTPException(status_code=exc.status, detail="CVAT could not read that job") from exc
        if job.get("project_id") != project_id:
            raise HTTPException(status_code=404, detail="That job is not in this project")
        learned = learn_job(session, client, state, job, settings, event_key="user-edit")
        return {
            "samples": learned,
            "skipped": False,
            "finished_frames": state.finished_frames,
            "rules": len((state.profile_json or {}).get("rules") or []),
        }

    @app.get("/projects/{project_id}/export")
    def export_requirement_format(
        project_id: int,
        task_id: int | None = None,
        job_id: int | None = None,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ):
        _, state = open_project(project_id, client, session, who["organization"])
        parsed = state.spec_json or {}
        output_format = parsed.get("output_format") or "CVAT"
        kind = exporting.export_kind(output_format)
        if not kind:
            return {
                "converted": False,
                "output_format": output_format,
                "message": "The requirement does not name an output format other than CVAT.",
            }
        try:
            images = _client_images(client, project_id, task_id, job_id)
        except CvatError as exc:
            raise HTTPException(status_code=exc.status, detail="CVAT could not read the annotations for this export") from exc
        orders = {
            item["name"]: list(item.get("keypoints") or [])
            for item in parsed.get("labels") or []
            if item.get("name")
        }
        for image in images:
            for shape in image["shapes"]:
                if orders.get(shape["label"]):
                    shape["keypoint_order"] = orders[shape["label"]]
        label_names = [item["name"] for item in parsed.get("labels") or [] if item.get("name")]
        blob = exporting.build_archive(kind, output_format, label_names, images)
        filename = f"project-{project_id}-{kind}.zip"
        return Response(
            blob,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/projects/{project_id}/jobs/{job_id}/diff")
    def job_diff(
        project_id: int,
        job_id: int,
        who: dict = Depends(identity),
        client: CvatClient = Depends(cvat_client),
        session: Session = Depends(db),
    ) -> dict:
        open_project(project_id, client, session, who["organization"])
        labels = {item["id"]: item for item in client.get_labels(project_id)}
        annotations = client.get_annotations("jobs", job_id)
        frames = service.group_frames(annotations, labels)
        diff = []
        for frame, shapes in sorted(frames.items()):
            snapshot = service.latest_snapshot(session, project_id, job_id, frame)
            diff.append(
                {
                    "frame": frame,
                    "agent": snapshot,
                    "final": shapes,
                    "items": service.learning.classify_frame(snapshot, shapes),
                }
            )
        return {"frames": diff}

    @app.post("/webhooks/cvat")
    async def cvat_webhook(request: Request, session: Session = Depends(db)) -> dict:
        raw = await request.body()
        _verify_signature(raw, request.headers.get("x-signature-256"), settings.webhook_secret)
        payload = json.loads(raw.decode("utf-8"))
        event = payload.get("event")
        job = payload.get("job") or {}
        dedupe = ":".join(
            [
                str(payload.get("webhook_id") or ""),
                str(event or ""),
                str(job.get("id") or payload.get("task", {}).get("id") or ""),
                str(job.get("updated_date") or ""),
            ]
        )
        if not service.remember_event(session, dedupe, {"event": event, "job_id": job.get("id")}):
            return {"status": "duplicate"}
        if event == "update:job" and job.get("state") == "completed" and job.get("project_id"):
            if not settings.cvat_access_token:
                return {"status": "stored", "learning": "waiting_for_service_token"}
            worker = CvatClient(
                settings.cvat_base_url,
                settings.cvat_access_token,
                settings.cvat_auth_scheme,
                None,
            )
            try:
                state = get_project(session, int(job["project_id"]))
                learned = learn_job(session, worker, state, job, settings)
            finally:
                worker.close()
            return {"status": "learned", "samples": learned}
        return {"status": "ignored"}

    @app.get("/agents")
    def agents() -> dict:
        return {
            "available": "The agent reads the requirement and annotates each task image directly.",
            "cvat_models": "CVAT's YOLO and SAM models are not used for this annotation.",
        }

    app.state.identity = identity
    app.state.cvat_client = cvat_client
    app.state.db = db
    return app


def _verify_signature(raw: bytes, header: str | None, secret: str) -> None:
    if not secret:
        raise HTTPException(status_code=401, detail="Webhook secret is not configured")
    if not header or not header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing webhook signature")
    expected = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    provided = header.split("=", 1)[1]
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


def _ensure_webhook(client: CvatClient, state, settings: Settings) -> None:
    if state.webhook_id or not settings.analyzer_webhook_url or not settings.webhook_secret:
        return
    try:
        created = client.create_webhook(
            state.cvat_project_id,
            settings.analyzer_webhook_url,
            settings.webhook_secret,
            ["update:job", "update:task"],
        )
    except CvatError:
        return
    state.webhook_id = created.get("id")


def _apply_spec(client: CvatClient, project_id: int, parsed: dict) -> int:
    existing = client.get_labels(project_id)
    replace_ids = set(service.skeleton_ids_to_replace(parsed, existing))
    replaced = 0
    if replace_ids:
        client.patch_project_labels(project_id, [{"id": label_id, "deleted": True} for label_id in replace_ids])
        existing = [
            item
            for item in existing
            if item.get("id") not in replace_ids and item.get("parent_id") not in replace_ids
        ]
        replaced = len(replace_ids)
    diff = spec.diff_labels(parsed, existing)
    patch = service.label_patch(diff, existing, False)
    if patch:
        client.patch_project_labels(project_id, patch)
    return replaced + len(patch)


def _with_guide(parsed: dict, text: str, settings: Settings) -> dict:
    from .vision import VisionError, analyze_requirement

    try:
        guide = analyze_requirement(settings.anthropic_api_key, settings.anthropic_model, text, parsed)
    except (VisionError, httpx.HTTPError):
        guide = spec.explain_requirement(parsed)
    return spec.apply_guide(parsed, guide)


def run_annotation(session: Session, client: CvatClient, state, task_id: int, run: Run, settings: Settings, detect=None) -> dict:
    task = client.get_task(task_id)
    if task.get("project_id") != state.cvat_project_id:
        raise HTTPException(status_code=404, detail="That task is not in this project")
    labels = client.get_labels(state.cvat_project_id)
    jobs = client.get_jobs(task_id=task_id)
    if not int(task.get("size") or 0) or not jobs:
        run.status = "finished"
        run.message = "This task has no images yet. Choose the image files, create the task, and wait until CVAT finishes importing them."
        return {"written": 0, "flagged": 0, "message": run.message}
    if state.learn_enabled:
        learn_finished_jobs(session, client, state, settings)
    profile = state.profile_json if state.learn_enabled else {"rules": []}
    lesson_rows = session.scalars(select(Sample).where(Sample.cvat_project_id == state.cvat_project_id)).all()
    lessons = prompt_lessons(profile, [{"items": row.items_json} for row in lesson_rows]) if state.learn_enabled else []
    labels = _annotation_labels(labels)
    spec_by_name = {item["name"]: item for item in (state.spec_json or {}).get("labels") or []}
    spec_labels = []
    for item in labels:
        extra = spec_by_name.get(item["name"]) or {}
        sublabels = item.get("sublabels") or []
        keypoints = [sub.get("name") for sub in sublabels if sub.get("name")] or list(extra.get("keypoints") or [])
        cvat_type = item.get("type")
        spec_labels.append(
            {
                "name": item["name"],
                "type": cvat_type if cvat_type not in {None, "", "any"} else extra.get("type") or "any",
                "attributes": item.get("attributes") or [],
                "detect_as": extra.get("detect_as") or [],
                "min_size": extra.get("min_size"),
                "keypoints": keypoints,
                "edges": list(extra.get("edges") or []),
                "how_to_annotate": extra.get("how_to_annotate") or "",
                "how_to_decide": extra.get("how_to_decide") or "",
            }
        )
    if not spec_labels:
        run.status = "finished"
        run.message = "This project has no labels yet. Read the requirement and apply the labels first."
        return {"written": 0, "flagged": 0, "message": run.message}
    written = 0
    flagged = 0
    manual_frames: set[int] = set()
    run.total = sum(int(job.get("stop_frame", 0)) - int(job.get("start_frame", 0)) + 1 for job in jobs)
    video = task.get("mode") == "interpolation"
    for job in jobs:
        if run.cancel_requested:
            run.status = "cancelled"
            return {"written": written, "flagged": flagged}
        start = int(job.get("start_frame") or 0)
        stop = int(job.get("stop_frame") or start)
        existing = client.get_annotations("jobs", job["id"])
        manual_frames = {
            int(shape.get("frame") or 0)
            for shape in existing.get("shapes") or []
            if shape.get("source") == "manual"
        }
        per_frame: dict[int, list[dict]] = {}
        for frame in range(start, stop + 1):
            if run.cancel_requested:
                run.status = "cancelled"
                return {"written": written, "flagged": flagged}
            run.progress += 1
            if frame in manual_frames:
                continue
            raw = _detect(client, settings, task_id, frame, spec_labels, detect, lessons)
            adapted = service.shapes_for_labels(raw, spec_labels, profile)
            per_frame[frame] = adapted
        if video:
            per_frame = service.associate_by_iou(per_frame)
        cvat_shapes = []
        for frame, shapes in per_frame.items():
            session.add(
                Prediction(
                    run_id=run.id,
                    cvat_project_id=state.cvat_project_id,
                    task_id=task_id,
                    job_id=job["id"],
                    frame=frame,
                    profile_version=run.profile_version,
                    shapes_json=shapes,
                )
            )
            for shape in shapes:
                payload = service.to_cvat_shape(shape, {item["name"]: item for item in labels}, frame)
                if payload:
                    cvat_shapes.append(payload)
                    written += 1
                reasons = service.review_reasons(
                    shape,
                    profile,
                    next((item for item in (state.spec_json or {}).get("labels") or [] if item["name"] == shape["label"]), None),
                    weak_track=bool(shape.get("weak_track")),
                )
                if reasons:
                    points = shape.get("points") or [0, 0]
                    position = [float(points[0]), float(points[1])]
                    client.create_issue(
                        job["id"],
                        frame,
                        "Review first: " + "; ".join(reasons),
                        position,
                    )
                    flagged += 1
        kept = [
            shape
            for shape in existing.get("shapes") or []
            if int(shape.get("frame") or 0) in manual_frames
        ]
        if cvat_shapes or kept != (existing.get("shapes") or []):
            client.put_annotations(
                "jobs",
                job["id"],
                {
                    "version": existing.get("version") or 0,
                    "tags": existing.get("tags") or [],
                    "shapes": kept + cvat_shapes,
                    "tracks": existing.get("tracks") or [],
                },
            )
    run.status = "finished"
    if written == 0 and not manual_frames:
        detail = _detect_error or _empty_annotation_reason(spec_labels)
        run.message = "No shapes were added. " + detail
    else:
        learned = (
            f" Applied {len(lessons)} correction(s) learned from this project."
            if lessons
            else " No saved corrections were found yet, so this pass follows the requirement."
        )
        run.message = (
            f"Added {written} shapes and flagged {flagged} for review.{learned} "
            "Save your edits in this project and the next task follows them. "
            "Frames you already edited were left as you drew them."
        )
    return {"written": written, "flagged": flagged, "message": run.message}


_detect_error = ""


def _annotation_labels(labels: list[dict]) -> list[dict]:
    """Keep project labels and attach skeleton joints. Joints are not separate labels."""
    children: dict[int, list[dict]] = {}
    for item in labels:
        parent = item.get("parent_id")
        if parent:
            children.setdefault(int(parent), []).append(item)
    prepared = []
    for item in labels:
        if item.get("parent_id") or item.get("has_parent"):
            continue
        copy = dict(item)
        if not copy.get("sublabels"):
            copy["sublabels"] = children.get(int(item["id"]), []) if item.get("id") is not None else []
        prepared.append(copy)
    return prepared


def _empty_annotation_reason(labels: list[dict]) -> str:
    skeletons = [item["name"] for item in labels if item.get("type") == "skeleton"]
    if skeletons:
        joints = []
        for item in labels:
            if item.get("type") == "skeleton":
                joints.extend(item.get("keypoints") or [])
        if joints:
            return "The skeleton keypoints were not found on the image: " + ", ".join(joints[:12]) + "."
        return "This project expects a skeleton, but the label has no keypoints. Analyze the requirement again."
    return "The vision agent did not find any labels that belong to this project."


def _detect(client, settings: Settings, task_id: int, frame: int, labels: list[dict], detect, lessons: list[str] | None = None) -> list[dict]:
    global _detect_error
    if detect is not None:
        return detect(frame, labels)
    if not settings.anthropic_api_key:
        _detect_error = "The agent annotates the task images directly and needs an API key."
        return []
    image = client.get_frame(task_id, frame)
    try:
        raw = annotate_image(settings.anthropic_api_key, settings.anthropic_model, image, labels, lessons or [])
    except Exception as exc:
        _detect_error = str(exc)[:240]
        return []
    _detect_error = ""
    return raw


def _client_images(client: CvatClient, project_id: int, task_id: int | None, job_id: int | None) -> list[dict]:
    labels = {int(item["id"]): item for item in client.get_labels(project_id) if item.get("id") is not None}
    if job_id:
        job = client.get_job(int(job_id))
        if job.get("project_id") != project_id:
            raise HTTPException(status_code=404, detail="That job is not in this project")
        jobs = [job]
    elif task_id:
        task = client.get_task(int(task_id))
        if task.get("project_id") != project_id:
            raise HTTPException(status_code=404, detail="That task is not in this project")
        jobs = client.get_jobs(task_id=int(task_id))
    else:
        jobs = []
        for task in client.list_project_tasks(project_id):
            jobs.extend(client.get_jobs(task_id=int(task["id"])))
    images = []
    used_names: set[str] = set()
    meta_cache: dict[int, dict] = {}
    for job in jobs:
        task_key = int(job.get("task_id") or 0)
        if task_key not in meta_cache:
            meta_cache[task_key] = client.get_task_meta(task_key)
        frames = meta_cache[task_key].get("frames") or []
        grouped: dict[int, list[dict]] = {}
        annotations = client.get_annotations("jobs", int(job["id"]))
        for shape in annotations.get("shapes") or []:
            grouped.setdefault(int(shape.get("frame") or 0), []).append(shape)
        for track in annotations.get("tracks") or []:
            for shape in track.get("shapes") or []:
                if shape.get("outside"):
                    continue
                copy = dict(shape)
                copy["label_id"] = track.get("label_id")
                grouped.setdefault(int(shape.get("frame") or 0), []).append(copy)
        start = int(job.get("start_frame") or 0)
        stop = int(job.get("stop_frame") or start)
        for frame in range(start, stop + 1):
            info = frames[frame] if 0 <= frame < len(frames) else {}
            name = str(info.get("name") or f"task_{task_key}_frame_{frame}.jpg")
            if name in used_names:
                name = f"task_{task_key}_{name}"
            used_names.add(name)
            images.append(
                {
                    "name": name,
                    "width": int(info.get("width") or 0),
                    "height": int(info.get("height") or 0),
                    "shapes": [_export_shape(shape, labels) for shape in grouped.get(frame, [])],
                }
            )
    return images


def _export_shape(shape: dict, labels: dict[int, dict]) -> dict:
    label = labels.get(int(shape.get("label_id") or 0), {})
    elements = []
    for element in shape.get("elements") or []:
        joint = labels.get(int(element.get("label_id") or 0), {})
        elements.append(
            {
                "name": joint.get("name") or "",
                "points": element.get("points") or [],
                "outside": bool(element.get("outside")),
                "occluded": bool(element.get("occluded")),
            }
        )
    return {
        "label": label.get("name") or "",
        "type": shape.get("type") or label.get("type"),
        "points": shape.get("points") or [],
        "elements": elements,
        "keypoint_order": [],
    }


def learn_job(session: Session, client: CvatClient, state, job: dict, settings: Settings, event_key: str | None = None) -> int:
    if not state.learn_enabled:
        return 0
    labels = {item["id"]: item for item in client.get_labels(state.cvat_project_id)}
    annotations = client.get_annotations("jobs", job["id"])
    frames = service.group_frames(annotations, labels)
    event_key = event_key or str(job.get("updated_date") or job.get("id"))
    count = 0
    for frame, shapes in frames.items():
        snapshot = service.latest_snapshot(session, state.cvat_project_id, job["id"], frame)
        if state.compare_with_agent and not snapshot:
            snapshot = _quiet_comparison(client, settings, int(job.get("task_id") or 0), int(frame), list(labels.values()))
        service.record_frame(
            session,
            state,
            task_id=int(job.get("task_id") or 0),
            job_id=int(job["id"]),
            frame=int(frame),
            event_key=event_key,
            snapshot=snapshot,
            final=shapes,
        )
        count += 1
    if not frames:
        service.record_frame(
            session,
            state,
            task_id=int(job.get("task_id") or 0),
            job_id=int(job["id"]),
            frame=int(job.get("start_frame") or 0),
            event_key=event_key,
            snapshot=[],
            final=[],
        )
    return count


def learn_finished_jobs(session: Session, client: CvatClient, state, settings: Settings) -> int:
    jobs = client.get_jobs(project_id=state.cvat_project_id)
    total = 0
    for job in jobs:
        total += learn_job(session, client, state, job, settings, event_key="user-edit")
    return total


def _quiet_comparison(client: CvatClient, settings: Settings, task_id: int, frame: int, labels: list[dict]) -> list[dict]:
    """Learning uses the shapes already stored from the direct annotation pass."""
    return []


app = create_app()
