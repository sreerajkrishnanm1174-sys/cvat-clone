from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProjectState(Base):
    __tablename__ = "projects"

    cvat_project_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization: Mapped[str] = mapped_column(String(64), default="")
    annotation_mode: Mapped[str] = mapped_column(String(32), default="mixed")
    learn_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    compare_with_agent: Mapped[bool] = mapped_column(Boolean, default=False)
    allow_external_models: Mapped[bool] = mapped_column(Boolean, default=False)
    spec_json: Mapped[dict] = mapped_column(JSON, default=dict)
    spec_version: Mapped[int] = mapped_column(Integer, default=0)
    profile_json: Mapped[dict] = mapped_column(JSON, default=dict)
    profile_version: Mapped[int] = mapped_column(Integer, default=0)
    webhook_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished_frames: Mapped[int] = mapped_column(Integer, default=0)


class SpecVersion(Base):
    __tablename__ = "spec_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cvat_project_id: Mapped[int] = mapped_column(Integer, index=True)
    version: Mapped[int] = mapped_column(Integer)
    document_name: Mapped[str] = mapped_column(String(256), default="")
    spec_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProfileVersion(Base):
    __tablename__ = "profile_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cvat_project_id: Mapped[int] = mapped_column(Integer, index=True)
    version: Mapped[int] = mapped_column(Integer)
    profile_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cvat_project_id: Mapped[int] = mapped_column(Integer, index=True)
    task_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    mode: Mapped[str] = mapped_column(String(32), default="auto_first")
    profile_version: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text, default="")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("run_id", "job_id", "frame", name="uq_prediction_frame"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, index=True)
    cvat_project_id: Mapped[int] = mapped_column(Integer, index=True)
    task_id: Mapped[int] = mapped_column(Integer)
    job_id: Mapped[int] = mapped_column(Integer)
    frame: Mapped[int] = mapped_column(Integer)
    profile_version: Mapped[int] = mapped_column(Integer, default=0)
    shapes_json: Mapped[list] = mapped_column(JSON)


class Sample(Base):
    __tablename__ = "samples"
    __table_args__ = (UniqueConstraint("cvat_project_id", "job_id", "frame", "event_key", name="uq_sample"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cvat_project_id: Mapped[int] = mapped_column(Integer, index=True)
    task_id: Mapped[int] = mapped_column(Integer)
    job_id: Mapped[int] = mapped_column(Integer)
    frame: Mapped[int] = mapped_column(Integer)
    event_key: Mapped[str] = mapped_column(String(128))
    items_json: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dedupe_key: Mapped[str] = mapped_column(String(256), unique=True)
    payload_json: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="received")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ClientFormat(Base):
    """A reusable client output profile. Scoped to one CVAT organization."""

    __tablename__ = "client_formats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization: Mapped[str] = mapped_column(String(64), default="", index=True)
    cvat_project_id: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String(160), default="Client format")
    document_name: Mapped[str] = mapped_column(String(256), default="")
    version: Mapped[int] = mapped_column(Integer, default=1)
    profile_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cvat_project_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def make_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, future=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
