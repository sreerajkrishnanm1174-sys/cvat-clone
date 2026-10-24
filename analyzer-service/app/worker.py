from celery import Celery

from .config import get_settings
from .main import learn_finished_jobs
from .cvat import CvatClient
from .db import init_db, make_engine, make_session_factory
from .service import get_project

settings = get_settings()
celery = Celery("requirement_analyzer", broker=settings.redis_url, backend=settings.redis_url)


@celery.task(name="analyzer.sync_all")
def sync_all() -> int:
    engine = make_engine(settings.database_url)
    init_db(engine)
    sessions = make_session_factory(engine)
    if not settings.cvat_access_token:
        return 0
    session = sessions()
    client = CvatClient(
        settings.cvat_base_url,
        settings.cvat_access_token,
        settings.cvat_auth_scheme,
    )
    try:
        from .db import ProjectState

        total = 0
        for state in session.query(ProjectState).all():
            if not state.learn_enabled:
                continue
            total += learn_finished_jobs(session, client, get_project(session, state.cvat_project_id), settings)
        session.commit()
        return total
    finally:
        client.close()
        session.close()


celery.conf.beat_schedule = {
    "sync-finished-jobs": {"task": "analyzer.sync_all", "schedule": 300.0},
}
