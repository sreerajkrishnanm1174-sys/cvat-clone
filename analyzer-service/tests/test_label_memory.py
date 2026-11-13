"""No requirement document: labels come from CVAT, the agent learns from manual annotation."""

from io import BytesIO

from PIL import Image

from app import labelmemory
from app.db import LabelExemplar, Run
from app.main import run_annotation
from app.service import get_project

from tests.test_analyzer import FakeCvat, _client

HEADERS = {"Authorization": "Bearer user"}


def _jpeg(color=(200, 40, 40)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (200, 160), color).save(buffer, format="JPEG")
    return buffer.getvalue()


class PhotoCvat(FakeCvat):
    """FakeCvat with real image bytes, project labels only (no requirement), and a frame-fetch counter."""

    def __init__(self):
        super().__init__()
        self.labels = [
            {"id": 1, "name": "pothole", "type": "any", "attributes": []},
            {"id": 2, "name": "crack", "type": "rectangle", "attributes": []},
        ]
        self.frame_fetches = 0

    def get_annotations(self, kind, resource_id):
        if resource_id == 3:
            return self.annotations
        return {"shapes": [], "tracks": [], "tags": []}

    def list_tasks(self, project_id=None):
        return [self.get_task(7)]

    def add_new_job(self):
        self.jobs.append({**self.jobs[0], "id": 4, "updated_date": "2026-01-02T00:00:00Z"})

    def get_frame(self, task_id, frame):
        self.frame_fetches += 1
        return _jpeg()


def _hand_drawn(count: int) -> dict:
    shapes = []
    for index in range(count):
        shapes.append(
            {
                "id": 100 + index,
                "frame": index % 2,
                "label_id": 1,
                "type": "polygon",
                "points": [20, 20, 90 + index, 25, 80, 70, 25, 60],
                "source": "manual",
                "attributes": [],
            }
        )
    shapes.append(
        {"id": 200, "frame": 0, "label_id": 2, "type": "rectangle", "points": [5, 5, 60, 30], "source": "manual", "attributes": []}
    )
    return {"shapes": shapes, "tracks": [], "tags": []}


def _setup(tmp_path):
    client, _ = _client(tmp_path)
    fake = PhotoCvat()

    def cvat_client():
        yield fake

    client.app.dependency_overrides[client.app.state.cvat_client] = cvat_client
    return client, fake


def test_learns_from_manual_work_without_a_requirement_and_applies_it(tmp_path, monkeypatch):
    client, fake = _setup(tmp_path)
    fake.annotations = _hand_drawn(4)

    # Learning is triggered by a sync. No requirement was ever uploaded.
    synced = client.post("/projects/1/sync", headers=HEADERS)
    assert synced.status_code == 200, synced.text

    report = client.get("/projects/1/label-memory", headers=HEADERS).json()
    assert report["label_source"] == "CVAT project labels"
    pothole = next(item for item in report["labels"] if item["label"] == "pothole")
    assert pothole["examples_kept"] == 4
    assert pothole["draws_as"] == "polygon"
    assert "annotators draw 'pothole' as a polygon" in pothole["note"]
    crack = next(item for item in report["labels"] if item["label"] == "crack")
    assert crack["examples_kept"] == 1
    assert crack["note"] is None  # one drawing is remembered, not yet a habit

    # The next set of data: the vision agent receives the learned crops and notes.
    seen = {}

    def fake_annotate(api_key, model, image, labels, lessons, refine=True, exemplars=None):
        seen["labels"] = labels
        seen["exemplars"] = exemplars
        return [{"label": "crack", "type": "rectangle", "points": [5, 5, 60, 30], "score": 0.9, "attributes": {}}]

    monkeypatch.setattr("app.main.annotate_image", fake_annotate)
    settings = client.app.state.settings.model_copy(update={"anthropic_api_key": "test-key"})
    fake.add_new_job()  # the next set of data: a new job with nothing drawn yet
    session = client.app.state.sessions()
    state = get_project(session, 1)
    run = Run(cvat_project_id=1, task_id=7, status="running", mode="auto_first", profile_version=state.profile_version)
    session.add(run)
    session.flush()
    run_annotation(session, fake, state, 7, run, settings)
    session.commit()
    message = run.message
    session.close()

    by_name = {item["name"]: item for item in seen["labels"]}
    # Label type "any" became the shape the annotators draw; the note fills the missing instructions.
    assert by_name["pothole"]["type"] == "polygon"
    assert "polygon" in by_name["pothole"]["how_to_annotate"]
    assert by_name["crack"]["how_to_annotate"] == ""
    assert {item["label"] for item in seen["exemplars"]} == {"pothole", "crack"}
    assert all(item["crop"][:2] == b"\xff\xd8" for item in seen["exemplars"])  # real JPEG crops
    assert "Showed the agent" in message


def test_capture_does_not_refetch_frames_already_learned(tmp_path):
    client, fake = _setup(tmp_path)
    fake.annotations = _hand_drawn(2)
    client.post("/projects/1/sync", headers=HEADERS)
    first = fake.frame_fetches
    assert first > 0
    client.post("/projects/1/sync", headers=HEADERS)
    assert fake.frame_fetches == first


def test_agent_shapes_the_annotator_left_alone_are_not_learned(tmp_path):
    client, fake = _setup(tmp_path)
    fake.annotations = {
        "shapes": [
            {"id": 1, "frame": 0, "label_id": 1, "type": "rectangle", "points": [10, 10, 80, 60], "source": "auto", "attributes": []}
        ],
        "tracks": [],
        "tags": [],
    }
    client.post("/projects/1/sync", headers=HEADERS)
    session = client.app.state.sessions()
    assert session.query(LabelExemplar).count() == 0
    session.close()


def test_requirement_text_is_not_overwritten_by_notes():
    labels = [{"name": "pothole", "type": "polygon", "how_to_annotate": "Outline the hole edge."}]
    notes = {"pothole": {"type": "rectangle", "text": "annotators draw boxes"}}
    assert labelmemory.apply_notes(labels, notes) == []
    assert labels[0]["how_to_annotate"] == "Outline the hole edge."
    assert labels[0]["type"] == "polygon"


def test_keeps_only_a_bounded_number_of_crops_per_label(tmp_path):
    client, fake = _setup(tmp_path)
    shapes = _hand_drawn(0)["shapes"]
    for index in range(labelmemory.KEEP_PER_LABEL + 5):
        shapes.append(
            {"id": 300 + index, "frame": 0, "label_id": 1, "type": "rectangle", "points": [10, 10, 50 + index, 60], "source": "manual", "attributes": []}
        )
    fake.annotations = {"shapes": shapes, "tracks": [], "tags": []}
    client.post("/projects/1/sync", headers=HEADERS)
    session = client.app.state.sessions()
    count = session.query(LabelExemplar).filter_by(label="pothole").count()
    session.close()
    assert count == labelmemory.KEEP_PER_LABEL


def test_forgetting_a_label_removes_its_examples(tmp_path):
    client, fake = _setup(tmp_path)
    fake.annotations = _hand_drawn(3)
    client.post("/projects/1/sync", headers=HEADERS)
    removed = client.delete("/projects/1/label-memory?label=pothole", headers=HEADERS).json()["removed"]
    assert removed == 3
    report = client.get("/projects/1/label-memory", headers=HEADERS).json()
    assert next(item for item in report["labels"] if item["label"] == "crack")["examples_kept"] == 1


def test_vision_request_puts_examples_before_the_target_image(monkeypatch):
    import base64

    from app import vision

    sent = {}

    class Reply:
        status_code = 200
        text = ""

        def json(self):
            return {"content": [{"type": "text", "text": '{"objects":[]}'}]}

    def post(url, headers=None, json=None, timeout=None):
        sent["content"] = json["messages"][0]["content"]
        return Reply()

    monkeypatch.setattr(vision.httpx, "post", post)
    crop = base64.b64decode(base64.b64encode(_jpeg((10, 200, 10))))
    examples = [
        {"label": "pothole", "shape_type": "polygon", "attributes": {}, "size": [80, 40], "points": 4, "crop": crop}
    ]
    labels = [{"name": "pothole", "type": "polygon", "attributes": []}]
    vision._ask_vision("k", "m", _jpeg(), labels, [], refine=False, exemplars=examples)
    kinds = [block["type"] for block in sent["content"]]
    assert kinds.count("image") == 2
    assert kinds[-2:] == ["image", "text"]  # the image to annotate is last
    assert any("pothole" in block.get("text", "") and "polygon" in block.get("text", "") for block in sent["content"])


def test_a_shape_the_annotator_deleted_is_forgotten(tmp_path):
    client, fake = _setup(tmp_path)
    fake.annotations = _hand_drawn(3)
    client.post("/projects/1/sync", headers=HEADERS)
    fake.annotations = {"shapes": [], "tracks": [], "tags": []}
    client.post("/projects/1/sync", headers=HEADERS)
    session = client.app.state.sessions()
    assert session.query(LabelExemplar).count() == 0
    session.close()


def test_a_second_sync_keeps_what_was_learned_from_hand_drawn_shapes(tmp_path):
    """Regression: a re-read job must not turn the person's drawings into 'untouched' agent shapes."""
    client, fake = _setup(tmp_path)
    fake.annotations = _hand_drawn(4)
    for _ in range(3):
        client.post("/projects/1/sync", headers=HEADERS)
    pothole = next(
        item for item in client.get("/projects/1/label-memory", headers=HEADERS).json()["labels"] if item["label"] == "pothole"
    )
    assert pothole["objects_seen"] == 4
    assert pothole["draws_as"] == "polygon"


def test_task_without_a_project_uses_its_own_labels(tmp_path, monkeypatch):
    client, fake = _setup(tmp_path)
    task_labels = [{"id": 9, "name": "pothole", "type": "rectangle", "attributes": []}]
    fake.labels = []  # the project has no labels
    fake.get_task = lambda task_id: {"id": task_id, "project_id": None, "name": "loose", "size": 2, "mode": "annotation"}
    fake.get_task_labels = lambda task_id: task_labels
    seen = {}

    def fake_annotate(api_key, model, image, labels, lessons, refine=True, exemplars=None):
        seen["names"] = [item["name"] for item in labels]
        return []

    monkeypatch.setattr("app.main.annotate_image", fake_annotate)
    settings = client.app.state.settings.model_copy(update={"anthropic_api_key": "k"})
    session = client.app.state.sessions()
    state = get_project(session, 1)
    run = Run(cvat_project_id=1, task_id=7, status="running", mode="auto_first", profile_version=0)
    session.add(run)
    session.flush()
    run_annotation(session, fake, state, 7, run, settings)
    session.close()
    assert seen["names"] == ["pothole"]


def test_hand_made_labels_become_the_requirement_and_follow_later_changes(tmp_path):
    client, fake = _setup(tmp_path)
    first = client.get("/projects/1/spec", headers=HEADERS).json()
    assert first["spec"]["source"] == "cvat_labels"
    assert [item["name"] for item in first["spec"]["labels"]] == ["pothole", "crack"]
    version = first["version"]
    again = client.get("/projects/1/spec", headers=HEADERS).json()
    assert again["version"] == version  # unchanged labels do not create new versions
    fake.labels.append({"id": 3, "name": "manhole", "type": "rectangle", "attributes": []})
    changed = client.get("/projects/1/spec", headers=HEADERS).json()
    assert [item["name"] for item in changed["spec"]["labels"]][-1] == "manhole"
    assert changed["version"] == version + 1


def test_an_uploaded_requirement_is_never_replaced_by_cvat_labels(tmp_path):
    from app.service import sync_spec_from_cvat

    client, fake = _setup(tmp_path)
    session = client.app.state.sessions()
    state = get_project(session, 1)
    state.spec_json = {"labels": [{"name": "car", "type": "rectangle"}], "output_format": "CVAT"}
    assert sync_spec_from_cvat(session, state, fake.labels) is False
    assert state.spec_json["labels"][0]["name"] == "car"
    session.close()


def _auto_setup(tmp_path, monkeypatch, mode):
    client, fake = _setup(tmp_path)
    calls = []

    def fake_annotate(api_key, model, image, labels, lessons, refine=True, exemplars=None):
        calls.append(1)
        return [{"label": "crack", "type": "rectangle", "points": [5, 5, 60, 30], "score": 0.9, "attributes": {}}]

    monkeypatch.setattr("app.main.annotate_image", fake_annotate)
    object.__setattr__(client.app.state.settings, "anthropic_api_key", "k")
    client.post("/projects/1/learning/settings", headers=HEADERS, json={"annotation_mode": mode})
    return client, fake, calls


def test_new_tasks_are_annotated_automatically_in_agent_first_mode(tmp_path, monkeypatch):
    client, fake, calls = _auto_setup(tmp_path, monkeypatch, "auto_first")
    first = client.post("/projects/1/sync", headers=HEADERS).json()
    assert first["auto_annotated"] and first["auto_annotated"][0]["task_id"] == 7
    assert first["auto_annotated"][0]["written"] > 0
    assert calls
    again = client.post("/projects/1/sync", headers=HEADERS).json()
    assert again["auto_annotated"] == []  # a task is annotated once


def test_automatic_annotation_is_off_unless_agent_first(tmp_path, monkeypatch):
    client, fake, calls = _auto_setup(tmp_path, monkeypatch, "human_first")
    assert client.post("/projects/1/sync", headers=HEADERS).json()["auto_annotated"] == []
    assert not calls


def test_a_task_that_already_has_shapes_is_left_alone(tmp_path, monkeypatch):
    client, fake, calls = _auto_setup(tmp_path, monkeypatch, "auto_first")
    fake.annotations = _hand_drawn(1)
    assert client.post("/projects/1/sync", headers=HEADERS).json()["auto_annotated"] == []
    assert not calls


class SixImageCvat(PhotoCvat):
    """A task with six images (frames 0-5) in one job."""

    def __init__(self):
        super().__init__()
        self.jobs = [{**self.jobs[0], "id": 3, "start_frame": 0, "stop_frame": 5}]
        self.size = 6
        self.shapes = []

    def get_task(self, task_id):
        return {"id": task_id, "project_id": 1, "name": "batch", "size": self.size, "mode": "annotation"}


def _range_setup(tmp_path, monkeypatch):
    client, _ = _client(tmp_path)
    fake = SixImageCvat()

    def cvat_client():
        yield fake

    client.app.dependency_overrides[client.app.state.cvat_client] = cvat_client
    asked = []

    def fake_annotate(api_key, model, image, labels, lessons, refine=True, exemplars=None):
        return [{"label": "crack", "type": "rectangle", "points": [5, 5, 60, 30], "score": 0.9, "attributes": {}}]

    original = fake.get_frame

    def get_frame(task_id, frame):
        asked.append(frame)
        return original(task_id, frame)

    fake.get_frame = get_frame
    monkeypatch.setattr("app.main.annotate_image", fake_annotate)
    object.__setattr__(client.app.state.settings, "anthropic_api_key", "k")
    return client, fake, asked


def _annotate(client, **body):
    return client.post("/projects/1/annotate", headers=HEADERS, json={"task_id": 7, **body})


def test_start_and_end_frame_annotate_only_that_range(tmp_path, monkeypatch):
    client, fake, asked = _range_setup(tmp_path, monkeypatch)
    result = _annotate(client, start_frame=2, end_frame=3).json()
    assert sorted(set(asked)) == [2, 3]
    assert sorted(shape["frame"] for shape in fake.shapes) == [2, 3]
    assert "Annotated 2 image(s)" in result["message"]


def test_a_range_keeps_every_shape_outside_it(tmp_path, monkeypatch):
    client, fake, asked = _range_setup(tmp_path, monkeypatch)
    fake.annotations = {
        "shapes": [
            {"id": 1, "frame": 0, "label_id": 2, "type": "rectangle", "points": [1, 1, 40, 40], "source": "auto", "attributes": []},
            {"id": 2, "frame": 5, "label_id": 2, "type": "rectangle", "points": [2, 2, 41, 41], "source": "auto", "attributes": []},
        ],
        "tracks": [],
        "tags": [],
    }
    _annotate(client, start_frame=2, end_frame=3)
    frames = sorted(shape["frame"] for shape in fake.shapes)
    assert frames == [0, 2, 3, 5]  # the old auto shapes on frames 0 and 5 were not wiped


def test_pick_single_images_and_ranges(tmp_path, monkeypatch):
    client, fake, asked = _range_setup(tmp_path, monkeypatch)
    _annotate(client, frames="0, 4-5")
    assert sorted(set(asked)) == [0, 4, 5]


def test_only_new_skips_images_that_already_have_shapes(tmp_path, monkeypatch):
    client, fake, asked = _range_setup(tmp_path, monkeypatch)
    fake.annotations = {
        "shapes": [
            {"id": 1, "frame": frame, "label_id": 2, "type": "rectangle", "points": [1, 1, 40, 40], "source": "auto", "attributes": []}
            for frame in (0, 1, 2, 3)
        ],
        "tracks": [],
        "tags": [],
    }
    _annotate(client, only_new=True)
    assert sorted(set(asked)) == [4, 5]
    assert sorted(shape["frame"] for shape in fake.shapes) == [0, 1, 2, 3, 4, 5]


def test_a_bad_range_is_refused_and_an_empty_one_says_so(tmp_path, monkeypatch):
    client, fake, asked = _range_setup(tmp_path, monkeypatch)
    assert _annotate(client, start_frame=4, end_frame=1).status_code == 400
    assert _annotate(client, frames="abc").status_code == 400
    empty = _annotate(client, start_frame=50, end_frame=60).json()
    assert "No images matched" in empty["message"]
    assert not asked


def test_images_added_to_an_annotated_task_are_annotated_automatically(tmp_path, monkeypatch):
    client, fake, asked = _range_setup(tmp_path, monkeypatch)
    fake.list_tasks = lambda project_id=None: [fake.get_task(7)]
    client.post("/projects/1/learning/settings", headers=HEADERS, json={"annotation_mode": "auto_first"})
    fake.size = 4
    fake.jobs[0]["stop_frame"] = 3
    first = client.post("/projects/1/sync", headers=HEADERS).json()["auto_annotated"]
    assert first and sorted(set(asked)) == [0, 1, 2, 3]
    asked.clear()
    fake.annotations = {"shapes": list(fake.shapes), "tracks": [], "tags": []}
    fake.size = 6  # two images were added to the task
    fake.jobs[0]["stop_frame"] = 5
    second = client.post("/projects/1/sync", headers=HEADERS).json()["auto_annotated"]
    assert second and sorted(set(asked)) == [4, 5]  # only the new images
    asked.clear()
    assert client.post("/projects/1/sync", headers=HEADERS).json()["auto_annotated"] == []
