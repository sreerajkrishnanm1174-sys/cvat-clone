import hashlib
import hmac
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.cvat import CvatError
from app.learning import classify_frame, derive_profile
from app.main import create_app
from app.planning import plan_agents
from app.service import apply_profile_to_shape, label_patch, review_reasons
from app.spec import diff_labels, extract_spec
from app.geometry import crosses_self, resample_polygon
from app.service import associate_by_iou


DOCUMENT = """
Output format: CVAT

Label: pothole
Type: polygon
Min size: 30
Attribute: severity
Input: select
Values: low, medium, high
Default: medium
Required: yes

Label: car
Type: rectangle
Attribute: moving
Input: checkbox
Values: true, false
Default: false
Required: no

Label: crack
Min size: 10
"""


def test_spec_questions_and_idempotent_diff():
    parsed = extract_spec(DOCUMENT)
    names = {label["name"]: label["type"] for label in parsed["labels"]}
    assert names["pothole"] == "polygon"
    assert names["car"] == "rectangle"
    assert any(question["id"] == "crack:type" for question in parsed["questions"])
    diff = diff_labels(parsed, [])
    assert {item["name"] for item in diff["create"]} == {"pothole", "car"}
    assert any(item["name"] == "crack" for item in diff["blocked"])
    existing = [
        {
            "id": 1,
            "name": "pothole",
            "type": "polygon",
            "attributes": [
                {
                    "id": 9,
                    "name": "severity",
                    "input_type": "select",
                    "values": ["low", "medium", "high"],
                    "default_value": "medium",
                }
            ],
        },
        {"id": 2, "name": "car", "type": "rectangle", "attributes": [
            {"id": 10, "name": "moving", "input_type": "checkbox", "values": ["true", "false"], "default_value": "false"}
        ]},
    ]
    again = diff_labels(parsed, existing)
    assert again["create"] == []
    assert again["update"] == []
    assert set(again["unchanged"]) == {"pothole", "car"}
    patch = label_patch(diff, [], confirm=False)
    assert all("deleted" not in item for item in patch)


PROSE = """
Annotate vehicles and road defects. Defects are outlined as polygons.
Ignore objects smaller than 20 px
Label: vehicle
Definition: Any motorised road vehicle: car, van, truck, bus, motorcycle.
Draw the box around the whole vehicle including mirrors
Label: road_defect
Definition: Visible damage on the road surface.
The road_defect row uses the segmentation-polygon format.
{
  "categories": [{"id": 1, "name": "vehicle"}, {"id": 2, "name": "road_defect"}],
  "annotations": [
    {"category_id": 1, "bbox": [1, 2, 3, 4], "attributes": {"color": "white"}},
    {"category_id": 2, "segmentation": [[1, 2, 3, 4, 5, 6]], "attributes": {"severity": "medium"}}
  ]
}
4. Notes
Existing annotations found
COCO: car, truck
"""


def test_requirement_guide_explains_type_drawing_and_decision():
    from app.spec import apply_guide, explain_requirement

    parsed = extract_spec(PROSE)
    guide = explain_requirement(parsed)
    decisions = {item["name"]: item for item in guide["decisions"]}
    assert decisions["vehicle"]["type"] == "rectangle"
    assert "box" in decisions["vehicle"]["how_to_annotate"].lower()
    assert "20 px" in decisions["vehicle"]["how_to_decide"]
    assert "color" in decisions["vehicle"]["how_to_decide"]
    assert decisions["road_defect"]["type"] == "polygon"
    assert "boundary" in decisions["road_defect"]["how_to_annotate"].lower()
    merged = apply_guide(parsed, {
        "summary": guide["summary"],
        "workflow": guide["workflow"],
        "decisions": [
            {**decisions["vehicle"], "type": "polygon"},
            decisions["road_defect"],
        ],
    })
    labels = {item["name"]: item for item in merged["labels"]}
    assert labels["vehicle"]["type"] == "rectangle"
    assert labels["vehicle"]["how_to_decide"]
    assert merged["guide"]["decisions"][0]["how_to_annotate"]


def test_skeleton_connections_follow_the_requirement():
    from app.service import _label_create_body, skeleton_ids_to_replace
    from app.spec import HUMAN_KEYPOINTS, skeleton_svg

    document = """
Label: person
Type: skeleton
Keypoints: nose, left_shoulder, neck
Connections: nose-neck, neck-left_shoulder
"""
    parsed = extract_spec(document)
    label = parsed["labels"][0]
    assert label["edges"] == [["nose", "neck"], ["neck", "left_shoulder"]]
    svg = skeleton_svg(label["keypoints"], label["edges"])
    assert 'data-node-from="1" stroke-width="0.5" data-node-to="3"' in svg
    assert 'data-node-from="1" stroke-width="0.5" data-node-to="2"' not in svg
    body = _label_create_body(label)
    assert [item["name"] for item in body["sublabels"]] == ["nose", "left_shoulder", "neck"]
    chained = skeleton_svg(list(HUMAN_KEYPOINTS))
    assert 'data-node-from="10" stroke-width="0.5" data-node-to="11"' not in chained
    existing = [{
        "id": 9,
        "name": "person",
        "type": "skeleton",
        "svg": skeleton_svg(["nose", "left_shoulder", "neck"], [["nose", "left_shoulder"]]),
        "sublabels": [{"name": "nose"}, {"name": "left_shoulder"}, {"name": "neck"}],
    }]
    assert skeleton_ids_to_replace(parsed, existing) == [9]


def test_skeleton_properties_become_a_project_label_without_a_label_heading():
    from app.spec import apply_guide

    document = """
Skeleton properties
The person is annotated as a skeleton.
Keypoints: nose, left_shoulder, right_shoulder
"""
    parsed = extract_spec(document)
    labels = {item["name"]: item for item in parsed["labels"]}
    assert labels["person"]["type"] == "skeleton"
    assert labels["person"]["keypoints"] == ["nose", "left_shoulder", "right_shoulder"]
    added = apply_guide({"labels": [], "questions": [], "tracking": False}, {
        "summary": "A pose.",
        "workflow": "Place every joint.",
        "tracking": False,
        "decisions": [{
            "name": "hand",
            "type": "skeleton",
            "keypoints": ["wrist", "thumb"],
            "why_this_type": "The properties describe a skeleton.",
            "how_to_annotate": "Place each joint.",
            "how_to_decide": "A hand is visible.",
        }],
    })
    hand = {item["name"]: item for item in added["labels"]}["hand"]
    assert hand["type"] == "skeleton"
    assert hand["keypoints"] == ["wrist", "thumb"]


def test_requirement_can_add_skeleton_keypoints_and_tracking():
    from app.spec import explain_requirement

    document = """
Label: person
Draw a human skeleton. Keypoints: nose, left_shoulder, right_shoulder
Track the person across frames.

Label: wheel
Type: ellipse
"""
    parsed = extract_spec(document)
    labels = {item["name"]: item for item in parsed["labels"]}
    assert labels["person"]["type"] == "skeleton"
    assert labels["person"]["keypoints"] == ["nose", "left_shoulder", "right_shoulder"]
    assert labels["wheel"]["type"] == "ellipse"
    assert parsed["tracking"] is True
    guide = explain_requirement(parsed)
    person = {item["name"]: item for item in guide["decisions"]}["person"]
    assert "nose" in person["how_to_decide"]
    assert guide["tracking"] is True
    patch = label_patch(diff_labels(parsed, []), [], False)
    skeleton = next(item for item in patch if item["name"] == "person")
    assert [item["name"] for item in skeleton["sublabels"]] == ["nose", "left_shoulder", "right_shoulder"]
    assert 'data-label-name="nose"' in skeleton["svg"]
    assert next(item for item in patch if item["name"] == "wheel")["type"] == "ellipse"


def test_real_requirement_wording_becomes_labels():
    parsed = extract_spec(PROSE)
    labels = {item["name"]: item for item in parsed["labels"]}
    assert labels["vehicle"]["type"] == "rectangle"
    assert "car" in labels["vehicle"]["detect_as"]
    assert labels["vehicle"]["min_size"] == 20
    assert any(attr["name"] == "color" for attr in labels["vehicle"]["attributes"])
    assert labels["road_defect"]["type"] == "polygon"
    assert labels["road_defect"]["detect_as"] == []
    assert any(attr["name"] == "severity" for attr in labels["road_defect"]["attributes"])
    assert parsed["questions"] == []


def test_detector_aliases_map_onto_the_project_label():
    from app.service import shapes_for_labels

    kept = shapes_for_labels(
        [{"label": "car", "type": "rectangle", "points": [0, 0, 10, 10], "score": 0.9, "attributes": {}}],
        [{"name": "vehicle", "type": "rectangle", "detect_as": ["car", "truck", "bus", "motorbike"]}],
        {"rules": []},
    )
    assert kept[0]["label"] == "vehicle"


def test_agent_plan_uses_only_deployed_models():
    labels = [
        {"name": "car", "type": "rectangle"},
        {"name": "pothole", "type": "polygon"},
        {"name": "lane", "type": "polyline"},
    ]
    plan = {item["label"]: item for item in plan_agents(labels, allow_external=False)}
    assert plan["car"]["agent"] == "claude_vision"
    assert plan["pothole"]["agent"] == "claude_vision"
    assert "CVAT's own models are not used" in plan["pothole"]["reason"]
    assert plan["lane"]["agent"] == "claude_vision"


def test_one_correction_is_not_yet_a_rule():
    from app.learning import prompt_lessons

    samples = [
        {
            "frame_key": "4:0",
            "items": [
                {
                    "role": "correction",
                    "agent": {"label": "car", "type": "rectangle", "points": [0, 0, 100, 80], "attributes": {}},
                    "human": {"label": "car", "type": "rectangle", "points": [5, 4, 90, 70], "attributes": {}},
                }
            ],
        }
    ]
    profile = derive_profile(samples, {"labels": [{"name": "car", "type": "rectangle"}]})
    assert not any(rule["kind"] == "box_offset" and rule["status"] == "active" for rule in profile["rules"])
    lessons = prompt_lessons(profile, samples)
    assert not any("left +5px" in lesson for lesson in lessons)


def test_three_corrections_on_two_images_become_a_rule():
    from app.learning import prompt_lessons

    samples = []
    for image in ("a", "b"):
        for _ in range(2):
            samples.append(
                {
                    "frame_key": image,
                    "items": [
                        {
                            "role": "correction",
                            "agent": {"label": "car", "type": "rectangle", "points": [0, 0, 100, 80], "attributes": {}},
                            "human": {"label": "car", "type": "rectangle", "points": [5, 4, 90, 70], "attributes": {}},
                        }
                    ],
                }
            )
    profile = derive_profile(samples, {"labels": [{"name": "car", "type": "rectangle"}]})
    assert any(rule["kind"] == "box_offset" and rule["status"] == "active" for rule in profile["rules"])
    lessons = prompt_lessons(profile, samples)
    assert any("left +5px" in lesson and "car" in lesson for lesson in lessons)


def test_one_drawing_does_not_become_a_rule():
    samples = [
        {
            "frame_key": "1:0",
            "items": [{"role": "human", "human": {"label": "car", "type": "rectangle", "points": [0, 0, 10, 10], "attributes": {}}}],
        }
    ]
    profile = derive_profile(samples, {"labels": [{"name": "car", "type": "rectangle"}]})
    assert profile["rules"] == []


def _human_car(image: str, box):
    return {
        "frame_key": image,
        "items": [{"role": "human", "human": {"label": "car", "type": "rectangle", "points": box, "attributes": {"moving": "false"}}}],
    }


def test_habits_need_several_images_and_stay_inside_one_project():
    boxes = [
        [0, 0, 40, 20],
        [2, 2, 42, 22],
        [1, 1, 39, 19],
        [0, 1, 41, 21],
        [3, 0, 44, 20],
        [1, 2, 40, 22],
    ]
    samples = [_human_car(f"job:{index % 3}", box) for index, box in enumerate(boxes)]
    profile = derive_profile(samples, {"labels": [{"name": "car", "type": "rectangle", "attributes": []}]})
    assert any(rule["kind"] == "size_range" and rule["status"] == "active" for rule in profile["rules"])
    other = derive_profile([], {"labels": []})
    assert other["rules"] == []


def test_correction_changes_the_next_box_and_point_count():
    corrections = []
    for image in ("a", "b"):
        for _ in range(2):
            corrections.append(
                {
                    "frame_key": image,
                    "items": [
                        {
                            "role": "correction",
                            "agent": {"label": "car", "type": "rectangle", "points": [0, 0, 100, 80], "score": 0.9},
                            "human": {"label": "car", "type": "rectangle", "points": [5, 4, 90, 70], "attributes": {}},
                        }
                    ],
                }
            )
    profile = derive_profile(corrections, {"labels": [{"name": "car", "type": "rectangle"}]})
    adjusted = apply_profile_to_shape(
        {"label": "car", "type": "rectangle", "points": [0, 0, 100, 80], "score": 0.9, "attributes": {}},
        profile,
        "rectangle",
    )
    assert adjusted["points"][0] == 5
    assert adjusted["points"][2] == 90

    polygons = []
    for image in ("p", "q"):
        for _ in range(2):
            polygons.append(
                {
                    "frame_key": image,
                    "items": [
                        {
                            "role": "correction",
                            "agent": {
                                "label": "pothole",
                                "type": "polygon",
                                "points": [0, 0, 30, 0, 30, 30, 20, 30, 10, 30, 0, 30],
                            },
                            "human": {"label": "pothole", "type": "polygon", "points": [2, 2, 28, 2, 28, 28, 2, 28]},
                        }
                    ],
                }
            )
    poly_profile = derive_profile(polygons, {"labels": [{"name": "pothole", "type": "polygon"}]})
    dense = [0, 0, 10, 0, 20, 0, 30, 0, 30, 10, 30, 20, 30, 30, 20, 30, 10, 30, 0, 30, 0, 20, 0, 10]
    thinned = apply_profile_to_shape(
        {"label": "pothole", "type": "polygon", "points": dense, "attributes": {}},
        poly_profile,
        "polygon",
    )
    assert len(thinned["points"]) // 2 == 4


def test_classification_does_not_double_count():
    snapshot = [{"label": "car", "type": "rectangle", "points": [0, 0, 10, 10], "attributes": {}, "source": "auto"}]
    final = [
        {"label": "car", "type": "rectangle", "points": [0, 0, 10, 10], "attributes": {}, "source": "auto"},
        {"label": "pothole", "type": "polygon", "points": [1, 1, 4, 1, 2, 4], "attributes": {}, "source": "manual"},
        {"label": "car", "type": "rectangle", "points": [50, 50, 60, 60], "attributes": {}, "source": "file"},
    ]
    roles = [item["role"] for item in classify_frame(snapshot, final)]
    assert roles.count("untouched") == 1
    assert roles.count("human") == 1
    assert roles.count("imported") == 1
    human_only = classify_frame([], [final[1]])
    assert human_only[0]["role"] == "human"


def test_spec_conflict_is_not_applied():
    rows = []
    for image in ("a", "b"):
        for _ in range(2):
            rows.append(
                {
                    "frame_key": image,
                    "items": [
                        {
                            "role": "correction",
                            "agent": {"label": "car", "type": "rectangle", "points": [0, 0, 10, 10]},
                            "human": {"label": "truck", "type": "rectangle", "points": [0, 0, 10, 10], "attributes": {}},
                        }
                    ],
                }
            )
    profile = derive_profile(
        rows,
        {"labels": [{"name": "car", "type": "rectangle"}, {"name": "truck", "type": "rectangle"}]},
    )
    rule = next(rule for rule in profile["rules"] if rule["kind"] == "label_confusion")
    assert rule["status"] == "conflict"
    kept = apply_profile_to_shape(
        {"label": "car", "type": "rectangle", "points": [0, 0, 10, 10], "attributes": {}},
        profile,
        "rectangle",
    )
    assert kept["label"] == "car"


def test_review_flags_and_self_crossing_and_tracks():
    bowtie = [0, 0, 10, 10, 0, 10, 10, 0]
    assert crosses_self(bowtie)
    reasons = review_reasons(
        {"label": "pothole", "type": "polygon", "points": bowtie, "attributes": {"severity": "huge"}, "score": 0.1},
        {"rules": []},
        {"name": "pothole", "attributes": [{"name": "severity", "values": ["low", "high"]}]},
    )
    assert any("allowed" in reason for reason in reasons)
    assert any("crosses" in reason for reason in reasons)
    linked = associate_by_iou(
        {
            0: [{"label": "car", "type": "rectangle", "points": [0, 0, 10, 10]}],
            1: [{"label": "car", "type": "rectangle", "points": [1, 1, 11, 11]}],
        }
    )
    assert linked[0][0]["track_id"] == linked[1][0]["track_id"]
    assert resample_polygon([0, 0, 10, 0, 10, 10, 0, 10], 4)


class FakeCvat:
    def __init__(self):
        self.labels = []
        self.issues = []
        self.shapes = []
        self.jobs = [
            {
                "id": 3,
                "task_id": 7,
                "project_id": 1,
                "start_frame": 0,
                "stop_frame": 1,
                "state": "completed",
                "stage": "annotation",
                "updated_date": "2026-01-01T00:00:00Z",
            }
        ]
        self.annotations = {"shapes": [], "tracks": [], "tags": []}
        self.job_updates = []
        self.webhooks = []

    def close(self):
        return None

    def get_project(self, project_id):
        if project_id == 99:
            raise CvatError(403, "forbidden")
        if project_id != 1:
            raise CvatError(404, "missing")
        return {"id": 1, "name": "road"}

    def get_labels(self, project_id):
        return list(self.labels)

    def patch_project_labels(self, project_id, labels):
        for label in labels:
            self.labels.append({**label, "id": len(self.labels) + 1, "attributes": [
                {**attr, "id": index + 1} for index, attr in enumerate(label.get("attributes") or [])
            ]})
        return {}

    def get_task(self, task_id):
        return {"id": task_id, "project_id": 1, "name": "batch", "size": 2, "mode": "annotation"}

    def get_task_meta(self, task_id):
        return {"frames": [{"width": 100, "height": 80, "name": "a.jpg"}, {"width": 100, "height": 80, "name": "b.jpg"}]}

    def get_jobs(self, task_id=None, project_id=None):
        return list(self.jobs)

    def get_annotations(self, kind, resource_id):
        return self.annotations

    def put_annotations(self, kind, resource_id, payload):
        self.shapes = list(payload["shapes"])

    def create_annotations(self, kind, resource_id, payload):
        self.shapes.extend(payload["shapes"])

    def update_job(self, job_id, payload):
        self.job_updates.append(payload)

    def create_issue(self, job_id, frame, message, position):
        self.issues.append({"job": job_id, "frame": frame, "message": message, "position": position})
        return {"id": len(self.issues)}

    def create_webhook(self, project_id, target_url, secret, events):
        self.webhooks.append(target_url)
        return {"id": 5}

    def get_frame(self, task_id, frame):
        return b""


def _client(tmp_path: Path):
    app = create_app(
        Settings(
            database_url=f"sqlite:///{tmp_path / 'test.sqlite3'}",
            analyzer_data_dir=str(tmp_path),
            webhook_secret="test-secret",
            cvat_base_url="http://cvat.invalid",
            analyzer_webhook_url="http://analyzer:8090/webhooks/cvat",
        )
    )
    fake = FakeCvat()

    def identity():
        return {"scheme": "Bearer", "token": "user", "organization": ""}

    def cvat_client():
        yield fake

    app.dependency_overrides[app.state.identity] = identity
    app.dependency_overrides[app.state.cvat_client] = cvat_client
    return TestClient(app), fake


def test_api_apply_is_idempotent_and_permissions_fail(tmp_path):
    client, fake = _client(tmp_path)
    uploaded = client.post(
        "/projects/1/requirements",
        files={"file": ("req.txt", DOCUMENT.encode("utf-8"), "text/plain")},
        headers={"Authorization": "Bearer user"},
    )
    assert uploaded.status_code == 200, uploaded.text
    applied = client.post(
        "/projects/1/labels/apply",
        json={"confirm_type_changes": False},
        headers={"Authorization": "Bearer user"},
    )
    assert applied.status_code == 200, applied.text
    assert {label["name"] for label in fake.labels} == {"pothole", "car"}
    second = client.post(
        "/projects/1/labels/apply",
        json={"confirm_type_changes": False},
        headers={"Authorization": "Bearer user"},
    )
    assert second.json()["applied"] == 0
    denied = client.get("/projects/99/spec", headers={"Authorization": "Bearer user"})
    assert denied.status_code == 403
    bare = create_app(
        Settings(
            database_url=f"sqlite:///{tmp_path / 'auth.sqlite3'}",
            analyzer_data_dir=str(tmp_path),
            webhook_secret="test-secret",
        )
    )
    assert TestClient(bare).get("/projects/1/spec").status_code == 401


def test_webhook_signature_duplicate_and_both_modes(tmp_path):
    client, fake = _client(tmp_path)
    headers = {"Authorization": "Bearer user"}
    client.post("/projects/1/requirements", files={"file": ("req.txt", DOCUMENT.encode(), "text/plain")}, headers=headers)
    client.post("/projects/1/labels/apply", json={"confirm_type_changes": False}, headers=headers)
    client.post("/projects/1/learning/settings", json={"annotation_mode": "human_first", "learn_enabled": True}, headers=headers)
    fake.annotations = {
        "shapes": [
            {
                "frame": 0,
                "label_id": fake.labels[1]["id"],
                "type": "rectangle",
                "points": [1, 1, 20, 10],
                "source": "manual",
                "attributes": [],
            }
        ],
        "tracks": [],
        "tags": [],
    }
    payload = {
        "event": "update:job",
        "webhook_id": 5,
        "job": {"id": 3, "task_id": 7, "project_id": 1, "state": "completed", "updated_date": "t1", "start_frame": 0},
    }
    raw = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(b"test-secret", raw, hashlib.sha256).hexdigest()
    bad = client.post("/webhooks/cvat", content=raw, headers={"X-Signature-256": "sha256=nope"})
    assert bad.status_code == 401
    # No service token: the event is stored and not treated as a person's drawing yet.
    stored = client.post("/webhooks/cvat", content=raw, headers={"X-Signature-256": signature})
    assert stored.status_code == 200
    assert stored.json()["status"] in {"stored", "learned"}
    duplicate = client.post("/webhooks/cvat", content=raw, headers={"X-Signature-256": signature})
    assert duplicate.json()["status"] == "duplicate"

    learned = client.post("/projects/1/sync", headers=headers)
    assert learned.status_code == 200
    assert learned.json()["samples"] == 1

    def detect(frame, labels):
        return [{"label": "car", "type": "rectangle", "points": [0, 0, 30, 20], "score": 0.8, "attributes": {}}]

    from app.main import run_annotation
    from app.db import Run
    from app.service import get_project

    session = client.app.state.sessions()
    state = get_project(session, 1)
    # Give the profile something to apply by recording enough corrections through the API samples.
    run = Run(cvat_project_id=1, task_id=7, status="running", mode="auto_first", profile_version=state.profile_version)
    session.add(run)
    session.flush()
    summary = run_annotation(session, fake, state, 7, run, client.app.state.settings, detect=detect)
    session.commit()
    session.close()
    assert summary["written"] == 1
    autos = [shape for shape in fake.shapes if shape["source"] == "auto"]
    manuals = [shape for shape in fake.shapes if shape["source"] == "manual"]
    assert len(autos) == 1
    assert len(manuals) == 1
    assert autos[0]["label_id"] == fake.labels[1]["id"]
    assert fake.job_updates == []


def test_chat_can_revise_requirement_labels_and_delete_a_task():
    from app.chat import revise_spec, run_tool

    parsed = {
        "labels": [
            {"name": "standing", "type": "skeleton"},
            {"name": "person", "type": "skeleton"},
            {"name": "person_bbox", "type": "rectangle"},
        ]
    }
    removed = revise_spec(parsed, remove=["standing"])
    assert [item["name"] for item in removed["labels"]] == ["person", "person_bbox"]
    kept = revise_spec(parsed, keep=["person"])
    assert [item["name"] for item in kept["labels"]] == ["person"]

    class Tasks:
        def __init__(self):
            self.deleted = []

        def get_task(self, task_id):
            return {"id": task_id, "project_id": 6, "name": "task"}

        def delete_task(self, task_id):
            self.deleted.append(task_id)

    client = Tasks()
    result = run_tool("delete_task", {"project_id": 6, "task_id": 14}, client, None, None, {})
    assert client.deleted == [14]
    assert "14" in result


def test_chat_agent_creates_a_project_for_the_signed_in_user():
    from app.chat import run_tool

    class Projects:
        def __init__(self):
            self.created = []

        def list_projects(self):
            return [{"id": 2, "name": "road"}]

        def create_project(self, name):
            self.created.append(name)
            return {"id": 9, "name": name}

    client = Projects()
    listed = run_tool("list_projects", {}, client, None, None, {})
    assert "road" in listed
    created = run_tool("create_project", {"name": "Night defects"}, client, None, None, {})
    assert client.created == ["Night defects"]
    assert "/projects/9" in created


def test_modes_are_both_visible_in_learning_panel(tmp_path):
    client, _fake = _client(tmp_path)
    headers = {"Authorization": "Bearer user"}
    client.post("/projects/1/learning/settings", json={"annotation_mode": "auto_first"}, headers=headers)
    view = client.get("/projects/1/learning", headers=headers)
    assert view.json()["mode_text"] == "The agent annotates first, you correct"
    client.post("/projects/1/learning/settings", json={"annotation_mode": "human_first"}, headers=headers)
    view = client.get("/projects/1/learning", headers=headers)
    assert view.json()["mode_text"] == "You annotate, the agent learns"
    forgotten = client.post("/projects/1/learning/reset", headers=headers)
    assert forgotten.json()["summary"] == "The agent has learned from 0 finished frames."


def test_vision_shapes_and_correction_lessons():
    from app.learning import prompt_lessons
    from app.vision import parse_shapes

    labels = [
        {"name": "vehicle", "type": "rectangle", "detect_as": ["car"], "min_size": 20, "attributes": []},
        {"name": "road_defect", "type": "polygon", "attributes": []},
    ]
    text = json.dumps(
        {
            "shapes": [
                {"label": "car", "type": "bbox", "points": [10, 10, 40, 40], "attributes": {"color": "white"}},
                {"label": "road_defect", "type": "polygon", "points": [1, 2, 3, 4, 5, 6], "attributes": {"severity": "medium"}},
                {"label": "person", "type": "rectangle", "points": [0, 0, 9, 9]},
                {"label": "vehicle", "type": "rectangle", "points": [0, 0, 5, 5]},
            ]
        }
    )
    reference = json.dumps(
        {
            "objects": [
                {"label": "vehicle", "bbox": [10, 10, 40, 40], "score": 0.9, "attributes": {"color": "white"}},
                {"label": "road_defect", "polygon": [[1, 2], [30, 2], [30, 40], [1, 40]]},
            ]
        }
    )
    referenced = parse_shapes(reference, labels, scale=2, width=100, height=80)
    assert referenced[0]["type"] == "rectangle" and referenced[0]["points"] == [20, 20, 80, 80]
    assert referenced[1]["type"] == "polygon" and referenced[1]["points"][0] == 2
    from app.vision import _coarse_polygons
    assert _coarse_polygons([{"type": "polygon", "points": [0, 0, 1, 0, 1, 1, 0, 1, 2, 2, 3, 3]}])
    assert not _coarse_polygons([{"type": "polygon", "points": list(range(16))}])
    shapes = parse_shapes(text, labels, scale=2, width=100, height=80)
    assert [item["label"] for item in shapes] == ["vehicle", "road_defect"]
    messy = parse_shapes('{"shapes":["vehicle", {"label":"road_defect","type":"polygon","points":[[1,2],[3,4],[5,6]]}]}', labels, 1, 100, 80)
    assert messy[0]["label"] == "road_defect" and messy[0]["type"] == "polygon"
    assert shapes[0]["points"] == [20, 20, 80, 80]
    assert shapes[1]["type"] == "polygon"
    lessons = prompt_lessons(
        {"rules": []},
        [
            {
                "items": [
                    {
                        "role": "correction",
                        "agent": {"label": "vehicle", "type": "rectangle", "points": [0, 0, 10, 10], "attributes": {}},
                        "human": {"label": "road_defect", "type": "polygon", "points": [0, 0, 10, 0, 10, 10], "attributes": {}},
                    }
                ]
            }
        ],
    )
    assert not any("road_defect" in line for line in lessons)
    skeleton = parse_shapes(
        json.dumps({
            "objects": [{
                "label": "person",
                "keypoints": [
                    {"name": "nose", "x": 10, "y": 12, "v": 2},
                    {"name": "left_shoulder", "x": 8, "y": 20, "v": 1},
                    {"name": "right_shoulder", "x": 0, "y": 0, "v": 0},
                ],
            }]
        }),
        [{"name": "person", "type": "skeleton", "keypoints": ["nose", "left_shoulder", "right_shoulder"]}],
        scale=2,
        width=100,
        height=80,
    )
    assert skeleton[0]["type"] == "skeleton"
    assert skeleton[0]["elements"][0]["points"] == [20, 24]
    assert skeleton[0]["elements"][2]["outside"] is True
    from app.service import to_cvat_shape
    written = to_cvat_shape(
        skeleton[0],
        {"person": {"id": 3, "name": "person", "sublabels": [
            {"id": 4, "name": "nose"},
            {"id": 5, "name": "left_shoulder"},
            {"id": 6, "name": "right_shoulder"},
        ], "attributes": []}},
        0,
    )
    assert written["type"] == "skeleton" and written["points"] == []
    assert [item["label_id"] for item in written["elements"]] == [4, 5, 6]
    assert written["elements"][2]["outside"] is True


def test_project_steps_and_clearing_a_wrong_label():
    from app.chat import run_tool
    from app.report import extra_label_deletions, requirement_steps

    spec = {
        "output_format": "YOLO",
        "labels": [{"name": "person", "type": "skeleton", "keypoints": ["nose"]}],
        "guide": {"summary": "Mark each person.", "decisions": [{"name": "person", "how_to_annotate": "Place the nose."}]},
    }
    steps = requirement_steps(spec)
    assert steps[0]["title"] == "Requirement"
    assert any("person" in step["title"] for step in steps)
    assert steps[-1]["title"] == "Export"
    existing = [
        {"id": 1, "name": "person", "type": "skeleton"},
        {"id": 2, "name": "person", "type": "skeleton"},
        {"id": 3, "name": "vehicle", "type": "rectangle"},
        {"id": 4, "name": "nose", "parent_id": 2},
    ]
    removed = extra_label_deletions(spec, existing)
    assert {item["id"] for item in removed} == {2, 3, 4}

    class Project:
        def __init__(self):
            self.cleared = []

        def get_task(self, task_id):
            return {"id": task_id, "project_id": 4}

        def get_jobs(self, task_id=None):
            return [{"id": 8}]

        def get_annotations(self, kind, resource_id):
            return {"version": 1, "tags": [], "shapes": [{"id": 1}, {"id": 2}], "tracks": []}

        def put_annotations(self, kind, resource_id, payload):
            self.cleared.append(payload["shapes"])

    project = Project()
    result = run_tool("clear_annotations", {"project_id": 4, "task_id": 14}, project, None, None, {})
    assert project.cleared == [[]]
    assert "2" in result


def test_requirement_export_format():
    from app.exporting import build_archive, export_kind
    import io
    import zipfile

    parsed = extract_spec("Export format: YOLO v8 detection\n\nLabel: vehicle\nType: rectangle\n")
    assert parsed["output_format"].lower().startswith("yolo")
    assert export_kind(parsed["output_format"]) == "yolo"
    assert export_kind("CVAT") == ""
    assert export_kind("COCO keypoints") == "coco"
    blob = build_archive(
        "yolo",
        "YOLO v8 detection",
        ["vehicle", "road_defect"],
        [{
            "name": "road.jpg",
            "width": 100,
            "height": 50,
            "shapes": [{"label": "vehicle", "type": "rectangle", "points": [10, 10, 30, 40]}],
        }],
    )
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        line = archive.read("labels/road.txt").decode().split()
        classes = archive.read("classes.txt").decode().splitlines()
    assert classes == ["vehicle", "road_defect"]
    assert line[0] == "0"
    assert line[1] == "0.200000"
    assert line[2] == "0.500000"
    pose = build_archive(
        "yolo-pose",
        "YOLO pose",
        ["person"],
        [{
            "name": "person.jpg",
            "width": 100,
            "height": 100,
            "shapes": [{
                "label": "person",
                "type": "skeleton",
                "points": [],
                "keypoint_order": ["nose", "left_shoulder"],
                "elements": [
                    {"name": "nose", "points": [40, 20], "outside": False, "occluded": False},
                    {"name": "left_shoulder", "points": [0, 0], "outside": True, "occluded": False},
                ],
            }],
        }],
    )
    with zipfile.ZipFile(io.BytesIO(pose)) as archive:
        pose_line = archive.read("labels/person.txt").decode().split()
    assert pose_line[-1] == "0"
    assert pose_line[-4] == "2"


def test_image_upload_waits_for_cvat_import(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, status, payload):
            self.status_code = status
            self.content = json.dumps(payload).encode()
            self.text = self.content.decode()

        def json(self):
            return json.loads(self.content)

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs.get("headers") or {}, kwargs.get("files")))
            if kwargs.get("headers", {}).get("Upload-Finish"):
                return FakeResponse(202, {"rq_id": "import-1"})
            if path == "/api/requests/import-1":
                return FakeResponse(200, {"status": "finished", "message": ""})
            if path == "/api/tasks/7":
                return FakeResponse(200, {"id": 7, "size": 2})
            return FakeResponse(202, {})

        def close(self):
            return None

    monkeypatch.setattr("app.cvat.httpx.Client", FakeClient)
    from app.cvat import CvatClient

    CvatClient("http://cvat", "token").upload_task_images(7, [("road.jpg", b"jpeg"), ("lane.jpg", b"jpeg")])
    headers = [item[2] for item in calls if item[1].endswith("/data")]
    assert {"Upload-Start": "true"} in [{key: value} for header in headers for key, value in header.items() if key.startswith("Upload-")] or any(
        header.get("Upload-Start") == "true" for header in headers
    )
    assert any(header.get("Upload-Multiple") == "true" for header in headers)
    assert any(header.get("Upload-Finish") == "true" for header in headers)
    uploaded = next(item[3] for item in calls if item[2].get("Upload-Multiple"))
    assert uploaded[0][0] == "client_files[0]"
    assert uploaded[1][0] == "client_files[1]"
    assert any(item[1] == "/api/requests/import-1" for item in calls)
