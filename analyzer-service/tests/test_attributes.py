"""Attributes: read from the label, filled by the agent, written to CVAT in a form CVAT accepts."""

import pytest

from app import attributes
from app.service import _attribute_body, to_cvat_shape
from app.vision import build_prompt

SPECS = [
    {"id": 11, "name": "severity", "input_type": "select", "values": ["low", "medium", "high"], "default_value": "low"},
    {"id": 12, "name": "wet", "input_type": "checkbox", "values": ["false"], "default_value": "false"},
    {"id": 13, "name": "depth_cm", "input_type": "number", "values": ["0", "50", "5"], "default_value": "0"},
    {"id": 14, "name": "note", "input_type": "text", "values": [], "default_value": ""},
    {"id": 15, "name": "side", "input_type": "radio", "values": ["left", "right"], "default_value": ""},
]


def _by_id(result):
    return {item["spec_id"]: item["value"] for item in result}


def test_values_are_mapped_onto_each_attribute_type():
    out = _by_id(
        attributes.cvat_attributes(
            {"Severity": "HIGH", "wet": True, "depth_cm": 17.4, "note": "  edge broken ", "side": "Right"}, SPECS
        )
    )
    assert out == {11: "high", 12: "true", 13: "15", 14: "edge broken", 15: "right"}


def test_checkbox_words_and_number_limits():
    assert _by_id(attributes.cvat_attributes({"wet": "No"}, SPECS))[12] == "false"
    assert _by_id(attributes.cvat_attributes({"wet": "yes"}, SPECS))[12] == "true"
    assert _by_id(attributes.cvat_attributes({"depth_cm": 900}, SPECS))[13] == "50"  # clamped to max
    assert _by_id(attributes.cvat_attributes({"depth_cm": -3}, SPECS))[13] == "0"  # clamped to min
    assert _by_id(attributes.cvat_attributes({"depth_cm": "deep"}, SPECS))[13] == "0"  # default


def test_invalid_or_missing_values_fall_back_to_the_default():
    out = _by_id(attributes.cvat_attributes({"severity": "catastrophic"}, SPECS))
    assert out[11] == "low"  # not in the list: default, never an invalid value
    assert out[12] == "false" and out[13] == "0"  # skipped by the agent: default
    assert out[15] == "left"  # radio with no default takes its first value
    assert 14 not in out  # empty text default: nothing to write


def test_unknown_names_and_specs_without_an_id_are_ignored():
    out = attributes.cvat_attributes({"colour": "red"}, SPECS + [{"name": "ghost", "input_type": "text"}])
    assert all(item["spec_id"] != None for item in out)  # noqa: E711
    assert not any(item["value"] == "red" for item in out)


def test_shape_written_to_cvat_carries_valid_attributes():
    label = {"id": 7, "name": "pothole", "attributes": SPECS}
    shape = {"label": "pothole", "type": "rectangle", "points": [1, 1, 20, 20], "attributes": {"severity": "medium", "wet": "True"}}
    payload = to_cvat_shape(shape, {"pothole": label}, 0)
    values = _by_id(payload["attributes"])
    assert values[11] == "medium" and values[12] == "true"


def test_the_agent_is_told_each_attribute_type_and_its_choices():
    import json

    labels = [{"name": "pothole", "type": "rectangle", "attributes": SPECS}]
    prompt = build_prompt(labels, [], 800, 600)
    brief = json.loads(prompt[prompt.index("[") : prompt.rindex("]") + 1])
    told = {item["name"]: item for item in brief[0]["attributes"]}
    assert told["wet"]["input_type"] == "checkbox" and told["wet"]["values"] == ["true", "false"]
    assert told["severity"]["values"] == ["low", "medium", "high"] and told["severity"]["default"] == "low"
    assert (told["depth_cm"]["min"], told["depth_cm"]["max"], told["depth_cm"]["step"]) == ("0", "50", "5")
    assert told["side"]["default"] == "left"  # a radio with no default starts on its first value


def test_attributes_read_from_a_requirement_get_defaults_cvat_accepts():
    check = _attribute_body({"name": "wet", "input_type": "checkbox", "values": [], "default": ""})
    assert check["default_value"] == "false" and check["values"] == ["false"]
    pick = _attribute_body({"name": "severity", "input_type": "select", "values": ["low", "high"], "default": ""})
    assert pick["default_value"] == "low"  # a select default must be one of its values
    text = _attribute_body({"name": "note", "input_type": "text", "values": [], "default": ""})
    assert text["default_value"] == ""


def test_a_requirement_can_name_cvat_attribute_types_in_other_words_and_set_mutable():
    from app.spec import diff_labels, extract_spec

    text = (
        "Output format: CVAT\n\n"
        "Label: car\nType: rectangle\n"
        "Attribute: colour\nInput: dropdown\nValues: red, blue\nDefault: red\nRequired: yes\nMutable: no\n"
        "Attribute: moving\nInput: boolean\nValues: true, false\nDefault: false\nRequired: no\n"
        "Attribute: speed\nInput: integer\nValues: 0, 200, 5\nDefault: 0\nRequired: no\nMutable: yes\n"
    )
    label = extract_spec(text)["labels"][0]
    attrs = {item["name"]: item for item in label["attributes"]}
    assert attrs["colour"]["input_type"] == "select" and attrs["colour"]["mutable"] is False
    assert attrs["moving"]["input_type"] == "checkbox" and attrs["moving"]["mutable"] is True
    assert attrs["speed"]["input_type"] == "number" and attrs["speed"]["values"] == ["0", "200", "5"]
    # creating the label in CVAT keeps that flag
    assert _attribute_body(attrs["colour"])["mutable"] is False
    # a label that exists in CVAT with a mutable colour is updated when the requirement says it is fixed
    existing = [{"id": 1, "name": "car", "type": "rectangle", "attributes": [
        {"id": 9, "name": "colour", "input_type": "select", "values": ["red", "blue"], "mutable": True},
        {"id": 10, "name": "moving", "input_type": "checkbox", "values": ["false"], "mutable": True},
        {"id": 11, "name": "speed", "input_type": "number", "values": ["0", "200", "5"], "mutable": True},
    ]}]
    plan = diff_labels({"labels": [label]}, existing)
    changed = [change["name"] for entry in plan["update"] for change in entry["attributes"]]
    assert changed == ["colour", "speed"]  # the requirement's explicit numeric default is applied too


def test_attribute_default_conflicts_are_in_the_preview_and_can_be_applied():
    from app.spec import diff_labels
    from app.service import label_patch

    requirement = {"labels": [{"name": "car", "type": "rectangle", "attributes": [
        {"name": "colour", "input_type": "select", "values": ["red", "blue"], "default": "blue"},
    ]}]}
    current = [{"id": 1, "name": "car", "type": "rectangle", "attributes": [
        {"id": 9, "name": "colour", "input_type": "select", "values": ["red", "blue"], "default_value": "red"},
    ]}]
    diff = diff_labels(requirement, current)
    assert diff["update"][0]["attributes"][0]["action"] == "change"
    patch = label_patch(diff, current, confirm=False)
    assert patch[0]["attributes"][0]["id"] == 9
    assert patch[0]["attributes"][0]["default_value"] == "blue"


def test_attribute_name_case_difference_updates_existing_attribute_without_duplication():
    from app.service import label_patch
    from app.spec import diff_labels

    requirement = {"labels": [{"name": "car", "type": "rectangle", "attributes": [
        {"name": "Color", "input_type": "select", "values": ["red", "blue"], "default": "red"},
    ]}]}
    current = [{"id": 1, "name": "car", "type": "rectangle", "attributes": [
        {"id": 9, "name": "color", "input_type": "text", "values": [], "default_value": ""},
    ]}]
    diff = diff_labels(requirement, current)
    patch = label_patch(diff, current, confirm=False)
    attrs = patch[0]["attributes"]
    assert len(attrs) == 1 and attrs[0]["id"] == 9 and attrs[0]["name"] == "Color"


def test_project_report_lists_all_final_attributes_and_requirement_statuses():
    from app.report import project_report

    class ReportCvat:
        def get_labels(self, project_id):
            return [
                {"id": 1, "name": "car", "type": "rectangle", "attributes": [
                    {"id": 9, "name": "colour", "input_type": "select", "values": ["red", "blue"], "default_value": "blue"},
                    {"id": 10, "name": "moving", "input_type": "checkbox", "values": ["false"], "default_value": "false"},
                ]},
                {"id": 2, "name": "legacy", "type": "polygon", "attributes": [
                    {"id": 11, "name": "note", "input_type": "text", "values": [], "default_value": ""},
                ]},
            ]

        def list_tasks(self, project_id):
            return []

    parsed = {
        "labels": [{"name": "car", "type": "rectangle", "attributes": [
            {"name": "colour", "input_type": "select", "values": ["red", "blue"], "default": "blue"},
            {"name": "moving", "input_type": "checkbox", "values": [], "default": "false"},
        ]}],
        "_configuration_status": {"labels": {"car": {
            "status": "updated from requirement",
            "attributes": {"colour": "newly added", "moving": "already existed"},
        }}},
    }
    report = project_report(ReportCvat(), 4, parsed)
    by_name = {item["label"]: item for item in report["attribute_configuration"]}
    assert set(by_name) == {"car", "legacy"}
    assert by_name["car"]["status"] == "updated from requirement"
    attrs = {item["name"]: item for item in by_name["car"]["attributes"]}
    assert attrs["colour"]["status"] == "newly added" and attrs["colour"]["default"] == "blue"
    assert attrs["moving"]["values"] == ["true", "false"]
    assert by_name["legacy"]["status"] == "unchanged"


def test_document_reader_extracts_docx_table_text_and_embedded_images(tmp_path):
    from docx import Document
    from PIL import Image

    from app.spec import read_document, read_document_images

    picture_path = tmp_path / "screen.png"
    Image.new("RGB", (20, 20), "white").save(picture_path)
    document = Document()
    document.add_paragraph("Record the visible vehicle color.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Label"
    table.cell(0, 1).text = "Color values: red, yellow"
    document.add_picture(str(picture_path))
    path = tmp_path / "requirements.docx"
    document.save(path)

    text = read_document(path)
    images = read_document_images(path)
    assert "Record the visible vehicle color" in text
    assert "Color values: red, yellow" in text
    assert len(images) == 1 and images[0]["media_type"] == "image/jpeg"
    assert images[0]["data"].startswith(b"\xff\xd8")


def test_requirement_analysis_sends_full_text_existing_labels_and_visuals(monkeypatch):
    import base64

    from app import vision

    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"content": [{"type": "text", "text": '{"decisions":[{"name":"car","type":"rectangle","attributes":[]}]}'}]}

    def post(_url, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(vision.httpx, "post", post)
    full_text = "requirement " + ("x" * 13000)
    images = [{"name": "page-1", "media_type": "image/jpeg", "data": b"image bytes"}]
    result = vision.analyze_requirement(
        "fake-key", "model", full_text, {"labels": []}, images,
        [{"name": "car", "type": "rectangle", "attributes": []}],
    )
    content = captured["messages"][0]["content"]
    text_block = next(item["text"] for item in content if item["type"] == "text")
    image_block = next(item for item in content if item["type"] == "image")
    assert full_text in text_block
    assert '"name": "car"' in text_block
    assert base64.b64decode(image_block["source"]["data"]) == b"image bytes"
    assert result["decisions"][0]["name"] == "car"


def test_cvat_apply_verifies_persisted_attributes_and_surfaces_readback_failure():
    import json

    from app.main import _apply_and_verify_labels
    from app.cvat import CvatError
    from app.spec import diff_labels

    class Cvat:
        def __init__(self, drops_attributes=False):
            self.labels = [{"id": 204, "name": "car", "type": "rectangle", "attributes": []}]
            self.drops_attributes = drops_attributes
            self.patches = []

        def patch_project_labels(self, project_id, patch):
            self.patches.append(patch)
            if not self.drops_attributes:
                for update in patch:
                    label = next(item for item in self.labels if item["id"] == update["id"])
                    label["attributes"] = [dict(attr, id=301 + index) for index, attr in enumerate(update["attributes"])]
            return {}

        def get_labels(self, project_id):
            return self.labels

    parsed = {"labels": [{"name": "car", "type": "rectangle", "attributes": [
        {"name": "color", "input_type": "select", "values": ["red", "blue"], "default": "blue"},
    ]}]}
    cvat = Cvat()
    applied, actual = _apply_and_verify_labels(cvat, 5, parsed, cvat.get_labels(5), diff_labels(parsed, cvat.get_labels(5)), False)
    assert applied == 1 and actual[0]["id"] == 204
    assert actual[0]["attributes"][0]["default_value"] == "blue"

    rejecting = Cvat(drops_attributes=True)
    with pytest.raises(CvatError) as failure:
        _apply_and_verify_labels(rejecting, 5, parsed, rejecting.get_labels(5), diff_labels(parsed, rejecting.get_labels(5)), False)
    assert failure.value.status == 502
    assert "attribute is missing" in json.loads(failure.value.detail)["mismatches"][0]["problem"]


def test_requirement_analysis_http_failure_is_reported_instead_of_looking_complete(monkeypatch):
    from types import SimpleNamespace

    from app import vision
    from app.main import _attribute_sync_report, _with_guide

    class Response:
        status_code = 422
        text = '{"error":"image processing failed"}'

    monkeypatch.setattr(vision.httpx, "post", lambda *_args, **_kwargs: Response())
    parsed = {"labels": [{"name": "car", "type": "rectangle", "attributes": []}]}
    result = _with_guide(parsed, "Some full requirement text", SimpleNamespace(anthropic_api_key="key", anthropic_model="model"), [{"data": b"img"}], [])
    report = _attribute_sync_report(result, [])
    assert report["status"] == "extraction_incomplete"
    assert "HTTP 422" in report["warnings"][0]


def test_video_tracks_keep_fixed_attributes_on_the_track_and_changing_ones_on_keyframes():
    from app.service import to_cvat_tracks

    label = {
        "id": 3,
        "name": "car",
        "attributes": [
            {"id": 31, "name": "colour", "input_type": "select", "values": ["red", "blue"], "mutable": False},
            {"id": 32, "name": "moving", "input_type": "checkbox", "values": ["false"], "mutable": True},
        ],
    }
    frames = {
        frame: [
            {"label": "car", "type": "rectangle", "points": [10, 10, 50, 40], "track_id": 1,
             "attributes": {"colour": colour, "moving": moving}}
        ]
        for frame, colour, moving in ((0, "red", False), (1, "red", True), (2, "blue", True))
    }
    track = to_cvat_tracks(frames, {"car": label})[0]
    # the fixed attribute is one value for the whole object: the one most frames agree on
    assert track["attributes"] == [{"spec_id": 31, "value": "red"}]
    for shape in track["shapes"]:
        assert [item["spec_id"] for item in shape["attributes"]] == [32]
    assert [shape["attributes"][0]["value"] for shape in track["shapes"]] == ["false", "true", "true"]


# ---------------------------------------------------------------- attributes from the requirement document


def test_attributes_described_in_words_are_added_to_the_label():
    from app import spec

    parsed = {"labels": [{"name": "vehicle_mirror", "type": "polygon", "attributes": []}], "output_format": "CVAT"}
    guide = {
        "summary": "s",
        "workflow": "w",
        "decisions": [
            {
                "name": "vehicle_mirror",
                "type": "polygon",
                "attributes": [
                    {"name": "color", "input_type": "dropdown", "values": ["red", "blue", "black"], "default": "black"},
                    {"name": "broken", "input_type": "yes/no"},
                    {"name": "size", "input_type": "select"},  # a fixed list without values becomes free entry
                ],
            }
        ],
    }
    updated = spec.apply_guide(parsed, guide)
    attrs = {a["name"]: a for a in updated["labels"][0]["attributes"]}
    assert attrs["color"]["input_type"] == "select" and attrs["color"]["values"] == ["red", "blue", "black"]
    assert attrs["broken"]["input_type"] == "checkbox"
    assert attrs["size"]["input_type"] == "text"
    assert "color (select: red, blue, black)" in updated["guide"]["decisions"][0]["attributes"]
    # reading the document again changes nothing and adds nothing twice
    again = spec.apply_guide(updated, guide)
    assert [a["name"] for a in again["labels"][0]["attributes"]] == ["color", "broken", "size"]


def test_attributes_read_by_the_template_and_by_the_model_are_combined():
    from app.vision import _merge_decisions

    local = [{"name": "car", "attributes": [{"name": "moving", "input_type": "checkbox"}]}]
    found = [{"name": "car", "attributes": [{"name": "moving", "input_type": "checkbox"}, {"name": "colour", "input_type": "text"}]}]
    merged = _merge_decisions(local, found)
    assert [a["name"] for a in merged[0]["attributes"]] == ["moving", "colour"]


def test_chat_can_add_an_attribute_to_a_label_and_it_reaches_cvat(tmp_path):
    import json

    from app import chat
    from app.service import get_project
    from tests.test_analyzer import _client

    class Cvat:
        def __init__(self):
            self.labels = [
                {"id": 204, "name": "car", "type": "rectangle", "attributes": []},
                {"id": 6, "name": "legacy", "type": "rectangle", "attributes": [
                    {"id": 88, "name": "keep", "input_type": "text", "values": [], "default_value": ""},
                ]},
            ]
            self.patches = []

        def get_labels(self, project_id):
            return [dict(item) for item in self.labels]

        def patch_project_labels(self, project_id, patch):
            self.patches.append(patch)
            for change in patch:
                for label in self.labels:
                    if label["id"] == change.get("id") and change.get("attributes"):
                        deleted = {a.get("id") for a in change["attributes"] if a.get("deleted")}
                        label["attributes"] = [a for a in label["attributes"] if a.get("id") not in deleted]
                        upserts = [a for a in change["attributes"] if not a.get("deleted")]
                        for index, attr in enumerate(upserts):
                            current = next((a for a in label["attributes"] if a.get("id") == attr.get("id")), None)
                            if current:
                                current.update(attr)
                            else:
                                label["attributes"].append(dict(attr, id=100 + index))

    client, _ = _client(tmp_path)
    session = client.app.state.sessions()
    state = get_project(session, 54)
    state.spec_json = {"labels": [{"name": "car", "type": "rectangle", "attributes": []}], "output_format": "CVAT"}
    cvat = Cvat()
    reply = chat.run_tool(
        "update_requirement",
        {
            "project_id": 54,
            "add_attributes": [
                {"label": "car", "name": "color", "input_type": "select", "values": ["red", "black", "blue"], "default": "black"}
            ],
        },
        cvat,
        session,
        client.app.state.settings,
        {},
    )
    result = json.loads(reply)
    assert cvat.patches, "the attribute was never sent to CVAT"
    held = result["attributes_in_cvat"]["car"]
    assert held and held[0]["name"] == "color" and held[0]["values"] == ["red", "black", "blue"]
    assert state.spec_json["labels"][0]["attributes"][0]["name"] == "color"
    assert result["attributes_in_cvat"]["car"][0]["default_value"] == "black"
    assert result["attributes_in_cvat"]["legacy"][0]["name"] == "keep"
    removed = json.loads(
        chat.run_tool(
            "update_requirement",
            {"project_id": 54, "remove_attributes": [{"label": "car", "name": "COLOR"}]},
            cvat, session, client.app.state.settings, {},
        )
    )
    assert state.spec_json["labels"][0]["attributes"] == []
    assert removed["project_id"] == 54
    assert removed["attributes_in_cvat"]["car"] == []
    session.close()


def test_chat_apply_requirement_creates_labels_with_attributes_on_a_new_project(tmp_path):
    import json

    from app import chat
    from tests.test_analyzer import _client

    class Cvat:
        def __init__(self):
            self.labels = []
            self.patches = []

        def get_labels(self, project_id):
            return [dict(item) for item in self.labels]

        def patch_project_labels(self, project_id, patch):
            self.patches.append(patch)
            for item in patch:
                created = dict(item, id=204)
                created["attributes"] = [dict(attr, id=301 + index) for index, attr in enumerate(item.get("attributes") or [])]
                self.labels.append(created)

    client, _ = _client(tmp_path)
    session = client.app.state.sessions()
    cvat = Cvat()
    upload = {
        "document_name": "vehicle-requirement.txt",
        "text": "The car has a color attribute with values red, black, blue. Default black.",
        "visuals": [],
        "spec": {"labels": [{"name": "car", "type": "rectangle", "attributes": [
            {"name": "color", "input_type": "select", "values": ["red", "black", "blue"], "default": "black"},
        ]}]},
    }
    response = json.loads(chat.run_tool(
        "apply_requirement", {"project_id": 57}, cvat, session, client.app.state.settings, upload
    ))
    assert cvat.labels[0]["name"] == "car" and cvat.labels[0]["id"] == 204
    assert cvat.labels[0]["attributes"][0]["values"] == ["red", "black", "blue"]
    assert cvat.labels[0]["attributes"][0]["default_value"] == "black"
    assert response["attribute_sync"]["status"] == "extraction_incomplete"  # no vision key in this test environment
    assert response["attribute_sync"]["verified_attribute_count"] == 1
    session.close()


# ---------------------------------------------------------------- learning attribute values from manual annotation


def _human(label, attrs):
    return {"role": "human", "human": {"label": label, "type": "rectangle", "points": [0, 0, 40, 30], "attributes": attrs}}


def test_notes_say_how_annotators_set_each_attribute_and_reach_the_agent_next_to_a_requirement():
    from app import labelmemory
    from app.vision import build_prompt

    samples = [
        {"frame_key": f"1:{i}", "items": [_human("mirror", {"color": color, "broken": "false"})]}
        for i, color in enumerate(["red", "red", "red", "red", "blue"])
    ]
    notes = labelmemory.derive_label_notes(samples)
    text = notes["mirror"]["attribute_text"]
    assert "color is usually 'red' (4 of 5)" in text and "broken is usually 'false' (5 of 5)" in text
    varied = labelmemory.derive_label_notes(
        [{"frame_key": f"1:{i}", "items": [_human("mirror", {"color": c})]} for i, c in enumerate(["red", "blue", "red", "blue"])]
    )
    assert "color varies: 'red' 2, 'blue' 2" in varied["mirror"]["attribute_text"]

    # a requirement that already explains the label keeps its text, and still gets the habits
    labels = [{"name": "mirror", "type": "rectangle", "how_to_annotate": "From the document.",
               "attributes": [{"id": 1, "name": "color", "input_type": "select", "values": ["red", "blue"]}]}]
    labelmemory.apply_notes(labels, notes)
    assert labels[0]["how_to_annotate"] == "From the document."
    assert "color is usually 'red'" in labels[0]["learned_attributes"]
    assert "attribute_habits" in build_prompt(labels, [], 800, 600)


def test_examples_cover_each_attribute_value_and_trimming_keeps_them(tmp_path):
    from app import labelmemory
    from app.db import LabelExemplar
    from tests.test_analyzer import _client

    client, _ = _client(tmp_path)
    session = client.app.state.sessions()

    def add(index, color, size):
        session.add(LabelExemplar(
            cvat_project_id=9, label="mirror", task_id=1, job_id=1, frame=index, shape_key=f"{index}:k",
            shape_type="rectangle", role="human", points_json=[0, 0, size, size],
            attributes_json={"color": color}, box_json=[0, 0, size, size], crop=b"\xff\xd8x",
        ))

    add(0, "blue", 20)                      # the only blue one, and the oldest
    for index in range(1, 15):              # many newer red ones
        add(index, "red", 40 + index)
    session.flush()
    labelmemory._trim(session, 9, "mirror")
    left = session.query(LabelExemplar).filter_by(label="mirror").all()
    assert len(left) <= labelmemory.KEEP_PER_LABEL * 2
    assert any(row.attributes_json == {"color": "blue"} for row in left)  # not trimmed away

    chosen = labelmemory.select_exemplars(session, 9, ["mirror"])
    assert {item["attributes"]["color"] for item in chosen} == {"red", "blue"}
    session.close()
