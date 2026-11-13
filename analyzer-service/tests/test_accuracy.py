"""Tiling for large images and crop verification of boxes."""

import json
import re
from io import BytesIO

from PIL import Image, ImageDraw

from app import accuracy, vision


def _jpeg(width, height, rect=None):
    picture = Image.new("RGB", (width, height), (255, 255, 255))
    if rect:
        ImageDraw.Draw(picture).rectangle(rect, fill=(0, 0, 0))
    buffer = BytesIO()
    picture.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def test_tiles_cover_the_whole_image_and_overlap():
    for width, height in ((3200, 2000), (4000, 900), (6000, 4500), (2100, 2100)):
        windows = accuracy.tile_windows(width, height)
        assert min(w[0] for w in windows) == 0 and min(w[1] for w in windows) == 0
        assert max(w[2] for w in windows) == width and max(w[3] for w in windows) == height
        assert len(windows) <= accuracy.MAX_TILES_PER_AXIS ** 2
        # every pixel column and row is inside some window
        for axis, size in ((0, width), (1, height)):
            spans = sorted({(w[axis], w[axis + 2]) for w in windows})
            reach = 0
            for start, end in spans:
                assert start <= reach
                reach = max(reach, end)
            assert reach == size
    assert accuracy.tile_windows(1200, 800) == [(0, 0, 1200, 800)]


def test_small_images_are_not_tiled():
    assert not accuracy.needs_tiles(1920, 1080, 2000)
    assert accuracy.needs_tiles(3000, 2000, 2000)
    assert accuracy.detect_tiled(_jpeg(1200, 800), lambda tile: [], 2000) == []


def test_tiled_detections_are_mapped_back_and_cut_objects_are_dropped():
    seen = []

    def detect_tile(tile_bytes):
        width, height = Image.open(BytesIO(tile_bytes)).size
        seen.append((width, height))
        return [
            {"label": "car", "type": "rectangle", "points": [100, 100, 300, 260], "score": 0.9, "attributes": {}},
            # runs into the right edge of the tile: cut in half unless that is also the image edge
            {"label": "car", "type": "rectangle", "points": [width - 120, 50, width - 1, 200], "score": 0.9, "attributes": {}},
        ]

    shapes = accuracy.detect_tiled(_jpeg(3200, 2000), detect_tile, 2000)
    windows = accuracy.tile_windows(3200, 2000)
    assert len(seen) == len(windows)
    inner = [s for s in shapes if s["points"][2] - s["points"][0] == 119]
    # only tiles that touch the image's right edge keep their edge-hugging box
    assert len(inner) == len([w for w in windows if w[2] == 3200])
    first = next(s for s in shapes if s["points"][:2] == [100.0, 100.0])
    assert first["points"] == [100.0, 100.0, 300.0, 260.0]  # first tile starts at the origin
    shifted = [s for s in shapes if s["points"][0] > 1000 and s["points"][2] - s["points"][0] == 200]
    assert shifted  # tiles further right were offset into full-image pixels


def _verifier(answers):
    """A fake model: answers[slot] is a verdict dict, or a function of the proposed box."""
    asked = {}

    def ask(system, content):
        notes = [block["text"] for block in content if block["type"] == "text"]
        asked["images"] = sum(1 for block in content if block["type"] == "image")
        results = []
        for note in notes:
            slot = int(re.search(r"Crop (\d+)", note).group(1))
            proposed = json.loads(re.search(r"Proposed box (\[[^\]]*\])", note).group(1))
            verdict = answers[slot]
            if callable(verdict):
                verdict = verdict(proposed)
            results.append({"index": slot, **verdict})
        return json.dumps({"results": results})

    ask.asked = asked
    return ask


def _shape(box, label="box"):
    return {"label": label, "type": "rectangle", "points": box, "score": 0.9, "attributes": {}}


def test_verification_tightens_boxes_and_drops_false_positives():
    image = _jpeg(1000, 800, rect=(300, 300, 600, 500))
    labels = [{"name": "box", "how_to_annotate": "black square"}]
    shapes = [
        _shape([250.0, 260.0, 650.0, 540.0]),  # loose around the real object
        _shape([700.0, 100.0, 800.0, 180.0]),  # background
        _shape([100.0, 600.0, 220.0, 700.0]),  # model answers nonsense: keep as is
        _shape([260.0, 270.0, 640.0, 530.0]),  # "tightened" onto a different object: keep as is
    ]
    ask = _verifier(
        {
            0: lambda box: {"keep": True, "box": [box[0] + 10, box[1] + 10, box[2] - 10, box[3] - 10]},
            1: {"keep": False},
            2: {"keep": True, "box": "nope"},
            3: {"keep": True, "box": [0, 0, 5, 5]},
        }
    )
    out = accuracy.verify_boxes(image, shapes, labels, ask)
    assert ask.asked["images"] == 4
    assert [s["points"] for s in out] == [
        [260.0, 270.0, 640.0, 530.0],  # the loose box, inset by 10 px in full-image pixels
        [100.0, 600.0, 220.0, 700.0],
        [260.0, 270.0, 640.0, 530.0],
    ]
    assert out[0].get("verified") is True and "verified" not in out[1]


def test_verification_failure_keeps_the_first_pass():
    shapes = [_shape([10.0, 10.0, 90.0, 90.0])]

    def broken(system, content):
        raise RuntimeError("model unavailable")

    assert accuracy.verify_boxes(_jpeg(200, 200), shapes, [{"name": "box"}], broken) == shapes
    assert accuracy.verify_boxes(_jpeg(200, 200), shapes, [{"name": "box"}], None) == shapes
    assert accuracy.verify_boxes(b"not an image", shapes, [{"name": "box"}], lambda s, c: "{}") == shapes


def test_annotate_image_verifies_by_default_and_can_be_switched_off(monkeypatch):
    image = _jpeg(1000, 800, rect=(300, 300, 600, 500))
    labels = [{"name": "box", "type": "rectangle", "detect_as": []}]
    monkeypatch.setattr("app.detect.detector_shapes", lambda *a, **k: [])
    monkeypatch.setattr(vision, "_ask_vision", lambda *a, **k: [_shape([700.0, 100.0, 800.0, 180.0])])
    monkeypatch.setattr(vision, "_make_ask", lambda key, model: (lambda system, content: json.dumps({"results": [{"index": 0, "keep": False}]})))

    assert vision.annotate_image("key", "m", image, labels, [], refine=False) == []  # false positive removed
    monkeypatch.setenv("ANNOTATION_VERIFY", "false")
    kept = vision.annotate_image("key", "m", image, labels, [], refine=False)
    assert [s["points"] for s in kept] == [[700.0, 100.0, 800.0, 180.0]]


def test_annotate_image_scans_big_images_in_tiles(monkeypatch):
    image = _jpeg(3200, 2000)
    labels = [{"name": "box", "type": "rectangle", "detect_as": []}]
    sizes = []

    def fake_vision(api_key, model, data, labs, lessons, refine=True, exemplars=None):
        sizes.append(Image.open(BytesIO(data)).size)
        return [_shape([100.0, 100.0, 220.0, 200.0])]

    monkeypatch.setattr("app.detect.detector_shapes", lambda *a, **k: [])
    monkeypatch.setattr(vision, "_ask_vision", fake_vision)
    monkeypatch.setenv("ANNOTATION_VERIFY", "false")
    shapes = vision.annotate_image("key", "m", image, labels, [], refine=False)
    assert (3200, 2000) in sizes and len(sizes) == 1 + len(accuracy.tile_windows(3200, 2000))
    assert len(shapes) == len({tuple(s["points"]) for s in shapes}) > 1  # one per tile origin, no duplicates
    monkeypatch.setenv("ANNOTATION_TILING", "false")
    sizes.clear()
    vision.annotate_image("key", "m", image, labels, [], refine=False)
    assert sizes == [(3200, 2000)]
