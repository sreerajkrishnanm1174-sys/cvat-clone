"""Turn a client example or schema into a reusable profile, then convert CVAT shapes with fixed rules.

The model is not asked to write each annotation. Uploaded files are data: they are parsed, never executed.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from copy import deepcopy

import yaml

MAX_BYTES = 8_000_000
ALLOWED_SUFFIXES = {".json", ".txt", ".md", ".yaml", ".yml", ".csv", ".pdf", ".docx"}

_IMAGE_NAME = {"image_name", "file_name", "filename", "img_name", "image_file"}
_WIDTH = {"width", "image_width"}
_HEIGHT = {"height", "image_height"}
_OBJECTS = {"objects", "annotations", "shapes", "instances", "labels"}
_CLASS = {"class_name", "label", "category", "class"}
_TYPE = {"annotation_type", "shape_type", "type"}
_POLYGON = {"segmentation", "polygon", "polygons", "points", "contour"}
_BOX = {"bbox", "bounding_box", "box"}
_ATTR = {"attributes", "attrs"}
_ID = {"object_id", "annotation_id", "id"}
_KEYPOINTS = {"keypoints", "skeleton", "joints"}


def analyze_files(files: list[tuple[str, bytes]]) -> dict:
    """Read one or more client documents and return a profile. Nothing is saved."""
    if not files:
        return _empty("Upload a client example or schema.")
    if len(files) > 8:
        return _empty("Upload at most 8 documents at a time.")
    texts: list[tuple[str, str]] = []
    structured: list[tuple[str, object]] = []
    renames: dict[str, str] = {}
    for name, raw in files:
        safe = _safe_name(name)
        if len(raw) > MAX_BYTES:
            return _empty(f"{safe} is larger than 8 MB.")
        suffix = _suffix(safe)
        if suffix not in ALLOWED_SUFFIXES:
            return _empty(f"{safe} is not a supported requirement file.")
        if suffix == ".csv":
            renames.update(_csv_renames(raw))
            continue
        if suffix in {".json", ".yaml", ".yml"}:
            loaded = _load_structured(raw, suffix)
            if isinstance(loaded, str):
                return _empty(loaded)
            structured.append((safe, loaded))
            continue
        texts.append((safe, _read_text(safe, raw)))
    example = None
    example_name = ""
    schema_note = ""
    for name, data in structured:
        if _is_schema(data):
            sample = _schema_example(data)
            if sample is not None:
                example, example_name = sample, name
            else:
                schema_note = f"{name} is a schema without an example record."
            continue
        example, example_name = data, name
    if example is None:
        for name, text in texts:
            found = _json_value(text)
            if found is not None:
                example, example_name = found, name
                break
    notes = "\n".join(text for _, text in texts)
    if texts:
        best_name, best_example, best_notes, best_score = "", None, "", -1
        for name, text in texts:
            found = _json_value(text)
            if found is None:
                continue
            score = _example_score(found)
            if re.search(r"annotations/|folder structure|one json", text, flags=re.I):
                score += 3
            if score > best_score:
                best_name, best_example, best_notes, best_score = name, found, text, score
        if best_example is not None and _example_score(best_example) >= _example_score(example):
            example, example_name, notes = best_example, best_name, best_notes
    if example is None:
        message = schema_note or "No example record was found. Upload an example of one image and its objects."
        profile = _empty(message)
        profile["document_name"] = example_name or (files[0][0] if files else "")
        return profile
    profile = _profile_from_example(example, example_name, notes)
    if schema_note and example_name:
        profile["ambiguities"].append(schema_note + " The example file was used, and inferred fields are marked below.")
    if renames:
        _apply_renames(profile, renames)
    return _apply_delivery(profile, notes)


def convert_dataset(profile: dict, dataset_name: str, images: list[dict]) -> tuple[dict | list[dict], dict]:
    """Convert canonical images. Returns the document (or per-image documents) and a report."""
    report = {"processed": 0, "converted": 0, "warnings": [], "errors": [], "dropped": []}
    if not profile.get("ready"):
        report["errors"].append("The client format is not confirmed. Analyze an example and review the mapping first.")
        return {}, report
    if profile.get("layout") == "hseg":
        return _convert_hseg(profile, images)
    if profile.get("layout") == "coco_linked":
        return _convert_linked(profile, images)
    if profile.get("layout") == "persons":
        return _convert_persons(profile, images)
    warnings = list(profile.get("ambiguities") or [])
    report["warnings"].extend(warnings)
    built_images: list[dict] = []
    seen_names: set[str] = set()
    for image in images:
        report["processed"] += 1
        name = str(image.get("name") or "")
        if not name:
            report["errors"].append("An image has no file name.")
            continue
        if name in seen_names:
            report["errors"].append(f"{name} appears more than once.")
        seen_names.add(name)
        objects = []
        for index, shape in enumerate(image.get("shapes") or [], start=1):
            obj, notes, blocked = _convert_shape(profile, shape, index, image)
            report["warnings"].extend(notes)
            if blocked:
                report["errors"].append(blocked)
                report["dropped"].append({"image": name, "type": shape.get("type"), "label": shape.get("label")})
                continue
            objects.append(obj)
            report["converted"] += 1
        built_images.append(_image_record(profile, image, objects))
    report["warnings"] = list(dict.fromkeys(report["warnings"]))
    _check_counts(images, built_images, profile, report)
    if profile.get("layout") == "json_per_image":
        for record in built_images:
            _validate_objects(record, profile, report)
        return built_images, report
    document = deepcopy(profile.get("root") or {})
    _assign_path(document, profile.get("image_path") or [], built_images)
    name_path = profile.get("dataset_name_path")
    if name_path:
        _assign_path(document, name_path, dataset_name)
    _validate_document(document, profile, report)
    return document, report


def package(profile: dict, dataset_name: str, images: list[dict]) -> tuple[bytes | None, dict]:
    document, report = convert_dataset(profile, dataset_name, images)
    if report["errors"]:
        return None, report
    files = report.pop("files", [])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("conversion-report.json", json.dumps(_public_report(report, dataset_name, profile), indent=2))
        archive.writestr("client-profile.json", json.dumps(_public_profile(profile), indent=2))
        if profile.get("layout") == "hseg":
            _write_hseg(archive, profile, document, files)
        elif profile.get("layout") == "persons":
            _write_person_files(archive, profile, document)
        elif profile.get("layout") == "coco_linked":
            path = profile.get("annotation_file") or "annotations/instances.json"
            archive.writestr(path, json.dumps(document, indent=2))
        elif profile.get("layout") == "json_per_image":
            for image_doc in document:
                stem = _safe_stem(str(_image_name(image_doc, profile) or "image"))
                path = (profile.get("annotation_file") or "annotations/{stem}.json").replace("{stem}", stem)
                archive.writestr(path, json.dumps(image_doc, indent=2))
        else:
            path = profile.get("annotation_file") or "annotations.json"
            archive.writestr(path, json.dumps(document, indent=2))
    return buffer.getvalue(), report


def _profile_from_example(example: object, document_name: str, notes: str) -> dict:
    if isinstance(example, dict) and isinstance(example.get("assets"), list):
        return _profile_hseg(example, document_name, notes)
    if isinstance(example, dict) and isinstance(example.get("images"), list) and isinstance(example.get("annotations"), list):
        return _profile_linked(example, document_name)
    if isinstance(example, dict) and isinstance(example.get("image"), dict) and isinstance(example.get("persons"), list):
        return _profile_persons(example, document_name, notes)
    profile = _empty("")
    profile["document_name"] = document_name
    profile["origin"] = "inferred"
    profile["ambiguities"] = ["Field roles were inferred from the example. Review them before exporting."]
    if isinstance(example, list):
        root = {"images": example}
        image_path = ["images"]
    elif isinstance(example, dict) and _score_image(example) >= 3 and not _find_image_list(example)[1]:
        profile["layout"] = "json_per_image"
        root = {"images": [example]}
        image_path = ["images"]
        profile["ambiguities"].append("The example is one image. The export writes one JSON file per image.")
    elif isinstance(example, dict):
        root = example
        _, image_path = _find_image_list(example)
    else:
        profile["ambiguities"] = ["The example is not a JSON object or list."]
        return profile
    if not image_path:
        profile["ambiguities"] = ["No image list was found. The example needs a file name, width, height, and an object list."]
        return profile
    sample = _at_path(root, image_path)
    sample = sample[0] if isinstance(sample, list) and sample else None
    if not isinstance(sample, dict):
        profile["ambiguities"] = ["The image list does not contain an object."]
        return profile
    image_fields, object_key, object_sample, object_fields = _bind(sample)
    if not object_key or not (object_fields.get("segmentation") or object_fields.get("bbox") or object_fields.get("keypoints")):
        profile["ambiguities"] = ["The example has no polygon, box, or keypoint field to write into."]
        return profile
    profile.update(
        {
            "ready": True,
            "name": _safe_stem(document_name) or "Client format",
            "root": _blank_images(root, image_path),
            "image_path": image_path,
            "image_fields": image_fields,
            "object_key": object_key,
            "object_fields": object_fields,
            "segmentation_style": _point_style(object_sample.get(object_fields["segmentation"])) if object_fields.get("segmentation") else "",
            "bbox_style": _box_style(object_sample.get(object_fields["bbox"])) if object_fields.get("bbox") else "",
            "coordinates": "pixel",
            "id_style": "label_index" if isinstance(object_sample.get(object_fields.get("object_id") or ""), str) else "number",
            "attribute_keys": list((object_sample.get(object_fields["attributes"]) or {}).keys()) if object_fields.get("attributes") else [],
            "dataset_name_path": _dataset_name_path(root),
        }
    )
    if _mentions_normalized(notes) and not _values_look_normalized(object_sample, object_fields):
        profile["ambiguities"].append("The text asks for normalized coordinates, but the example uses pixel values. Pixel values from the example are kept.")
    elif _values_look_normalized(object_sample, object_fields):
        profile["coordinates"] = "normalized"
        profile["ambiguities"].append("Coordinates in the example are between 0 and 1, so the export normalizes them.")
    if profile["object_fields"].get("bbox"):
        profile["ambiguities"].append("Boxes are written as left, top, width, height.")
    if profile["dataset_name_path"]:
        shown = _at_path(root, profile["dataset_name_path"])
        profile["ambiguities"].append("The dataset name in the example is " + repr(shown) + ". The export uses the CVAT project name instead.")
    return profile


def _bind(sample: dict) -> tuple[dict, str, dict, dict]:
    keys = {_norm(key): key for key in sample}
    image_fields = {
        "name": _first(keys, _IMAGE_NAME),
        "width": _first(keys, _WIDTH),
        "height": _first(keys, _HEIGHT),
    }
    object_key = _first(keys, _OBJECTS) or ""
    objects = sample.get(object_key) if object_key else None
    object_sample = objects[0] if isinstance(objects, list) and objects and isinstance(objects[0], dict) else {}
    object_keys = {_norm(key): key for key in object_sample}
    object_fields = {
        "class_name": _first(object_keys, _CLASS),
        "annotation_type": _first(object_keys, _TYPE),
        "segmentation": _first(object_keys, _POLYGON),
        "bbox": _first(object_keys, _BOX),
        "attributes": _first(object_keys, _ATTR),
        "object_id": _first(object_keys, _ID),
        "keypoints": _first(object_keys, _KEYPOINTS),
    }
    return image_fields, object_key, object_sample, object_fields


def _convert_shape(profile: dict, shape: dict, index: int, image: dict) -> tuple[dict, list[str], str]:
    fields = profile["object_fields"]
    kind = str(shape.get("type") or "")
    notes: list[str] = []
    if kind == "mask":
        return {}, notes, f"A mask on {shape.get('label') or 'an object'} has no mask field in the client format."
    if kind == "skeleton" and not fields.get("keypoints"):
        return {}, notes, f"A skeleton on {shape.get('label') or 'an object'} has no keypoint field in the client format."
    obj: dict = {}
    if fields.get("object_id"):
        obj[fields["object_id"]] = f"{shape.get('label') or 'object'}_{index:03d}" if profile.get("id_style") == "label_index" else int(shape.get("id") or index)
    if fields.get("class_name"):
        obj[fields["class_name"]] = shape.get("label") or ""
    if fields.get("annotation_type"):
        obj[fields["annotation_type"]] = kind or "shape"
    points = [float(value) for value in (shape.get("points") or [])]
    width = int(image.get("width") or 0)
    height = int(image.get("height") or 0)
    located = {**shape, "width": width, "height": height}
    pairs = _scale(_pairs(points), width, height, profile.get("coordinates") == "normalized")
    if kind == "rectangle":
        pairs = _scale(_rectangle_pairs(points), width, height, profile.get("coordinates") == "normalized")
    if fields.get("bbox") and kind in {"rectangle", "polygon"}:
        flat = points if kind == "rectangle" else _bounds_flat(points)
        obj[fields["bbox"]] = _box(flat, "xywh", profile.get("coordinates") == "normalized", located)
        if kind == "polygon":
            notes.append(f"{shape.get('label') or 'A polygon'} is also written as a box.")
    if fields.get("segmentation") and kind in {"rectangle", "polygon", "points", "point"}:
        obj[fields["segmentation"]] = _encode_points(pairs, profile)
        if kind == "rectangle" and not fields.get("bbox"):
            notes.append(f"{shape.get('label') or 'A box'} is written as a four-point polygon because the client example has no box field.")
    if kind == "skeleton" and fields.get("keypoints"):
        obj[fields["keypoints"]] = _keypoints(located, profile.get("coordinates") == "normalized")
    if kind not in {"rectangle", "polygon", "points", "point", "skeleton"}:
        return {}, notes, f"A {kind or 'shape'} on {shape.get('label') or 'an object'} does not match the client example."
    if fields.get("attributes"):
        attrs = {}
        source = shape.get("attributes") or {}
        for key in profile.get("attribute_keys") or []:
            if key in source:
                attrs[key] = source[key]
            elif key == "occluded":
                attrs[key] = bool(shape.get("occluded"))
            else:
                attrs[key] = None
                notes.append(f"The client attribute {key} has no value on {shape.get('label') or 'an object'}.")
        extra = [key for key in source if key not in attrs]
        if extra:
            notes.append(f"CVAT attributes {', '.join(extra)} have no field in the client format and are listed only in this report.")
        obj[fields["attributes"]] = attrs
    return obj, notes, ""


def _image_record(profile: dict, image: dict, objects: list[dict]) -> dict:
    fields = profile["image_fields"]
    record = {}
    if fields.get("name"):
        record[fields["name"]] = image.get("name")
    if fields.get("width"):
        record[fields["width"]] = int(image.get("width") or 0)
    if fields.get("height"):
        record[fields["height"]] = int(image.get("height") or 0)
    record[profile["object_key"]] = objects
    return record


def _validate_document(document: dict, profile: dict, report: dict) -> None:
    try:
        json.loads(json.dumps(document))
    except (TypeError, ValueError):
        report["errors"].append("The converted document is not valid JSON.")
        return
    images = _at_path(document, profile.get("image_path") or [])
    if not isinstance(images, list):
        report["errors"].append("The converted document has no image list.")
        return
    for image in images:
        _validate_objects(image, profile, report)


def _validate_objects(image: dict, profile: dict, report: dict) -> None:
    fields = profile["object_fields"]
    objects = image.get(profile["object_key"]) if isinstance(image, dict) else None
    if not isinstance(objects, list):
        report["errors"].append("An image has no object list.")
        return
    for obj in objects:
        for role, key in fields.items():
            if key and role != "keypoints" and key not in obj:
                report["errors"].append(f"A converted object is missing {key}.")
        if fields.get("segmentation") and not _points_ok(obj.get(fields["segmentation"]), profile.get("segmentation_style")):
            report["errors"].append("A polygon does not match the client coordinate layout.")


def _check_counts(images: list[dict], built: list[dict], profile: dict, report: dict) -> None:
    source = sum(len(image.get("shapes") or []) for image in images)
    written = sum(len(image.get(profile["object_key"]) or []) for image in built)
    if source != written + len(report["dropped"]):
        report["errors"].append(f"Converted {written} objects from {source} source shapes, and the difference is not explained.")


def _scale(pairs: list[list[float]], width: int, height: int, normalized: bool) -> list[list[float]]:
    if not normalized or not width or not height:
        return pairs
    return [[x / width, y / height] for x, y in pairs]


def _encode_points(pairs: list[list[float]], profile: dict) -> list:
    if profile.get("segmentation_style") == "flat":
        flat: list[float] = []
        for pair in pairs:
            flat.extend(pair)
        return flat
    return pairs


def _box(points: list[float], style: str, normalized: bool, shape: dict) -> list[float]:
    if len(points) < 4:
        return []
    left, top, right, bottom = points[0], points[1], points[2], points[3]
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    width = float(shape.get("width") or 0)
    height = float(shape.get("height") or 0)
    if normalized and width and height:
        left, right = left / width, right / width
        top, bottom = top / height, bottom / height
    if style == "xyxy":
        return [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)]
    return [round(left, 2), round(top, 2), round(right - left, 2), round(bottom - top, 2)]


def _keypoints(shape: dict, normalized: bool) -> list[dict]:
    width = float(shape.get("width") or 0)
    height = float(shape.get("height") or 0)
    joints = []
    for element in shape.get("elements") or []:
        pts = element.get("points") or []
        if len(pts) < 2:
            continue
        x, y = float(pts[0]), float(pts[1])
        if normalized and width and height:
            x, y = x / width, y / height
        joints.append({"name": element.get("name") or "", "x": round(x, 2), "y": round(y, 2), "visible": not element.get("outside")})
    return joints


def _rectangle_pairs(points: list[float]) -> list[list[float]]:
    if len(points) < 4:
        return []
    left, top, right, bottom = points[:4]
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _pairs(points: list[float]) -> list[list[float]]:
    return [[float(points[index]), float(points[index + 1])] for index in range(0, len(points) - 1, 2)]


def _bounds_flat(points: list[float]) -> list[float]:
    pairs = _pairs(points)
    if not pairs:
        return []
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    return [min(xs), min(ys), max(xs), max(ys)]


def _points_ok(value, style: str) -> bool:
    if style == "flat":
        return isinstance(value, list) and len(value) >= 6 and len(value) % 2 == 0 and all(isinstance(item, (int, float)) for item in value)
    if not isinstance(value, list) or len(value) < 3:
        return False
    return all(isinstance(pair, list) and len(pair) == 2 and all(isinstance(item, (int, float)) for item in pair) for pair in value)


def _find_image_list(node: object, path: list | None = None) -> tuple[int, list]:
    path = path or []
    best = (0, [])
    if isinstance(node, dict):
        for key, value in node.items():
            score, found = _find_image_list(value, path + [key])
            if score > best[0]:
                best = (score, found)
    elif isinstance(node, list) and node and isinstance(node[0], dict):
        score = _score_image(node[0])
        if score > best[0]:
            best = (score, path)
    return best


def _score_image(item: dict) -> int:
    keys = {_norm(key) for key in item}
    score = 0
    if keys & _IMAGE_NAME:
        score += 3
    if keys & _WIDTH and keys & _HEIGHT:
        score += 2
    for value in item.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and _score_object(value[0]) >= 2:
            score += 3
    return score


def _score_object(item: dict) -> int:
    keys = {_norm(key) for key in item}
    score = 0
    if keys & _CLASS:
        score += 2
    if keys & (_POLYGON | _BOX | _KEYPOINTS):
        score += 2
    return score


def _blank_images(root: dict, image_path: list) -> dict:
    copy = deepcopy(root)
    _assign_path(copy, image_path, [])
    return copy


def _dataset_name_path(root: dict) -> list | None:
    dataset = root.get("dataset") if isinstance(root, dict) else None
    if isinstance(dataset, dict) and "name" in dataset:
        return ["dataset", "name"]
    return None


def _assign_path(node: dict, path: list, value) -> None:
    if not path:
        return
    cursor = node
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def _at_path(node, path: list):
    cursor = node
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _apply_renames(profile: dict, renames: dict[str, str]) -> None:
    if not profile.get("ready"):
        profile["ambiguities"].append("A field-mapping CSV was uploaded, but there is no example to apply it to.")
        return
    for group in ("image_fields", "object_fields"):
        for role, key in list((profile.get(group) or {}).items()):
            if key and _norm(key) in renames:
                profile[group][role] = renames[_norm(key)]
    profile["ambiguities"].append("CSV column names replaced the matching example field names.")


def _csv_renames(raw: bytes) -> dict[str, str]:
    text = raw.decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return {}
    header = [_norm(cell) for cell in rows[0]]
    if "client_field" not in header or not ({"cvat_field", "source", "field"} & set(header)):
        return {}
    source_at = next(index for index, name in enumerate(header) if name in {"cvat_field", "source", "field"})
    client_at = header.index("client_field")
    mapping = {}
    for row in rows[1:]:
        if len(row) <= max(source_at, client_at):
            continue
        mapping[_norm(row[source_at])] = row[client_at].strip()
    return {key: value for key, value in mapping.items() if key and value}


def _load_structured(raw: bytes, suffix: str):
    text = raw.decode("utf-8", errors="replace")
    try:
        if suffix == ".json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError):
        return f"The {suffix.lstrip('.')} file could not be parsed."


def _read_text(name: str, raw: bytes) -> str:
    suffix = _suffix(name)
    if suffix in {".txt", ".md"}:
        return raw.decode("utf-8", errors="replace")
    from .spec import read_document
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / _safe_name(name)
        path.write_bytes(raw)
        try:
            return read_document(path)
        except (ValueError, OSError):
            return ""


def _json_value(text: str):
    cleaned = "\n".join(
        line for line in (text or "").splitlines()
        if not re.search(r"Page \d+\s*$", line.strip()) and not line.strip().startswith("HSEG v1")
    )
    decoder = json.JSONDecoder()
    best = None
    best_score = -1
    start = 0
    while start < len(cleaned):
        positions = [item for item in (cleaned.find("{", start), cleaned.find("[", start)) if item >= 0]
        if not positions:
            break
        pos = min(positions)
        try:
            value, end = decoder.raw_decode(cleaned[pos:])
        except json.JSONDecodeError:
            start = pos + 1
            continue
        score = _example_score(value)
        if isinstance(value, dict) and score >= best_score:
            best, best_score = value, score
        start = pos + max(end, 1)
    return best


def _example_score(data: object) -> int:
    if not isinstance(data, dict):
        return 0
    score = 1
    if isinstance(data.get("assets"), list):
        score += 6
    if isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list):
        score += 6
    if isinstance(data.get("persons"), list) and isinstance(data.get("image"), dict):
        score += 6
    return score


def _profile_hseg(example: dict, document_name: str, notes: str) -> dict:
    assets = [item for item in example.get("assets") or [] if isinstance(item, dict)]
    instance = {}
    for asset in assets:
        rows = [item for item in asset.get("instances") or [] if isinstance(item, dict)]
        if rows:
            instance = rows[0]
            break
    geometry = instance.get("geometry") if isinstance(instance.get("geometry"), dict) else {}
    if not assets or not geometry.get("vertices"):
        profile = _empty("The HSEG example needs an asset and polygon vertices.")
        profile["document_name"] = document_name
        return profile
    root_name = "human_segmentation_export"
    match = re.search(r"([A-Za-z0-9_-]+)/[\s\S]{0,60}?manifest\.hseg\.json", notes or "")
    if match:
        root_name = match.group(1)
    labels = []
    project = example.get("project") if isinstance(example.get("project"), dict) else {}
    for item in project.get("label_set") or []:
        if isinstance(item, dict) and item.get("label_key"):
            labels.append({"label_key": item.get("label_key"), "display_name": item.get("display_name") or item.get("label_key")})
    root = deepcopy(example)
    root["assets"] = []
    profile = _empty("")
    profile.update(
        {
            "ready": True,
            "name": "HSEG JSON v1",
            "document_name": document_name,
            "origin": "inferred",
            "layout": "hseg",
            "container": "json",
            "root": root,
            "package_root": root_name,
            "annotation_file": f"{root_name}/manifest.hseg.json",
            "label_set": labels,
            "coordinate_space": geometry.get("coordinate_space") or "image_pixels_top_left",
            "attribute_keys": list((instance.get("attributes") or {}).keys()) if isinstance(instance.get("attributes"), dict) else [],
            "ambiguities": ["Polygon vertices come from the CVAT shapes. Each person also gets a mask PNG, which the requirement treats as the pixel mask."],
        }
    )
    return profile


def _convert_hseg(profile: dict, images: list[dict]) -> tuple[dict, dict]:
    report = {"processed": 0, "converted": 0, "warnings": list(profile.get("ambiguities") or []), "errors": [], "dropped": [], "files": []}
    document = deepcopy(profile.get("root") or {})
    document["format"] = document.get("format") or "HSEG"
    document["format_version"] = document.get("format_version") or "1.0"
    labels = [dict(item) for item in profile.get("label_set") or []]
    known = {item["label_key"] for item in labels}
    assets = []
    seen_assets = set()
    seen_instances = set()
    for image in images:
        report["processed"] += 1
        name = str(image.get("name") or "")
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if not name or not width or not height:
            report["errors"].append(f"{name or 'An image'} has no file name or size.")
            continue
        if not image.get("content"):
            report["errors"].append(f"{name} has no image file to place in images/.")
            continue
        asset_key = _safe_stem(name)
        if asset_key in seen_assets:
            report["errors"].append(f"{asset_key} is used by more than one image.")
        seen_assets.add(asset_key)
        source_file = "images/" + name.replace("\\", "/").split("/")[-1]
        report["files"].append({"path": f"{profile['package_root']}/{source_file}", "data": image["content"]})
        instances = []
        human_index = 0
        for shape in image.get("shapes") or []:
            kind = str(shape.get("type") or "")
            label = str(shape.get("label") or "human")
            if kind != "polygon":
                report["errors"].append(f"A {kind or 'shape'} on {label} is not a polygon. HSEG needs one polygon per person.")
                report["dropped"].append({"image": name, "type": kind, "label": label})
                continue
            if label not in known:
                labels.append({"label_key": label, "display_name": label[:1].upper() + label[1:]})
                known.add(label)
                report["warnings"].append(f"{label} was not in the example label set, so it was added.")
            pairs = _pairs([float(value) for value in (shape.get("points") or [])])
            if len(pairs) < 3:
                report["errors"].append(f"{label} on {name} has fewer than 3 polygon points.")
                report["dropped"].append({"image": name, "type": kind, "label": label})
                continue
            human_index += 1
            vertices, clipped = _clip_pairs(pairs, width, height)
            if clipped:
                report["warnings"].append(f"A polygon on {name} was clipped to the image edge.")
            instance_key = f"{asset_key}_{label}_{human_index:04d}"
            if instance_key in seen_instances:
                report["errors"].append(f"{instance_key} is not unique.")
            seen_instances.add(instance_key)
            mask_file = f"masks/{asset_key}/{label}_{human_index:04d}.png"
            mask, filled = _mask_png(width, height, vertices)
            if not filled:
                report["errors"].append(f"{instance_key} produced an empty mask.")
                report["dropped"].append({"image": name, "type": kind, "label": label})
                continue
            report["files"].append({"path": f"{profile['package_root']}/{mask_file}", "data": mask})
            touches_edge = _touches_edge(pairs, width, height)
            source_attrs = shape.get("attributes") or {}
            occlusion = source_attrs.get("occlusion_level")
            if occlusion not in {"none", "partial", "heavy", "unknown"}:
                occlusion = "unknown"
                report["warnings"].append(f"{instance_key} has no occlusion level. It is set to unknown.")
            instances.append(
                {
                    "instance_key": instance_key,
                    "label_key": label,
                    "geometry": {
                        "kind": "polygon",
                        "coordinate_space": profile.get("coordinate_space") or "image_pixels_top_left",
                        "vertices": vertices,
                    },
                    "mask_file": mask_file,
                    "visible_region_only": True,
                    "attributes": {
                        "occlusion_level": occlusion,
                        "truncated_by_image_edge": touches_edge,
                    },
                    "annotation_state": "review_required",
                }
            )
            report["converted"] += 1
        assets.append(
            {
                "asset_key": asset_key,
                "source_file": source_file,
                "image_size": {"width_px": width, "height_px": height},
                "instances": instances,
            }
        )
    document["assets"] = assets
    project = document.setdefault("project", {})
    if isinstance(project, dict):
        project["label_set"] = labels
    if document.get("format") != "HSEG" or str(document.get("format_version")) != "1.0":
        report["errors"].append("The manifest is not HSEG version 1.0.")
    source = sum(len(image.get("shapes") or []) for image in images)
    if source != report["converted"] + len(report["dropped"]):
        report["errors"].append("The instance count does not match the source polygons.")
    report["warnings"] = list(dict.fromkeys(report["warnings"]))
    report["warnings"].append("annotation_state is review_required because the requirement example uses that until a person is approved.")
    report["warnings"] = list(dict.fromkeys(report["warnings"]))
    return document, report


def _write_hseg(archive: zipfile.ZipFile, profile: dict, document: dict, files: list[dict]) -> None:
    archive.writestr(profile.get("annotation_file") or "manifest.hseg.json", json.dumps(document, indent=2))
    for item in files:
        archive.writestr(item["path"], item["data"])


def _clip_pairs(pairs: list[list[float]], width: int, height: int) -> tuple[list[list[float]], bool]:
    clipped = False
    vertices = []
    for x_value, y_value in pairs:
        x_clipped = min(max(x_value, 0), max(width - 1, 0))
        y_clipped = min(max(y_value, 0), max(height - 1, 0))
        if x_clipped != x_value or y_clipped != y_value:
            clipped = True
        vertices.append([round(x_clipped, 2), round(y_clipped, 2)])
    return vertices, clipped


def _touches_edge(pairs: list[list[float]], width: int, height: int) -> bool:
    for x_value, y_value in pairs:
        if x_value <= 1 or y_value <= 1 or (width and x_value >= width - 1) or (height and y_value >= height - 1):
            return True
    return False


def _mask_png(width: int, height: int, vertices: list[list[float]]) -> tuple[bytes, bool]:
    from PIL import Image, ImageDraw

    image = Image.new("L", (width, height), 0)
    if len(vertices) >= 3:
        ImageDraw.Draw(image).polygon([(point[0], point[1]) for point in vertices], fill=255)
    filled = image.getextrema()[1] > 0
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), filled


def _profile_linked(example: dict, document_name: str) -> dict:
    image_sample = example["images"][0] if example["images"] and isinstance(example["images"][0], dict) else {}
    annotations = [item for item in example["annotations"] if isinstance(item, dict)]
    if not image_sample or not annotations:
        profile = _empty("The example needs an image and at least one annotation.")
        profile["document_name"] = document_name
        return profile
    keys = set()
    attribute_keys = []
    segmentation_style = ""
    for item in annotations:
        keys.update(item.keys())
        if isinstance(item.get("attributes"), dict):
            for key in item["attributes"]:
                if key not in attribute_keys:
                    attribute_keys.append(key)
        if item.get("segmentation") is not None:
            segmentation_style = "coco" if _is_coco_segmentation(item["segmentation"]) else _point_style(item["segmentation"])
    categories = [dict(item) for item in example.get("categories") or [] if isinstance(item, dict) and item.get("name")]
    profile = _empty("")
    profile.update(
        {
            "ready": True,
            "name": _safe_stem(document_name) or "Client format",
            "document_name": document_name,
            "origin": "inferred",
            "layout": "coco_linked",
            "container": "json",
            "categories": categories,
            "annotation_keys": sorted(keys),
            "attribute_keys": attribute_keys,
            "segmentation_style": segmentation_style or "coco",
            "image_keys": list(image_sample.keys()),
            "ambiguities": [
                "The example is one JSON document with images and annotations linked by image id.",
                "Category ids follow the names in the example. A label that is not listed there is reported before export.",
            ],
        }
    )
    return profile


def _profile_persons(example: dict, document_name: str, notes: str) -> dict:
    person = example["persons"][0] if example["persons"] and isinstance(example["persons"][0], dict) else {}
    if not person:
        profile = _empty("The example has no person record.")
        profile["document_name"] = document_name
        return profile
    root = deepcopy(example)
    root["persons"] = []
    profile = _empty("")
    profile.update(
        {
            "ready": True,
            "name": _safe_stem(document_name) or "Client format",
            "document_name": document_name,
            "origin": "inferred",
            "layout": "persons",
            "container": "json",
            "root": root,
            "person_keys": list(person.keys()),
            "bbox_style": "dict" if isinstance(person.get("bbox"), dict) else "xywh",
            "keypoint_style": "dict" if isinstance(person.get("keypoints"), dict) else "list",
            "joint_keys": list(next(iter(person["keypoints"].values())).keys()) if isinstance(person.get("keypoints"), dict) and person["keypoints"] else [],
            "bbox_margin": 5 if re.search(r"5\s*px", notes or "") else 0,
            "ambiguities": ["Each image is written from the example, with the skeleton block kept and the person records replaced."],
        }
    )
    return profile


def _apply_delivery(profile: dict, notes: str) -> dict:
    if not profile.get("ready"):
        return profile
    if profile.get("layout") == "hseg":
        profile["container"] = "json"
        profile["ambiguities"].append(
            f"The requirement uses HSEG JSON. The manifest is {profile.get('annotation_file')} and each person is a polygon plus a mask PNG."
        )
        return profile
    asks_xml = bool(re.search(r"\bxml\b", notes or "", flags=re.I))
    rejects_xml = bool(re.search(r"not output[\s\S]{0,40}xml|must not output xml", notes or "", flags=re.I))
    if asks_xml and not rejects_xml and not re.search(r"\bjson\b", notes or "", flags=re.I):
        profile["container"] = "xml"
        profile["ready"] = False
        profile["ambiguities"].append("The document asks for XML, and it does not include an XML example. Export stays blocked until the XML structure is shown.")
        return profile
    profile["container"] = "json"
    named = re.search(r"annotations/[\s\S]{0,80}?([A-Za-z0-9_.-]+\.json)", notes or "")
    per_image = bool(re.search(r"one json file per image", notes or "", flags=re.I)) or profile.get("layout") == "persons"
    if named and not per_image:
        profile["annotation_file"] = f"annotations/{named.group(1)}"
    elif per_image or re.search(r"annotations/", notes or "", flags=re.I):
        profile["annotation_file"] = "annotations/{stem}.json" if per_image else (profile.get("annotation_file") or "annotations/instances.json")
    else:
        profile["annotation_file"] = profile.get("annotation_file") or "annotations.json"
    if re.search(r"index\.json", notes or "", flags=re.I):
        profile["index_file"] = "index.json"
    if re.search(r"images/", notes or "") or re.search(r"images in /images", notes or "", flags=re.I):
        profile["images_dir"] = "images"
    profile["ambiguities"].append(
        f"The requirement says the file is {profile['container'].upper()} and belongs in {profile['annotation_file']}."
    )
    return profile


def _convert_linked(profile: dict, images: list[dict]) -> tuple[dict, dict]:
    report = {"processed": 0, "converted": 0, "warnings": list(profile.get("ambiguities") or []), "errors": [], "dropped": []}
    categories = [dict(item) for item in profile.get("categories") or []]
    names = {str(item.get("name")): int(item.get("id") or index) for index, item in enumerate(categories, start=1)}
    next_id = max([int(item.get("id") or 0) for item in categories] or [0]) + 1
    image_rows = []
    annotations = []
    ann_id = 1
    keys = set(profile.get("annotation_keys") or [])
    for image_id, image in enumerate(images, start=1):
        report["processed"] += 1
        row = {}
        sample_keys = set(profile.get("image_keys") or ["id", "file_name", "width", "height"])
        if "id" in sample_keys:
            row["id"] = image_id
        if "file_name" in sample_keys or "image_name" in sample_keys:
            row["file_name" if "file_name" in sample_keys else "image_name"] = image.get("name")
        if "width" in sample_keys:
            row["width"] = int(image.get("width") or 0)
        if "height" in sample_keys:
            row["height"] = int(image.get("height") or 0)
        image_rows.append(row)
        for shape in image.get("shapes") or []:
            label = str(shape.get("label") or "")
            if label and label not in names:
                names[label] = next_id
                categories.append({"id": next_id, "name": label})
                report["warnings"].append(f"{label} is not in the example categories. It was added as id {next_id}.")
                next_id += 1
            if not label:
                report["errors"].append("An object has no label, so it has no category id.")
                report["dropped"].append({"image": image.get("name"), "type": shape.get("type"), "label": ""})
                continue
            kind = str(shape.get("type") or "")
            if kind not in {"rectangle", "polygon", "points", "point"}:
                report["errors"].append(f"A {kind or 'shape'} on {label} is not in the requirement example.")
                report["dropped"].append({"image": image.get("name"), "type": kind, "label": label})
                continue
            ann: dict = {}
            if "id" in keys:
                ann["id"] = ann_id
            if "image_id" in keys:
                ann["image_id"] = image_id
            category_id = names[label]
            if "category_id" in keys:
                ann["category_id"] = category_id
            if "class_id" in keys:
                ann["class_id"] = category_id
            points = [float(value) for value in (shape.get("points") or [])]
            box = _box(points if kind == "rectangle" else _bounds_flat(points), "xywh", False, shape)
            if "bbox" in keys and kind in {"rectangle", "polygon"}:
                ann["bbox"] = box
            if "box_xywh" in keys and kind in {"rectangle", "polygon"}:
                ann["box_xywh"] = list(box)
            if "segmentation" in keys and kind == "polygon":
                flat = [coord for pair in _pairs(points) for coord in pair]
                ann["segmentation"] = [flat] if profile.get("segmentation_style") == "coco" else _pairs(points)
            if "area" in keys:
                ann["area"] = round(_area(kind, points), 2)
            if "iscrowd" in keys:
                ann["iscrowd"] = 0
            if "attributes" in keys:
                source = shape.get("attributes") or {}
                ann["attributes"] = {key: source.get(key) for key in profile.get("attribute_keys") or []}
                missing = [key for key, value in ann["attributes"].items() if value is None]
                if missing:
                    report["warnings"].append(f"{label} has no value for {', '.join(missing)}.")
            if "track_id" in keys:
                ann["track_id"] = shape.get("track_id") or ""
                if not ann["track_id"]:
                    report["warnings"].append(f"{label} has no track id. The example includes track_id.")
            if "bbox" not in ann and "segmentation" not in ann:
                report["errors"].append(f"{label} could not be written as a box or a polygon.")
                report["dropped"].append({"image": image.get("name"), "type": kind, "label": label})
                continue
            annotations.append(ann)
            ann_id += 1
            report["converted"] += 1
    document = {"images": image_rows, "annotations": annotations}
    if categories:
        document["categories"] = categories
    _validate_linked(document, report)
    report["warnings"] = list(dict.fromkeys(report["warnings"]))
    return document, report


def _convert_persons(profile: dict, images: list[dict]) -> tuple[list[dict], dict]:
    report = {"processed": 0, "converted": 0, "warnings": list(profile.get("ambiguities") or []), "errors": [], "dropped": []}
    documents = []
    margin = int(profile.get("bbox_margin") or 0)
    for image in images:
        report["processed"] += 1
        document = deepcopy(profile.get("root") or {})
        info = document.setdefault("image", {})
        if isinstance(info, dict):
            if "file_name" in info:
                info["file_name"] = image.get("name")
            if "width" in info:
                info["width"] = int(image.get("width") or 0)
            if "height" in info:
                info["height"] = int(image.get("height") or 0)
        persons = []
        for index, shape in enumerate(image.get("shapes") or [], start=1):
            if shape.get("type") != "skeleton":
                report["errors"].append(f"A {shape.get('type') or 'shape'} on {shape.get('label') or 'an object'} is not a skeleton, and the example only shows persons.")
                report["dropped"].append({"image": image.get("name"), "type": shape.get("type"), "label": shape.get("label")})
                continue
            person = {"person_id": index}
            if "bbox" in (profile.get("person_keys") or []):
                person["bbox"] = _bbox_dict(shape, int(image.get("width") or 0), int(image.get("height") or 0), margin)
            if "keypoints" in (profile.get("person_keys") or []):
                person["keypoints"] = _keypoint_dict(shape, profile.get("joint_keys") or [])
            if "num_visible" in (profile.get("person_keys") or []):
                person["num_visible"] = sum(1 for joint in person.get("keypoints", {}).values() if joint.get("state") == "visible")
            if "attributes" in (profile.get("person_keys") or []):
                person["attributes"] = shape.get("attributes") or {}
            if not person.get("keypoints"):
                report["errors"].append(f"{shape.get('label') or 'A person'} has no keypoints.")
                report["dropped"].append({"image": image.get("name"), "type": "skeleton", "label": shape.get("label")})
                continue
            persons.append(person)
            report["converted"] += 1
        document["persons"] = persons
        if isinstance(document.get("summary"), dict):
            document["summary"]["person_count"] = len(persons)
            document["summary"]["total_visible_keypoints"] = sum(person.get("num_visible") or 0 for person in persons)
        documents.append(document)
        try:
            json.loads(json.dumps(document))
        except (TypeError, ValueError):
            report["errors"].append(f"{image.get('name')} did not produce valid JSON.")
    source = sum(len(image.get("shapes") or []) for image in images)
    if source != report["converted"] + len(report["dropped"]):
        report["errors"].append("The person count does not match the source shapes.")
    report["warnings"] = list(dict.fromkeys(report["warnings"]))
    return documents, report


def _validate_linked(document: dict, report: dict) -> None:
    try:
        json.loads(json.dumps(document))
    except (TypeError, ValueError):
        report["errors"].append("The converted document is not valid JSON.")
        return
    image_ids = {item.get("id") for item in document.get("images") or []}
    category_ids = {item.get("id") for item in document.get("categories") or []}
    for ann in document.get("annotations") or []:
        if ann.get("image_id") not in image_ids:
            report["errors"].append("An annotation points at an image that is not in the file.")
        if "category_id" in ann and category_ids and ann.get("category_id") not in category_ids:
            report["errors"].append("An annotation uses a category id that is not in the category list.")
        if "bbox" in ann and (not isinstance(ann["bbox"], list) or len(ann["bbox"]) != 4):
            report["errors"].append("A box is not four numbers in left, top, width, height order.")
        if "segmentation" in ann and not _coco_polygon_ok(ann["segmentation"]):
            report["errors"].append("A polygon does not match the example coordinate list.")


def _write_person_files(archive: zipfile.ZipFile, profile: dict, documents: list[dict]) -> None:
    index = []
    pattern = profile.get("annotation_file") or "annotations/{stem}.json"
    for document in documents:
        info = document.get("image") or {}
        stem = _safe_stem(str(info.get("file_name") or "image"))
        path = pattern.replace("{stem}", stem)
        archive.writestr(path, json.dumps(document, indent=2))
        index.append({"file": path, "person_count": len(document.get("persons") or [])})
    if profile.get("index_file"):
        archive.writestr(profile["index_file"], json.dumps({"files": index}, indent=2))


def _image_name(image_doc: dict, profile: dict) -> str:
    key = (profile.get("image_fields") or {}).get("name")
    if key and image_doc.get(key):
        return str(image_doc[key])
    info = image_doc.get("image")
    if isinstance(info, dict):
        return str(info.get("file_name") or "")
    return ""


def _is_coco_segmentation(value) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], list) and bool(value[0]) and isinstance(value[0][0], (int, float))


def _coco_polygon_ok(value) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0 and all(isinstance(item, (int, float)) for item in poly) for poly in value)


def _area(kind: str, points: list[float]) -> float:
    if kind == "rectangle" and len(points) >= 4:
        return abs((points[2] - points[0]) * (points[3] - points[1]))
    pairs = _pairs(points)
    if len(pairs) < 3:
        return 0.0
    total = 0.0
    for index, pair in enumerate(pairs):
        nxt = pairs[(index + 1) % len(pairs)]
        total += pair[0] * nxt[1] - nxt[0] * pair[1]
    return abs(total) / 2


def _bbox_dict(shape: dict, width: int, height: int, margin: int) -> dict:
    points = []
    for element in shape.get("elements") or []:
        if element.get("outside"):
            continue
        pts = element.get("points") or []
        if len(pts) >= 2:
            points.extend([float(pts[0]), float(pts[1])])
    if len(shape.get("points") or []) >= 4 and shape.get("type") == "rectangle":
        points = [float(value) for value in shape["points"]]
    bounds = _bounds_flat(points) if points else []
    if len(bounds) < 4:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    left, top, right, bottom = bounds
    left -= margin
    top -= margin
    right += margin
    bottom += margin
    if width:
        left = max(0, left)
        right = min(width, right)
    if height:
        top = max(0, top)
        bottom = min(height, bottom)
    return {"x": round(left, 2), "y": round(top, 2), "w": round(right - left, 2), "h": round(bottom - top, 2)}


def _keypoint_dict(shape: dict, joint_keys: list[str]) -> dict:
    joints = {}
    for element in shape.get("elements") or []:
        pts = element.get("points") or []
        if len(pts) < 2 or not element.get("name"):
            continue
        if element.get("outside"):
            state = "out_of_frame"
            confidence = 0.0
        elif element.get("occluded"):
            state = "occluded"
            confidence = 0.5
        else:
            state = "visible"
            confidence = 1.0
        value = {"x": round(float(pts[0]), 2), "y": round(float(pts[1]), 2)}
        if not joint_keys or "state" in joint_keys:
            value["state"] = state
        if not joint_keys or "confidence" in joint_keys:
            value["confidence"] = confidence
        joints[element["name"]] = value
    return joints


def _is_schema(data: object) -> bool:
    return isinstance(data, dict) and isinstance(data.get("properties"), dict) and (
        "$schema" in data or data.get("type") == "object"
    )


def _schema_example(data: dict):
    for key in ("example", "examples"):
        if isinstance(data.get(key), (dict, list)):
            value = data[key]
            return value[0] if isinstance(value, list) and value else value
    return None


def _mentions_normalized(notes: str) -> bool:
    return bool(re.search(r"normali[sz]ed coordinates|between 0 and 1", notes or "", flags=re.I))


def _values_look_normalized(sample: dict, fields: dict) -> bool:
    values: list[float] = []
    if fields.get("segmentation"):
        raw = sample.get(fields["segmentation"])
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, list):
                    values.extend(float(number) for number in item if isinstance(number, (int, float)))
                elif isinstance(item, (int, float)):
                    values.append(float(item))
    return bool(values) and max(values) <= 1


def _point_style(value) -> str:
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return "flat"
    return "pairs"


def _box_style(value) -> str:
    if isinstance(value, list) and len(value) == 4:
        left, top, right, bottom = value
        if right > left and bottom > top and (right - left) > bottom:
            return "xyxy"
    return "xywh"


def _public_report(report: dict, dataset_name: str, profile: dict) -> dict:
    return {
        "dataset": dataset_name,
        "profile": profile.get("name"),
        "origin": profile.get("origin"),
        "processed_images": report["processed"],
        "converted_objects": report["converted"],
        "warnings": report["warnings"],
        "errors": report["errors"],
        "unsupported": report["dropped"],
        "images_included": False,
    }


def _public_profile(profile: dict) -> dict:
    hidden = {"root"}
    return {key: value for key, value in profile.items() if key not in hidden}


def _empty(message: str) -> dict:
    return {
        "name": "Client format",
        "origin": "inferred",
        "ready": False,
        "layout": "single_json",
        "container": "json",
        "document_name": "",
        "ambiguities": [message] if message else [],
        "image_path": [],
        "image_fields": {},
        "object_key": "",
        "object_fields": {},
        "segmentation_style": "",
        "bbox_style": "",
        "coordinates": "pixel",
        "id_style": "number",
        "attribute_keys": [],
        "dataset_name_path": None,
        "root": {},
    }


def _first(keys: dict[str, str], names: set[str]) -> str | None:
    for name in names:
        if name in keys:
            return keys[name]
    return None


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _suffix(name: str) -> str:
    match = re.search(r"(\.[A-Za-z0-9]+)$", name)
    return match.group(1).lower() if match else ""


def _safe_name(name: str) -> str:
    base = (name or "requirement").replace("\\", "/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "requirement.txt"


def _safe_stem(name: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", _safe_name(name))
    return stem or "image"
