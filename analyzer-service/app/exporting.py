"""Turn CVAT annotations into the format named in the requirement."""

from __future__ import annotations

import io
import json
import re
import zipfile


def export_kind(output_format: str | None) -> str:
    """Return a converter name, or an empty string when CVAT's own export is enough."""
    value = re.sub(r"\s+", " ", (output_format or "").strip().lower())
    if not value or value in {"cvat", "cvat for images", "cvat 1.1", "default"}:
        return ""
    if "coco" in value:
        return "coco"
    if "pascal" in value or re.search(r"\bvoc\b", value):
        return "voc"
    if "pose" in value or "keypoint" in value or "skeleton" in value:
        return "yolo-pose"
    if "yolo" in value or "ultralytics" in value:
        return "yolo"
    return ""


def build_archive(kind: str, output_format: str, label_names: list[str], images: list[dict]) -> bytes:
    classes = _class_names(label_names, images)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "requirement-format.txt",
            f"Requirement output format: {output_format}\nConverted as: {kind}\n",
        )
        if kind == "coco":
            archive.writestr("annotations/instances.json", json.dumps(_coco(classes, images), indent=2))
        elif kind == "voc":
            for image in images:
                archive.writestr(f"Annotations/{_stem(image['name'])}.xml", _voc(image, classes))
        else:
            archive.writestr("classes.txt", "\n".join(classes) + ("\n" if classes else ""))
            pose = kind == "yolo-pose"
            order = _keypoint_order(images)
            if pose and order:
                archive.writestr("keypoints.txt", "\n".join(order) + "\n")
            for image in images:
                lines = [_yolo_line(shape, classes, image, order if pose else None) for shape in image.get("shapes") or []]
                lines = [line for line in lines if line]
                archive.writestr(f"labels/{_stem(image['name'])}.txt", "\n".join(lines) + ("\n" if lines else ""))
    return buffer.getvalue()


def _class_names(label_names: list[str], images: list[dict]) -> list[str]:
    names = [name for name in label_names if name]
    seen = set(names)
    for image in images:
        for shape in image.get("shapes") or []:
            label = shape.get("label") or ""
            if label and label not in seen:
                names.append(label)
                seen.add(label)
    return names


def _stem(name: str) -> str:
    base = (name or "image").replace("\\", "/").split("/")[-1]
    stem = re.sub(r"\.[^.]+$", "", base)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "image"


def _bounds(points: list[float]) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None
    xs = [float(value) for value in points[0::2]]
    ys = [float(value) for value in points[1::2]]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _shape_bounds(shape: dict) -> tuple[float, float, float, float] | None:
    kind = shape.get("type")
    if kind == "skeleton":
        points = []
        for element in shape.get("elements") or []:
            if element.get("outside"):
                continue
            points.extend(element.get("points") or [])
        return _bounds(points)
    return _bounds(shape.get("points") or [])


def _norm(value: float, size: float) -> float:
    if size <= 0:
        return 0.0
    return min(1.0, max(0.0, value / size))


def _yolo_line(shape: dict, classes: list[str], image: dict, keypoints: list[str] | None) -> str:
    label = shape.get("label") or ""
    if label not in classes:
        return ""
    bounds = _shape_bounds(shape)
    width = float(image.get("width") or 0)
    height = float(image.get("height") or 0)
    if not bounds or width <= 0 or height <= 0:
        return ""
    left, top, right, bottom = bounds
    cx = _norm((left + right) / 2, width)
    cy = _norm((top + bottom) / 2, height)
    box_w = _norm(right - left, width)
    box_h = _norm(bottom - top, height)
    class_id = classes.index(label)
    numbers = [f"{class_id}", f"{cx:.6f}", f"{cy:.6f}", f"{box_w:.6f}", f"{box_h:.6f}"]
    if keypoints is not None:
        by_name = {element.get("name"): element for element in shape.get("elements") or []}
        for name in keypoints:
            element = by_name.get(name) or {}
            points = element.get("points") or [0, 0]
            x = float(points[0]) if points else 0.0
            y = float(points[1]) if len(points) > 1 else 0.0
            if element.get("outside") or not element:
                numbers.extend(["0.000000", "0.000000", "0"])
            else:
                visible = "1" if element.get("occluded") else "2"
                numbers.extend([f"{_norm(x, width):.6f}", f"{_norm(y, height):.6f}", visible])
    elif shape.get("type") == "polygon":
        points = shape.get("points") or []
        for index, value in enumerate(points):
            size = width if index % 2 == 0 else height
            numbers.append(f"{_norm(float(value), size):.6f}")
    return " ".join(numbers)


def _keypoint_order(images: list[dict]) -> list[str]:
    order = []
    seen = set()
    for image in images:
        for shape in image.get("shapes") or []:
            for name in shape.get("keypoint_order") or []:
                if name not in seen:
                    order.append(name)
                    seen.add(name)
            for element in shape.get("elements") or []:
                name = element.get("name")
                if name and name not in seen:
                    order.append(name)
                    seen.add(name)
    return order


def _coco(classes: list[str], images: list[dict]) -> dict:
    coco_images = []
    annotations = []
    keypoints = _keypoint_order(images)
    categories = []
    for index, name in enumerate(classes, start=1):
        item = {"id": index, "name": name}
        if keypoints:
            item["keypoints"] = keypoints
            item["skeleton"] = []
        categories.append(item)
    annotation_id = 1
    for image_id, image in enumerate(images, start=1):
        coco_images.append(
            {
                "id": image_id,
                "file_name": image.get("name") or f"{image_id}.jpg",
                "width": int(image.get("width") or 0),
                "height": int(image.get("height") or 0),
            }
        )
        for shape in image.get("shapes") or []:
            label = shape.get("label") or ""
            if label not in classes:
                continue
            bounds = _shape_bounds(shape)
            if not bounds:
                continue
            left, top, right, bottom = bounds
            box_w = max(0.0, right - left)
            box_h = max(0.0, bottom - top)
            item = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": classes.index(label) + 1,
                "bbox": [round(left, 2), round(top, 2), round(box_w, 2), round(box_h, 2)],
                "area": round(box_w * box_h, 2),
                "iscrowd": 0,
            }
            if shape.get("type") == "polygon" and shape.get("points"):
                item["segmentation"] = [[round(float(value), 2) for value in shape["points"]]]
            if shape.get("elements"):
                flat = []
                by_name = {element.get("name"): element for element in shape["elements"]}
                names = keypoints or [element.get("name") for element in shape["elements"]]
                visible = 0
                for name in names:
                    element = by_name.get(name) or {}
                    points = element.get("points") or [0, 0]
                    if element.get("outside") or not element:
                        flat.extend([0, 0, 0])
                    else:
                        flag = 1 if element.get("occluded") else 2
                        visible += 1
                        flat.extend([round(float(points[0]), 2), round(float(points[1]), 2) if len(points) > 1 else 0, flag])
                item["keypoints"] = flat
                item["num_keypoints"] = visible
            annotations.append(item)
            annotation_id += 1
    return {"images": coco_images, "annotations": annotations, "categories": categories}


def _voc(image: dict, classes: list[str]) -> str:
    parts = [
        "<annotation>",
        f"<filename>{_xml(image.get('name') or 'image.jpg')}</filename>",
        "<size>",
        f"<width>{int(image.get('width') or 0)}</width>",
        f"<height>{int(image.get('height') or 0)}</height>",
        "<depth>3</depth>",
        "</size>",
    ]
    for shape in image.get("shapes") or []:
        label = shape.get("label") or ""
        if label not in classes:
            continue
        bounds = _shape_bounds(shape)
        if not bounds:
            continue
        left, top, right, bottom = bounds
        parts.extend(
            [
                "<object>",
                f"<name>{_xml(label)}</name>",
                "<bndbox>",
                f"<xmin>{int(round(left))}</xmin>",
                f"<ymin>{int(round(top))}</ymin>",
                f"<xmax>{int(round(right))}</xmax>",
                f"<ymax>{int(round(bottom))}</ymax>",
                "</bndbox>",
                "</object>",
            ]
        )
    parts.append("</annotation>")
    return "\n".join(parts) + "\n"


def _xml(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
