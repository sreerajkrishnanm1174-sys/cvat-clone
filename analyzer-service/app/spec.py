"""Requirement documents become an annotation spec and a CVAT label diff."""

from __future__ import annotations

import json
import re
from pathlib import Path

TYPE_MAP = {
    "rectangle": "rectangle",
    "box": "rectangle",
    "bbox": "rectangle",
    "bounding_box": "rectangle",
    "boundingbox": "rectangle",
    "polygon": "polygon",
    "segmentation": "polygon",
    "mask": "mask",
    "points": "points",
    "keypoint": "points",
    "keypoints": "points",
    "skeleton": "skeleton",
    "polyline": "polyline",
    "line": "polyline",
    "tag": "tag",
    "ellipse": "ellipse",
    "cuboid": "cuboid",
    "interval": "interval",
    "pose": "skeleton",
    "pose-skeleton": "skeleton",
}

CVAT_TYPES = {
    "rectangle",
    "polygon",
    "polyline",
    "points",
    "ellipse",
    "cuboid",
    "mask",
    "skeleton",
    "tag",
    "interval",
}

HUMAN_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Bones for a person. Used only when the document lists these joints and does not name its own connections.
BODY_EDGES = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

BODY_LAYOUT = {
    "nose": (50, 14),
    "left_eye": (42, 12),
    "right_eye": (58, 12),
    "left_ear": (34, 16),
    "right_ear": (66, 16),
    "neck": (50, 28),
    "left_shoulder": (32, 36),
    "right_shoulder": (68, 36),
    "left_elbow": (22, 58),
    "right_elbow": (78, 58),
    "left_wrist": (16, 78),
    "right_wrist": (84, 78),
    "left_hip": (40, 78),
    "right_hip": (60, 78),
    "left_knee": (38, 104),
    "right_knee": (62, 104),
    "left_ankle": (36, 128),
    "right_ankle": (64, 128),
}

INPUT_TYPES = {"select", "radio", "checkbox", "number", "text"}


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        import docx

        document = docx.Document(str(path))
        chunks = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    chunks.append(" | ".join(dict.fromkeys(cells)))
        return "\n".join(chunks)
    raise ValueError(f"Unsupported document type {suffix or '(none)'}")


def _json_block(text: str) -> dict | None:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if not match:
        return None
    return json.loads(match.group(1))


def _block_value(block: str, key: str) -> str | None:
    match = re.search(rf"^{key}\s*:\s*(.+)$", block, flags=re.I | re.M)
    return match.group(1).strip() if match else None


COCO_NAMES = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorbike",
    "motorbike": "motorbike",
}


def _object_at(text: str, start: int) -> dict | None:
    depth = 0
    end = None
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        return None
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _example_annotations(text: str) -> dict:
    """Read a COCO sample even when it is pretty-printed and nested objects come first."""
    marker = text.find('"categories"')
    data = None
    while marker > 0:
        start = text.rfind("{", 0, marker)
        if start < 0:
            break
        candidate = _object_at(text, start)
        if candidate and candidate.get("categories") and candidate.get("annotations"):
            data = candidate
            break
        marker = start
    if not data:
        return {}
    categories = {item.get("id"): item.get("name") for item in data.get("categories") or []}
    found: dict[str, dict] = {}
    for annotation in data.get("annotations") or []:
        name = categories.get(annotation.get("category_id"))
        if not name:
            continue
        slot = found.setdefault(name, {"type": None, "attributes": {}})
        if annotation.get("segmentation"):
            slot["type"] = "polygon"
        elif annotation.get("bbox") and slot["type"] != "polygon":
            slot["type"] = "rectangle"
        for key, value in (annotation.get("attributes") or {}).items():
            slot["attributes"][key] = "" if value is None else str(value)
    return found


def _shape_near(name: str, text: str) -> str | None:
    votes = {"rectangle": 0, "polygon": 0}
    for match in re.finditer(re.escape(name.lower()), text.lower()):
        window = text.lower()[max(0, match.start() - 90) : match.end() + 90]
        if "polygon" in window or "segmentation" in window:
            votes["polygon"] += 1
        if "bounding-box" in window or "bounding box" in window or re.search(r"\bbox\b", window):
            votes["rectangle"] += 1
    if votes["polygon"] > votes["rectangle"]:
        return "polygon"
    if votes["rectangle"] > votes["polygon"]:
        return "rectangle"
    return None


def _own_section(block: str) -> str:
    """Keep the label's own rules. Later chapters list other classes and sample files."""
    match = re.search(r"\n\d+\.\s", block)
    return block[: match.start()] if match else block


def _explicit_shape(name: str, text: str) -> str | None:
    match = re.search(
        rf"the\s+{re.escape(name)}\s+row\s+uses\s+the\s+([^\n.]{{0,80}})",
        text,
        flags=re.I,
    )
    if not match:
        return None
    kind = match.group(1).lower()
    if "polygon" in kind or "segmentation" in kind:
        return "polygon"
    if "box" in kind:
        return "rectangle"
    return None


def _section_shape(name: str, section: str) -> str | None:
    lowered = section.lower()
    if "polygon" in lowered or "segmentation" in lowered:
        return "polygon"
    if "draw the box" in lowered or re.search(r"\bbounding[- ]box\b", lowered):
        return "rectangle"
    if "defect" in name.lower() and re.search(r"defects are outlined as polygons", lowered):
        return "polygon"
    return None


def _detector_names(block: str) -> list[str]:
    found = []
    for word, coco_name in COCO_NAMES.items():
        if re.search(rf"\b{word}\b", block, flags=re.I) and coco_name not in found:
            found.append(coco_name)
    return found


def _keyword_shape(section: str) -> str | None:
    lowered = section.lower()
    if re.search(r"\b(skeleton|human pose|body pose|body joints)\b", lowered):
        return "skeleton"
    if re.search(r"\b(keypoint|keypoints|landmark|landmarks)\b", lowered):
        return "points"
    if re.search(r"\b(cuboid|3d box)\b", lowered):
        return "cuboid"
    if "ellipse" in lowered:
        return "ellipse"
    if re.search(r"\bmask\b", lowered):
        return "mask"
    if re.search(r"\b(polyline|centerline)\b", lowered):
        return "polyline"
    if re.search(r"\btag\b", lowered):
        return "tag"
    if re.search(r"\binterval\b", lowered):
        return "interval"
    return None


def _keypoint_names(section: str) -> list[str]:
    match = re.search(r"(?:keypoints|joints|landmarks)\s*:\s*([^\n]+)", section, flags=re.I)
    if not match:
        return []
    return [item.strip(" .") for item in re.split(r"[,;/|]", match.group(1)) if item.strip(" .")]


def _joint_key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _connection_lines(text: str) -> list[str]:
    lines = []
    collecting = False
    for raw in text.splitlines():
        line = raw.strip()
        if re.match(r"(?:skeleton\s+)?(?:connections?|bones?|links?|edges?)\b", line, flags=re.I):
            collecting = True
            if ":" in line:
                tail = line.split(":", 1)[1].strip()
                if tail:
                    lines.append(tail)
            continue
        if not collecting:
            continue
        if not line or re.match(r"^(?:Label|Type|Attribute|Output|Export|Annotation|Min size)\b", line, flags=re.I):
            collecting = False
            continue
        lines.append(line)
    return lines


def _one_pair(piece: str) -> tuple[str, str] | None:
    piece = piece.strip(" .;|")
    piece = re.sub(r"^\d+[\).\]]\s*", "", piece)
    for separator in (" to ", " -> ", " → ", " — ", " – ", " - "):
        if separator in piece:
            left, right = piece.split(separator, 1)
            left, right = left.strip(), right.strip()
            if left and right:
                return left, right
    if piece.count("-") == 1:
        left, right = piece.split("-", 1)
        left, right = left.strip(), right.strip()
        if left and right and not left.isdigit() and not right.isdigit():
            return left, right
    numbers = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", piece)
    if numbers:
        return numbers.group(1), numbers.group(2)
    cells = [cell.strip() for cell in piece.split("|") if cell.strip()]
    if len(cells) == 2:
        return cells[0], cells[1]
    return None


def skeleton_edges(text: str, names: list[str]) -> list[list[str]]:
    """Connections named in the requirement. A joint is not tied to the next name in the list."""
    canonical = {_joint_key(name): name for name in names}
    edges = []
    seen = set()
    for line in _connection_lines(text):
        pieces = re.split(r"[,;]", line) if "," in line or ";" in line else [line]
        for piece in pieces:
            pair = _one_pair(piece)
            if not pair:
                continue
            left, right = pair
            if left.isdigit() and right.isdigit():
                left_index, right_index = int(left) - 1, int(right) - 1
                if not (0 <= left_index < len(names) and 0 <= right_index < len(names)):
                    continue
                resolved = (names[left_index], names[right_index])
            else:
                resolved_left = canonical.get(_joint_key(left))
                resolved_right = canonical.get(_joint_key(right))
                if not resolved_left or not resolved_right:
                    continue
                resolved = (resolved_left, resolved_right)
            key = tuple(sorted(resolved))
            if key in seen or resolved[0] == resolved[1]:
                continue
            seen.add(key)
            edges.append([resolved[0], resolved[1]])
    if edges:
        return edges
    if _body_label("", text):
        present = set(canonical)
        return [
            [canonical[_joint_key(left)], canonical[_joint_key(right)]]
            for left, right in BODY_EDGES
            if _joint_key(left) in present and _joint_key(right) in present
        ]
    return []


def skeleton_svg(names: list[str], edges: list | None = None) -> str:
    """Draw only the bones the requirement names. Node ids follow the keypoint order."""
    parts = []
    spots = {}
    spare = 0
    for index, name in enumerate(names):
        node = index + 1
        x, y = BODY_LAYOUT.get(_joint_key(name), (20 + (spare % 4) * 28, 150 + (spare // 4) * 24))
        if _joint_key(name) not in BODY_LAYOUT:
            spare += 1
        spots[name] = (x, y, node)
        parts.append(
            f'<circle r="1.5" stroke="black" fill="#b3b3b3" cx="{x}" cy="{y}" stroke-width="0.1" '
            f'data-type="element node" data-element-id="{node}" data-node-id="{node}" data-label-name="{name}"></circle>'
        )
    for pair in edges or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        start = spots.get(str(pair[0]))
        end = spots.get(str(pair[1]))
        if not start or not end:
            continue
        x1, y1, node_from = start
        x2, y2, node_to = end
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" data-type="edge" '
            f'data-node-from="{node_from}" stroke-width="0.5" data-node-to="{node_to}"></line>'
        )
    return "".join(parts)


def _body_label(name: str, section: str) -> bool:
    text = f"{name} {section}".lower()
    return bool(re.search(r"\b(person|human|body|pedestrian|pose|skeleton)\b", text))


def _wants_tracking(text: str) -> bool:
    return bool(re.search(
        r"\b(tracking|tracks|track the|track each|track this|track across|object track|same id|interpolation)\b",
        text,
        flags=re.I,
    ))


def _minimum_size(text: str) -> int | None:
    match = re.search(r"smaller than\s+(\d+)\s*px", text, flags=re.I)
    return int(match.group(1)) if match else None


def named_format(text: str) -> str | None:
    """The output format named in a requirement document or a chat instruction."""
    heading = re.search(
        r"(?:output|export|annotation|client|delivery)\s+format\s*[:\-]\s*([^\n]+)",
        text or "",
        flags=re.I,
    )
    if heading:
        return heading.group(1).strip().rstrip(".")
    sentence = re.search(
        r"(?:export|save|convert|deliver|provide|output|download)\s+(?:the\s+)?(?:annotations?\s+)?(?:in|as|to)\s+([A-Za-z0-9][A-Za-z0-9 ._+-]{1,48})",
        text or "",
        flags=re.I,
    )
    if sentence:
        return sentence.group(1).strip().rstrip(".")
    known = re.search(r"\b(COCO|YOLO(?:\s*v?\d+)?(?:\s*pose)?|Pascal\s*VOC|\bVOC\b|LabelMe|KITTI|MOT|CSV)\b", text or "", flags=re.I)
    if known and re.search(r"\b(format|export|output|download)\b", text or "", flags=re.I):
        return known.group(1).strip()
    return None


def extract_spec(text: str) -> dict:
    embedded = _json_block(text)
    if embedded and embedded.get("labels"):
        return _normalize(embedded)
    labels = []
    questions = []
    output_format = named_format(text) or "CVAT"
    examples = _example_annotations(text)
    shared_min_size = _minimum_size(text)
    blocks = re.split(r"\n(?=Label\s*:)", text, flags=re.I)
    for block in blocks:
        name = _block_value(block, "Label")
        if not name:
            continue
        section = _own_section(block)
        raw_type = _block_value(section, "Type")
        mapped = TYPE_MAP.get((raw_type or "").lower()) if raw_type else None
        type_locked = mapped in CVAT_TYPES
        if mapped is None:
            mapped = (examples.get(name) or {}).get("type")
            type_locked = mapped in CVAT_TYPES
        if mapped is None:
            explicit = _explicit_shape(name, text)
            mapped = explicit or _section_shape(name, section) or _keyword_shape(section) or _shape_near(name, section)
            type_locked = explicit in CVAT_TYPES
        keypoints = _keypoint_names(section)
        if mapped == "skeleton" and not keypoints and _body_label(name, section):
            keypoints = list(HUMAN_KEYPOINTS)
        edges = skeleton_edges(section, keypoints) if mapped == "skeleton" else []
        attributes = []
        for attr_name, input_type, values, default, required in re.findall(
            r"Attribute\s*:\s*(.+)\nInput\s*:\s*(.+)\nValues\s*:\s*(.*)\nDefault\s*:\s*(.*)\nRequired\s*:\s*(.+)",
            block,
            flags=re.I,
        ):
            input_name = input_type.strip().lower()
            value_list = [item.strip() for item in values.split(",") if item.strip()]
            if input_name not in INPUT_TYPES:
                questions.append(
                    {
                        "id": f"{name}:{attr_name.strip()}:input",
                        "text": f"What kind of field is {attr_name.strip()} on {name}?",
                    }
                )
                input_name = "text"
            if input_name in {"select", "radio"} and not value_list:
                questions.append(
                    {
                        "id": f"{name}:{attr_name.strip()}:values",
                        "text": f"Which values are allowed for {attr_name.strip()} on {name}?",
                    }
                )
            attributes.append(
                {
                    "name": attr_name.strip(),
                    "input_type": input_name,
                    "values": value_list,
                    "default": default.strip(),
                    "required": required.strip().lower() in {"yes", "true", "required"},
                }
            )
        known = {item["name"] for item in attributes}
        for attr_name, sample in ((examples.get(name) or {}).get("attributes") or {}).items():
            if attr_name not in known:
                attributes.append(
                    {
                        "name": attr_name,
                        "input_type": "text",
                        "values": [],
                        "default": sample,
                        "required": False,
                    }
                )
        min_size = _block_value(block, "Min size")
        if mapped is None:
            questions.append(
                {
                    "id": f"{name}:type",
                    "text": f"How should {name} be drawn: box, polygon, mask, points, line, or tag?",
                }
            )
        labels.append(
            {
                "name": name,
                "type": mapped or "any",
                "attributes": attributes,
                "min_size": int(float(min_size)) if min_size else shared_min_size,
                "special_rules": [],
                "detect_as": _detector_names(section),
                "type_locked": type_locked,
                "keypoints": keypoints,
                "edges": edges,
            }
        )
    if not labels:
        labels.extend(_labels_from_skeleton_properties(text))
    if not labels:
        questions.append(
            {
                "id": "labels",
                "text": "Which labels should be created, and how should each one be drawn?",
            }
        )
    return {
        "labels": labels,
        "output_format": output_format,
        "questions": questions,
        "tracking": _wants_tracking(text),
    }


def _normalize(spec: dict) -> dict:
    labels = []
    for item in spec.get("labels") or []:
        labels.append(
            {
                "name": item["name"],
                "type": TYPE_MAP.get(str(item.get("type", "any")).lower(), item.get("type") or "any"),
                "attributes": item.get("attributes") or [],
                "min_size": item.get("min_size"),
                "special_rules": item.get("special_rules") or [],
                "detect_as": item.get("detect_as") or [],
                "type_locked": bool(item.get("type_locked")),
                "keypoints": list(item.get("keypoints") or []),
                "edges": [list(pair) for pair in item.get("edges") or [] if isinstance(pair, (list, tuple)) and len(pair) == 2],
            }
        )
    return {
        "labels": labels,
        "output_format": spec.get("output_format") or "CVAT",
        "questions": spec.get("questions") or [],
        "tracking": bool(spec.get("tracking")),
    }


DRAWING = {
    "rectangle": "Draw a tight box around the whole visible object. The edges meet the outermost pixels.",
    "polygon": "Click around the visible boundary. Add a point wherever the edge changes direction, usually 12 to 40 points.",
    "mask": "Paint every visible pixel of the object.",
    "points": "Place one point on each landmark the requirement names.",
    "polyline": "Draw a line along the object. Follow bends instead of using one straight stroke.",
    "tag": "Add the tag to the image. Do not draw a shape.",
    "ellipse": "Draw an ellipse around the object. Fit the center and both radii to the visible region.",
    "cuboid": "Draw the cuboid so the front and the depth match the object.",
    "skeleton": "Use the skeleton tool. Place every keypoint on its landmark and keep the bones connected.",
    "interval": "Mark the time span on the timeline. Do not draw a shape on the image.",
    "any": "The requirement does not say which tool to use yet.",
}


def _labels_from_skeleton_properties(text: str) -> list[dict]:
    """A requirement can describe a skeleton by its properties instead of a Label block."""
    if not re.search(r"\bskeleton\b", text, flags=re.I):
        return []
    keypoints = _keypoint_names(text)
    if not keypoints and _body_label("", text):
        keypoints = list(HUMAN_KEYPOINTS)
    if not keypoints:
        return []
    edges = skeleton_edges(text, keypoints)
    name = "person" if re.search(r"\b(person|human|pedestrian)\b", text, flags=re.I) else "skeleton"
    named = re.search(
        r"(?:skeleton|pose)(?:\s+properties)?(?:\s+for|\s+of|:)\s*([A-Za-z][\w\- ]{0,40})",
        text,
        flags=re.I,
    )
    if named:
        candidate = named.group(1).strip(" .:")
        if candidate and candidate.lower() not in {"properties", "the", "a"}:
            name = candidate.split("\n", 1)[0].strip()
    return [
        {
            "name": name,
            "type": "skeleton",
            "attributes": [],
            "min_size": _minimum_size(text),
            "special_rules": [],
            "detect_as": [],
            "type_locked": False,
            "keypoints": keypoints,
            "edges": edges,
        }
    ]


def explain_requirement(spec: dict) -> dict:
    """Plain-language annotation guide taken from the requirement labels."""
    decisions = []
    for label in spec.get("labels") or []:
        kind = label.get("type") or "any"
        attributes = label.get("attributes") or []
        attribute_text = ", ".join(
            item["name"] + (f" ({', '.join(item.get('values') or [])})" if item.get("values") else "")
            for item in attributes
        ) or "none"
        decide = f"Mark it only when it is a {label['name']}"
        if label.get("min_size"):
            decide += f" and at least {label['min_size']} px on its shorter side"
        decide += f". Then set {attribute_text}."
        if label.get("detect_as"):
            decide += " Examples named in the requirement: " + ", ".join(label["detect_as"]) + "."
        keypoints = list(label.get("keypoints") or [])
        edges = label.get("edges") or []
        if kind == "skeleton" and keypoints:
            decide += " Place these keypoints: " + ", ".join(keypoints) + "."
        if kind == "skeleton" and edges:
            decide += " Connect only these bones: " + ", ".join(f"{pair[0]}-{pair[1]}" for pair in edges) + "."
        if spec.get("tracking"):
            decide += " Track it across frames and keep the same id."
        why = {
            "rectangle": "The requirement asks for the object's extent, so a box is enough.",
            "polygon": "The requirement asks for the object's outline, so the boundary has to be traced.",
            "mask": "The requirement asks for the filled region.",
            "points": "The requirement asks for landmarks, not a region.",
            "polyline": "The requirement asks for a line.",
            "tag": "The requirement asks for a classification, not a shape.",
            "ellipse": "The requirement asks for an ellipse.",
            "cuboid": "The requirement asks for a 3D box.",
            "skeleton": "The requirement asks for these keypoints and only the bones it lists.",
            "interval": "The requirement asks for a time span.",
            "any": "The drawing tool is still undecided.",
        }.get(kind, "Use the tool named in the requirement.")
        decisions.append(
            {
                "name": label["name"],
                "type": kind,
                "why_this_type": why,
                "how_to_annotate": DRAWING.get(kind, DRAWING["any"]),
                "how_to_decide": decide,
                "attributes": attribute_text,
                "keypoints": keypoints,
                "edges": [list(pair) for pair in (label.get("edges") or [])],
            }
        )
    names = ", ".join(item["name"] for item in decisions) or "no labels yet"
    workflow = "On each image, mark every required label, set its attributes, and skip anything smaller than the minimum size."
    if spec.get("tracking"):
        workflow = "Track each object across frames with one id. " + workflow
    return {
        "summary": f"This requirement uses {names}. Add only the annotation types the document asks for.",
        "decisions": decisions,
        "workflow": workflow,
        "tracking": bool(spec.get("tracking")),
    }


def _clean_edges(edges, names: list) -> list[list[str]]:
    allowed = {str(name) for name in names}
    cleaned = []
    seen = set()
    for pair in edges or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left, right = str(pair[0]).strip(), str(pair[1]).strip()
        if not left or not right or left == right:
            continue
        if allowed and (left not in allowed or right not in allowed):
            continue
        key = tuple(sorted((left, right)))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append([left, right])
    return cleaned


def apply_guide(spec: dict, guide: dict) -> dict:
    updated = json.loads(json.dumps(spec))
    by_name = {item.get("name"): item for item in guide.get("decisions") or [] if item.get("name")}
    present = {label["name"] for label in updated.get("labels") or []}
    for name, decision in by_name.items():
        if name in present:
            continue
        suggested = TYPE_MAP.get(str(decision.get("type") or "").lower())
        if suggested not in CVAT_TYPES:
            continue
        keypoints = decision.get("keypoints") if isinstance(decision.get("keypoints"), list) else []
        edges = _clean_edges(decision.get("edges"), keypoints)
        updated.setdefault("labels", []).append(
            {
                "name": name,
                "type": suggested,
                "attributes": [],
                "min_size": None,
                "special_rules": [],
                "detect_as": [],
                "type_locked": False,
                "keypoints": [str(item).strip() for item in keypoints if str(item).strip()],
                "edges": edges,
            }
        )
        present.add(name)
    for label in updated.get("labels") or []:
        decision = by_name.get(label["name"])
        if not decision:
            continue
        suggested = TYPE_MAP.get(str(decision.get("type") or "").lower())
        if suggested in CVAT_TYPES and not label.get("type_locked") and label.get("type") in {None, "any"}:
            label["type"] = suggested
        keypoints = decision.get("keypoints") if isinstance(decision.get("keypoints"), list) else None
        if keypoints:
            label["keypoints"] = [str(item).strip() for item in keypoints if str(item).strip()]
        edges = _clean_edges(decision.get("edges"), label.get("keypoints") or [])
        if edges:
            label["edges"] = edges
        if label.get("type") == "skeleton" and not label.get("keypoints") and _body_label(label["name"], ""):
            label["keypoints"] = list(HUMAN_KEYPOINTS)
        label["why_this_type"] = decision.get("why_this_type") or ""
        label["how_to_annotate"] = decision.get("how_to_annotate") or ""
        label["how_to_decide"] = decision.get("how_to_decide") or ""
    updated["guide"] = {
        "summary": guide.get("summary") or "",
        "workflow": guide.get("workflow") or "",
        "decisions": [
            {
                "name": label["name"],
                "type": label.get("type"),
                "why_this_type": label.get("why_this_type") or "",
                "how_to_annotate": label.get("how_to_annotate") or "",
                "how_to_decide": label.get("how_to_decide") or "",
                "attributes": (by_name.get(label["name"]) or {}).get("attributes") or "",
                "keypoints": list(label.get("keypoints") or []),
            }
            for label in updated.get("labels") or []
        ],
        "tracking": bool(guide.get("tracking", updated.get("tracking"))),
    }
    if "tracking" in guide:
        updated["tracking"] = bool(guide.get("tracking"))
    return updated


def apply_answer(spec: dict, question_id: str, answer: str) -> dict:
    updated = json.loads(json.dumps(spec))
    label_name, _, field = (question_id + "::").split(":", 2)
    field = field.rstrip(":")
    accepted = False
    if field == "type":
        mapped = TYPE_MAP.get((answer or "").strip().lower())
        if mapped is None:
            raise ValueError("Say box, polygon, mask, points, line, or tag.")
        for label in updated["labels"]:
            if label["name"] == label_name:
                label["type"] = mapped
                accepted = True
    else:
        accepted = True
    if not accepted:
        raise ValueError("That answer does not match a label in the requirement.")
    updated["questions"] = [
        question for question in updated.get("questions") or [] if question.get("id") != question_id
    ]
    return updated


def diff_labels(spec: dict, existing: list[dict]) -> dict:
    by_name = {item["name"]: item for item in existing}
    create, update, unchanged, blocked = [], [], [], []
    for label in spec.get("labels") or []:
        current = by_name.get(label["name"])
        if current is None:
            if label.get("type") in {None, "any"}:
                blocked.append(
                    {
                        "name": label["name"],
                        "reason": "The requirement does not say how this label is drawn yet.",
                    }
                )
            else:
                create.append(label)
            continue
        type_change = label.get("type") not in {None, "any", current.get("type")}
        current_attrs = {attr["name"]: attr for attr in current.get("attributes") or []}
        attr_changes = []
        for attr in label.get("attributes") or []:
            found = current_attrs.get(attr["name"])
            if found is None:
                attr_changes.append({"action": "add", **attr})
            elif found.get("input_type") != attr.get("input_type") or list(found.get("values") or []) != list(
                attr.get("values") or []
            ):
                attr_changes.append({"action": "change", "id": found.get("id"), **attr})
        if not type_change and not attr_changes:
            unchanged.append(label["name"])
            continue
        entry = {
            "id": current.get("id"),
            "name": label["name"],
            "attributes": attr_changes,
        }
        if type_change:
            entry["from_type"] = current.get("type")
            entry["to_type"] = label.get("type")
            entry["requires_confirmation"] = True
            blocked.append(
                {
                    "name": label["name"],
                    "reason": (
                        f"This would change {label['name']} from {current.get('type')} "
                        f"to {label.get('type')}."
                    ),
                    "change": entry,
                }
            )
        if attr_changes:
            update.append(entry)
    return {"create": create, "update": update, "unchanged": unchanged, "blocked": blocked}


def to_cvat_attributes(attributes: list[dict], existing: dict[str, dict] | None = None) -> list[dict]:
    payload = []
    for attr in attributes:
        item = {
            "name": attr["name"],
            "mutable": True,
            "input_type": attr.get("input_type") or "text",
            "default_value": str(attr.get("default") or ""),
            "values": list(attr.get("values") or []),
        }
        if existing and existing.get(attr["name"], {}).get("id"):
            item["id"] = existing[attr["name"]]["id"]
        payload.append(item)
    return payload
