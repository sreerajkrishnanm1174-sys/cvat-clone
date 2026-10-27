"""Geometry helpers shared by learning and review. CVAT-agnostic."""

from __future__ import annotations

import math


def pairs(points: list[float]) -> list[tuple[float, float]]:
    return list(zip(points[0::2], points[1::2]))


def flat(pts: list[tuple[float, float]]) -> list[float]:
    out: list[float] = []
    for x, y in pts:
        out.extend((float(x), float(y)))
    return out


def bbox(points: list[float], shape_type: str) -> tuple[float, float, float, float]:
    if shape_type == "rectangle" and len(points) >= 4:
        x1, y1, x2, y2 = points[:4]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if shape_type == "ellipse" and len(points) >= 4:
        cx, cy, rx, ry = points[:4]
        return cx - rx, cy - ry, cx + rx, cy + ry
    pts = pairs(points)
    if not pts:
        return 0, 0, 0, 0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def box_size(points: list[float], shape_type: str) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox(points, shape_type)
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def iou(a_points: list[float], a_type: str, b_points: list[float], b_type: str) -> float:
    ax1, ay1, ax2, ay2 = bbox(a_points, a_type)
    bx1, by1, bx2, by2 = bbox(b_points, b_type)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def apply_box_delta(points: list[float], delta: dict) -> list[float]:
    x1, y1, x2, y2 = points[:4]
    return [
        x1 + float(delta.get("d_xtl", 0)),
        y1 + float(delta.get("d_ytl", 0)),
        x2 + float(delta.get("d_xbr", 0)),
        y2 + float(delta.get("d_ybr", 0)),
        *points[4:],
    ]


def box_delta(agent: list[float], human: list[float]) -> dict:
    return {
        "d_xtl": human[0] - agent[0],
        "d_ytl": human[1] - agent[1],
        "d_xbr": human[2] - agent[2],
        "d_ybr": human[3] - agent[3],
    }


def _closed(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(pts) >= 2 and pts[0] == pts[-1]:
        return pts
    return [*pts, pts[0]]


def polygon_area(points: list[float]) -> float:
    pts = _closed(pairs(points))
    if len(pts) < 4:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def centroid(points: list[float]) -> tuple[float, float]:
    pts = pairs(points)
    if not pts:
        return 0.0, 0.0
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def scale_outline(points: list[float], scale: float) -> list[float]:
    cx, cy = centroid(points)
    scaled = [((x - cx) * scale + cx, (y - cy) * scale + cy) for x, y in pairs(points)]
    return flat(scaled)


def mean_spacing(points: list[float]) -> float:
    pts = pairs(points)
    if len(pts) < 2:
        return 0.0
    lengths = [
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(pts, pts[1:] + pts[:1])
    ]
    return sum(lengths) / len(lengths)


def simplify_polygon(points: list[float], ratio: float = 0.012) -> list[float]:
    """Drop points that sit on a straight edge. Keep a point where the outline turns."""
    pts = pairs(points)
    if len(pts) < 4:
        return list(points)
    cx, cy = centroid(points)
    start = max(range(len(pts)), key=lambda index: math.hypot(pts[index][0] - cx, pts[index][1] - cy))
    rotated = pts[start:] + pts[:start]
    xs = [point[0] for point in rotated]
    ys = [point[1] for point in rotated]
    diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    simplified = _rdp(rotated + [rotated[0]], max(1.5, diagonal * ratio))
    if simplified and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]
    if len(simplified) < 3:
        return list(points)
    return flat(simplified)


def outline_is_template(points: list[float]) -> bool:
    """A regular or four-corner outline is a stencil, not a traced silhouette."""
    pts = pairs(points)
    count = len(pts)
    if count < 4:
        return False
    if count == 4:
        return True
    edges = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:] + pts[:1])]
    if min(edges) <= 0:
        return False
    edge_spread = (max(edges) - min(edges)) / max(edges)
    turns = []
    for index, point in enumerate(pts):
        previous = pts[index - 1]
        nxt = pts[(index + 1) % count]
        first = math.atan2(point[1] - previous[1], point[0] - previous[0])
        second = math.atan2(nxt[1] - point[1], nxt[0] - point[0])
        turn = abs((second - first + math.pi) % (2 * math.pi) - math.pi)
        turns.append(turn)
    turn_spread = (max(turns) - min(turns)) / math.pi
    return count >= 6 and edge_spread < 0.18 and turn_spread < 0.12


def _point_line_distance(point, start, end) -> float:
    span = math.hypot(end[0] - start[0], end[1] - start[1])
    if span == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    return abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])) / span


def _rdp(pts: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(pts) < 3:
        return list(pts)
    start, end = pts[0], pts[-1]
    farthest = 0.0
    index = 0
    for cursor in range(1, len(pts) - 1):
        distance = _point_line_distance(pts[cursor], start, end)
        if distance > farthest:
            farthest = distance
            index = cursor
    if farthest > tolerance and index:
        left = _rdp(pts[: index + 1], tolerance)
        right = _rdp(pts[index:], tolerance)
        return left[:-1] + right
    return [start, end]


def resample_polygon(points: list[float], target: int) -> list[float]:
    """Match a polygon's point count by walking its outline."""
    pts = pairs(points)
    if len(pts) < 3 or target < 3:
        return list(points)
    ring = _closed(pts)
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ring, ring[1:])]
    total = sum(lengths)
    if total <= 0:
        return flat(pts[:target])
    step = total / target
    samples: list[tuple[float, float]] = []
    seg = 0
    walked = 0.0
    for i in range(target):
        need = i * step
        while seg < len(lengths) - 1 and walked + lengths[seg] < need:
            walked += lengths[seg]
            seg += 1
        span = lengths[seg] or 1.0
        t = (need - walked) / span
        x1, y1 = ring[seg]
        x2, y2 = ring[seg + 1]
        samples.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    return flat(samples)


def _cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d) -> bool:
    d1, d2 = _cross(a, b, c), _cross(a, b, d)
    d3, d4 = _cross(c, d, a), _cross(c, d, b)
    return d1 * d2 < 0 and d3 * d4 < 0


def crosses_self(points: list[float]) -> bool:
    pts = pairs(points)
    n = len(pts)
    if n < 4:
        return False
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) % n <= 1 or (i == 0 and j == n - 1):
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if _segments_cross(a, b, c, d):
                return True
    return False


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2
