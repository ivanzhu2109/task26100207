"""Convex polygon geometry helpers (pure standard library).

All polygons are lists of (x, y) tuples ordered along the boundary
(either winding). Rects are (x0, y0, x1, y1) with x0 <= x1, y0 <= y1.
"""

from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]


def polygon_area(points: Sequence[Point]) -> float:
    """Unsigned area of a simple polygon (shoelace formula)."""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return abs(total) * 0.5


def _clip_half_plane(points: Sequence[Point], edge: int,
                     value: float) -> List[Point]:
    """Keep the half-plane defined by one rectangle edge.

    edge 0: x >= value; 1: x <= value; 2: y >= value; 3: y <= value.
    """

    def coord(p: Point) -> float:
        return p[0] if edge < 2 else p[1]

    def inside(p: Point) -> bool:
        c = coord(p)
        return c >= value if edge % 2 == 0 else c <= value

    def intersect(a: Point, b: Point) -> Point:
        ca, cb = coord(a), coord(b)
        t = (value - ca) / (cb - ca)
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    out: List[Point] = []
    n = len(points)
    for i in range(n):
        cur = points[i]
        prv = points[i - 1]
        if inside(cur):
            if not inside(prv):
                out.append(intersect(prv, cur))
            out.append(cur)
        elif inside(prv):
            out.append(intersect(prv, cur))
    return out


def clip_polygon_to_rect(points: Sequence[Point], rect: Rect) -> List[Point]:
    """Clip a convex polygon against an axis-aligned rectangle.

    Returns the (possibly empty) vertex list of the clipped polygon.
    """
    x0, y0, x1, y1 = rect
    result = list(points)
    for edge, value in ((0, x0), (1, x1), (2, y0), (3, y1)):
        if not result:
            break
        result = _clip_half_plane(result, edge, value)
    return result


def common_area(points: Sequence[Point], rects: Sequence[Rect]) -> float:
    """Area of the polygon intersected with each rectangle in turn."""
    clipped = list(points)
    for rect in rects:
        if not clipped:
            return 0.0
        clipped = clip_polygon_to_rect(clipped, rect)
    return polygon_area(clipped)
