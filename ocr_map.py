"""OCR 词框到模板字段框的映射库。

仅接收页面元数据、模板字段框与 OCR 词框；
不读取 PDF、不调用 OCR、不做持久化与人工审核。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]
Quad = Tuple[Point, Point, Point, Point]
Rect = Tuple[float, float, float, float]  # (x0, y0, x1, y1)
Matrix3 = Tuple[Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float]]

COVERAGE_THRESHOLD = 0.5
TOLERANCE = 1e-9
_MIN_AREA = 1e-12


def _invert_matrix3(m: Matrix3) -> Matrix3:
    """3x3 矩阵求逆（伴随矩阵法）。"""
    (a, b, c), (d, e, f), (g, h, i) = m
    ca = e * i - f * h
    cb = c * h - b * i
    cc = b * f - c * e
    cd = f * g - d * i
    ce = a * i - c * g
    cf = c * d - a * f
    cg = d * h - e * g
    ch = b * g - a * h
    ci = a * e - b * d
    det = a * ca + b * cd + c * cg
    return (
        (ca / det, cb / det, cc / det),
        (cd / det, ce / det, cf / det),
        (cg / det, ch / det, ci / det),
    )


def _apply_homography(m: Matrix3, x: float, y: float) -> Point:
    """对点施加 3x3 齐次变换并做透视除法。"""
    (a, b, c), (d, e, f), (g, h, i) = m
    w = g * x + h * y + i
    return ((a * x + b * y + c) / w, (d * x + e * y + f) / w)


def _polygon_area(points: Sequence[Point]) -> float:
    """鞋带公式求多边形面积（取绝对值，顺逆时针均可）。"""
    total = 0.0
    count = len(points)
    for idx in range(count):
        x0, y0 = points[idx]
        x1, y1 = points[(idx + 1) % count]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _clip_half_plane(
    points: Sequence[Point],
    axis: int,
    bound: float,
    keep_greater: bool,
) -> List[Point]:
    """Sutherland–Hodgman：用半平面裁剪多边形。"""

    def inside(p: Point) -> bool:
        return p[axis] >= bound if keep_greater else p[axis] <= bound

    def intersect(p: Point, q: Point) -> Point:
        t = (bound - p[axis]) / (q[axis] - p[axis])
        return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

    output: List[Point] = []
    count = len(points)
    for idx in range(count):
        current = points[idx]
        previous = points[(idx - 1) % count]
        if inside(current):
            if not inside(previous):
                output.append(intersect(previous, current))
            output.append(current)
        elif inside(previous):
            output.append(intersect(previous, current))
    return output


def _clip_polygon_rect(points: Sequence[Point], rect: Rect) -> List[Point]:
    """用轴对齐矩形裁剪凸多边形。"""
    x0, y0, x1, y1 = rect
    result = _clip_half_plane(points, 0, x0, True)
    if result:
        result = _clip_half_plane(result, 0, x1, False)
    if result:
        result = _clip_half_plane(result, 1, y0, True)
    if result:
        result = _clip_half_plane(result, 1, y1, False)
    return result


@dataclass(frozen=True)
class Page:
    """页面元数据：CropBox、旋转、栅格画布尺寸与透视矩阵。"""
    crop_box: Rect      # 模板坐标中的 CropBox (llx, lly, urx, ury)
    rotate: int         # 顺时针角度，仅 0/90/180/270
    width: float        # 栅格画布宽 W
    height: float       # 栅格画布高 H
    matrix: Matrix3     # 作用于齐次列向量的 3x3 可逆矩阵 M


@dataclass(frozen=True)
class Field:
    """模板字段：唯一 ID + 模板坐标中的轴对齐矩形。"""
    id: str
    rect: Rect


@dataclass(frozen=True)
class Word:
    """OCR 词：唯一 ID + OCR 画布坐标中的凸四边形。"""
    id: str
    quad: Quad


@dataclass(frozen=True)
class AssignmentResult:
    """assign_words 的返回结果。"""
    # 字段 ID -> 按词 ID 字典序排列的 (词 ID, 覆盖率) 列表
    assignments: Dict[str, Tuple[Tuple[str, float], ...]]
    # 未归属词 ID，按字典序排列
    unassigned: Tuple[str, ...]


def map_quad(page: Page, quad: Sequence[Point]) -> List[Point]:
    """将 OCR 四边形映回模板坐标，保留真实四边形，不裁剪。"""
    llx, lly, urx, ury = page.crop_box
    crop_w = urx - llx
    crop_h = ury - lly
    if page.rotate not in (0, 90, 180, 270):
        raise ValueError(f"不支持的旋转角度: {page.rotate}")
    inverse = _invert_matrix3(page.matrix)
    mapped: List[Point] = []
    for ox, oy in quad:
        # 1) 逆透视：OCR 坐标 -> 栅格画布坐标（含透视除法）
        cx, cy = _apply_homography(inverse, ox, oy)
        # 2) 逆独立缩放 + y 轴翻转：画布 -> 旋转后外接矩形（y 向上）
        # 3) 逆顺时针旋转：旋转后矩形 -> 裁切页坐标
        if page.rotate == 0:
            u = cx * crop_w / page.width
            v = crop_h - cy * crop_h / page.height
        elif page.rotate == 90:
            u = cy * crop_w / page.height
            v = cx * crop_h / page.width
        elif page.rotate == 180:
            u = crop_w - cx * crop_w / page.width
            v = cy * crop_h / page.height
        else:  # 270
            u = crop_w - cy * crop_w / page.height
            v = crop_h - cx * crop_h / page.width
        # 4) 逆裁切平移：裁切页 -> 模板绝对坐标
        mapped.append((llx + u, lly + v))
    return mapped


def assign_words(
    page: Page,
    fields: Sequence[Field],
    words: Sequence[Word],
) -> AssignmentResult:
    """按模板坐标中的真实交叠面积把词归属到字段。"""
    buckets: Dict[str, List[Tuple[str, float]]] = {f.id: [] for f in fields}
    unassigned: List[str] = []
    for word in words:
        template_quad = map_quad(page, word.quad)
        word_area = _polygon_area(template_quad)
        candidates: List[Tuple[float, str]] = []
        if word_area > _MIN_AREA:
            for fld in fields:
                clipped = _clip_polygon_rect(template_quad, fld.rect)
                clipped = _clip_polygon_rect(clipped, page.crop_box)
                common = _polygon_area(clipped)
                if common <= _MIN_AREA:
                    continue
                coverage = common / word_area
                if coverage >= COVERAGE_THRESHOLD - TOLERANCE:
                    candidates.append((coverage, fld.id))
        if not candidates:
            unassigned.append(word.id)
            continue
        best_coverage = max(c for c, _ in candidates)
        winner = min(
            fid for c, fid in candidates if c >= best_coverage - TOLERANCE
        )
        winner_coverage = next(c for c, fid in candidates if fid == winner)
        buckets[winner].append((word.id, winner_coverage))
    assignments = {
        fid: tuple(sorted(entries)) for fid, entries in buckets.items()
    }
    return AssignmentResult(
        assignments=assignments,
        unassigned=tuple(sorted(unassigned)),
    )
