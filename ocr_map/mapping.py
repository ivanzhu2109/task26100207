"""Public API: map OCR word quads back to template space and assign
words to template fields by true overlap coverage."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from . import transform
from .geometry import Rect, common_area, polygon_area

COVERAGE_THRESHOLD = 0.5
EPSILON = 1e-9


@dataclass(frozen=True)
class Page:
    """Page metadata.

    crop_box: (left, bottom, right, top) in unrotated PDF coordinates
        (x right, y up); the lower-left corner need not be zero.
    rotate: clockwise page rotation, one of 0, 90, 180, 270.
    width, height: raster canvas size; the bottom-right corner is
        exactly (width, height).
    matrix: invertible 3x3 homography applied after the raster
        transform, acting on homogeneous column vectors.
    """
    crop_box: Rect
    rotate: int
    width: float
    height: float
    matrix: Tuple[Tuple[float, float, float],
                  Tuple[float, float, float],
                  Tuple[float, float, float]]


@dataclass(frozen=True)
class Field:
    """Axis-aligned template-space rectangle with a unique string id."""
    id: str
    rect: Rect


@dataclass(frozen=True)
class Word:
    """OCR word: unique string id plus a convex quad in OCR coordinates.

    Vertices are ordered along the boundary; either winding is fine.
    """
    id: str
    quad: Tuple[Tuple[float, float], ...]


def map_quad(page: Page, quad: Sequence[Tuple[float, float]]
             ) -> List[Tuple[float, float]]:
    """Map an OCR-space quad back to template coordinates.

    The true quadrilateral is preserved (vertex by vertex); it is not
    clipped to the page and never replaced by a bounding box.
    """
    return [
        transform.inverse_point(page.crop_box, page.rotate, page.width,
                                page.height, page.matrix, point)
        for point in quad
    ]


def assign_words(page: Page, fields: Sequence[Field],
                 words: Sequence[Word]) -> Dict:
    """Assign each word to at most one field by template-space coverage.

    Coverage = area(word quad AND field rect AND CropBox) divided by the
    area of the full, unclipped word quad, all measured in template
    coordinates. A word is eligible for a field when coverage >= 0.5
    (with 1e-9 tolerance) and the common area is non-zero. Among
    eligible fields the highest coverage wins; ties within 1e-9 go to
    the lexicographically smallest field id.

    Returns {"assignments": {field_id: [(word_id, coverage), ...]},
             "unassigned": [word_id, ...]} with every id list sorted
    lexicographically.
    """
    assignments: Dict[str, List[Tuple[str, float]]] = {
        field.id: [] for field in fields
    }
    unassigned: List[str] = []

    for word in words:
        template_quad = map_quad(page, word.quad)
        word_area = polygon_area(template_quad)
        best_field = None
        best_coverage = 0.0
        if word_area > 0.0:
            for field in fields:
                overlap = common_area(template_quad,
                                      [field.rect, page.crop_box])
                if overlap <= 0.0:
                    continue
                coverage = overlap / word_area
                if coverage < COVERAGE_THRESHOLD - EPSILON:
                    continue
                if best_field is None:
                    best_field, best_coverage = field.id, coverage
                elif coverage > best_coverage + EPSILON:
                    best_field, best_coverage = field.id, coverage
                elif (coverage > best_coverage - EPSILON
                      and field.id < best_field):
                    best_field = field.id
        if best_field is None:
            unassigned.append(word.id)
        else:
            assignments[best_field].append((word.id, best_coverage))

    for entries in assignments.values():
        entries.sort(key=lambda item: item[0])
    unassigned.sort()
    return {"assignments": assignments, "unassigned": unassigned}
