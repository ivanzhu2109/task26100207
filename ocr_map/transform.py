"""Coordinate pipeline between template space and the OCR canvas.

Template -> OCR canvas (semantic order):
  1. crop to CropBox (origin moves to the CropBox lower-left corner),
  2. rotate the cropped page clockwise by ``rotate`` degrees,
  3. flip into raster orientation and scale the rotated bounding box
     independently to width W and height H (origin top-left, y down,
     bottom-right corner is (W, H)),
  4. apply the invertible 3x3 homography ``matrix`` to homogeneous
     column vectors, with perspective division at the end.

Only the inverse direction (OCR -> template) is needed by the library,
but the forward direction is exposed for tests and documentation.
"""

from typing import Sequence, Tuple

Point = Tuple[float, float]
Matrix3 = Tuple[Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float]]


def matrix_inverse(m: Sequence[Sequence[float]]) -> Matrix3:
    """Inverse of a non-singular 3x3 matrix (adjugate / determinant)."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    cof = ((e * i - f * h, c * h - b * i, b * f - c * e),
           (f * g - d * i, a * i - c * g, c * d - a * f),
           (d * h - e * g, b * g - a * h, a * e - b * d))
    det = a * cof[0][0] + b * cof[1][0] + c * cof[2][0]
    if det == 0.0:
        raise ValueError("matrix is singular")
    return tuple(tuple(entry / det for entry in row) for row in cof)


def apply_homography(m: Sequence[Sequence[float]], p: Point) -> Point:
    """Apply a 3x3 homography to a point (perspective division included)."""
    x, y = p
    w = m[2][0] * x + m[2][1] * y + m[2][2]
    nx = (m[0][0] * x + m[0][1] * y + m[0][2]) / w
    ny = (m[1][0] * x + m[1][1] * y + m[1][2]) / w
    return (nx, ny)


def _rotated_size(cw: float, ch: float, rotate: int) -> Point:
    return (cw, ch) if rotate in (0, 180) else (ch, cw)


def _rotate_forward(u: float, v: float, cw: float, ch: float,
                    rotate: int) -> Point:
    if rotate == 0:
        return (u, v)
    if rotate == 90:
        return (v, cw - u)
    if rotate == 180:
        return (cw - u, ch - v)
    if rotate == 270:
        return (ch - v, u)
    raise ValueError("rotate must be one of 0, 90, 180, 270")


def _rotate_inverse(rx: float, ry: float, cw: float, ch: float,
                    rotate: int) -> Point:
    if rotate == 0:
        return (rx, ry)
    if rotate == 90:
        return (cw - ry, rx)
    if rotate == 180:
        return (cw - rx, ch - ry)
    if rotate == 270:
        return (ry, ch - rx)
    raise ValueError("rotate must be one of 0, 90, 180, 270")


def forward_point(crop_box, rotate, width, height, matrix, p: Point) -> Point:
    """Map a template-space point to OCR canvas coordinates."""
    left, bottom, right, top = crop_box
    cw, ch = right - left, top - bottom
    u, v = p[0] - left, p[1] - bottom
    rx, ry = _rotate_forward(u, v, cw, ch, rotate)
    rw, rh = _rotated_size(cw, ch, rotate)
    px = rx * (width / rw)
    py = (rh - ry) * (height / rh)
    return apply_homography(matrix, (px, py))


def inverse_point(crop_box, rotate, width, height, matrix, p: Point) -> Point:
    """Map an OCR canvas point back to template coordinates."""
    left, bottom, right, top = crop_box
    cw, ch = right - left, top - bottom
    rw, rh = _rotated_size(cw, ch, rotate)
    qx, qy = apply_homography(matrix_inverse(matrix), p)
    rx = qx / (width / rw)
    ry = rh - qy / (height / rh)
    u, v = _rotate_inverse(rx, ry, cw, ch, rotate)
    return (u + left, v + bottom)
