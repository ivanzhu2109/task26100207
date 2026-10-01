"""OCR word-quad to template-field mapping library."""

from .mapping import (
    COVERAGE_THRESHOLD,
    EPSILON,
    Field,
    Page,
    Word,
    assign_words,
    map_quad,
)

__all__ = [
    "COVERAGE_THRESHOLD",
    "EPSILON",
    "Field",
    "Page",
    "Word",
    "assign_words",
    "map_quad",
]
