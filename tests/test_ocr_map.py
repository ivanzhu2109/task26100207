"""Automated tests for ocr_map (standard library unittest only)."""

import random
import unittest

from ocr_map import Field, Page, Word, assign_words, map_quad

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def forward_point(page, p):
    """Independent test-side implementation of the documented pipeline."""
    left, bottom, right, top = page.crop_box
    cw, ch = right - left, top - bottom
    u, v = p[0] - left, p[1] - bottom
    if page.rotate == 0:
        rx, ry = u, v
    elif page.rotate == 90:
        rx, ry = v, cw - u
    elif page.rotate == 180:
        rx, ry = cw - u, ch - v
    elif page.rotate == 270:
        rx, ry = ch - v, u
    else:
        raise ValueError(page.rotate)
    rw, rh = (cw, ch) if page.rotate in (0, 180) else (ch, cw)
    px = rx * page.width / rw
    py = (rh - ry) * page.height / rh
    m = page.matrix
    x = m[0][0] * px + m[0][1] * py + m[0][2]
    y = m[1][0] * px + m[1][1] * py + m[1][2]
    w = m[2][0] * px + m[2][1] * py + m[2][2]
    return (x / w, y / w)


def forward_quad(page, tpl_quad):
    return tuple(forward_point(page, p) for p in tpl_quad)


def rect_quad(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def scale_matrix(m, c):
    return tuple(tuple(c * v for v in row) for row in m)


class MapQuadTest(unittest.TestCase):
    """Non-zero CropBox origin + 90-degree rotation + non-uniform scale
    + perspective term, all at once."""

    def setUp(self):
        self.page = Page(
            crop_box=(100.0, 200.0, 300.0, 500.0),
            rotate=90,
            width=600.0,
            height=800.0,
            matrix=((1.0, 0.5, 10.0), (0.0, 1.0, 20.0), (0.001, 0.0, 1.0)),
        )

    def test_map_quad_matches_independently_computed_coordinates(self):
        # Expected values derived by hand from the documented pipeline,
        # not produced by any library function.
        ocr_quad = [(10.0, 20.0), (410.0, 820.0),
                    (631.25, 512.5), (381.25, 12.5)]
        expected = [(100.0, 200.0), (300.0, 200.0),
                    (300.0, 500.0), (100.0, 500.0)]
        got = map_quad(self.page, ocr_quad)
        self.assertEqual(len(got), 4)
        for (gx, gy), (ex, ey) in zip(got, expected):
            self.assertAlmostEqual(gx, ex, delta=1e-9)
            self.assertAlmostEqual(gy, ey, delta=1e-9)

    def test_assignment_under_full_pipeline(self):
        fields = [
            Field("f1", (100.0, 200.0, 200.0, 350.0)),
            Field("f2", (200.0, 200.0, 300.0, 350.0)),
        ]
        inside = Word("in", forward_quad(
            self.page, rect_quad(110.0, 210.0, 190.0, 340.0)))
        outside = Word("out", forward_quad(
            self.page, rect_quad(400.0, 600.0, 450.0, 650.0)))
        result = assign_words(self.page, fields, [outside, inside])
        self.assertEqual([w for w, _ in result["assignments"]["f1"]], ["in"])
        self.assertAlmostEqual(result["assignments"]["f1"][0][1], 1.0,
                               delta=1e-9)
        self.assertEqual(result["assignments"]["f2"], [])
        self.assertEqual(result["unassigned"], ["out"])


class AssignmentRulesTest(unittest.TestCase):
    """Identity pipeline: crop (0,0,200,100), rotate 0, canvas 200x100."""

    def setUp(self):
        self.page = Page(crop_box=(0.0, 0.0, 200.0, 100.0), rotate=0,
                         width=200.0, height=100.0, matrix=IDENTITY)

    def word_from_template_rect(self, wid, rect):
        return Word(wid, forward_quad(self.page, rect_quad(*rect)))

    def test_skewed_thin_quad_beats_center_and_bbox_heuristics(self):
        fields = [Field("left", (0.0, 0.0, 100.0, 100.0)),
                  Field("right", (100.0, 0.0, 200.0, 100.0))]
        # Thin trapezoid: wide end inside "left", narrow tip in "right".
        tpl_quad = [(60.0, 40.0), (140.1, 48.0),
                    (140.1, 52.0), (60.0, 60.0)]
        word = Word("skew", forward_quad(self.page, tpl_quad))

        # The naive signals both point at "right":
        # bounding-box centre x = 100.05 lies in "right";
        cx = (60.0 + 140.1) / 2.0
        self.assertGreater(cx, 100.0)
        # bounding-box overlap with "right" (40.1*20) exceeds "left" (40*20).
        self.assertGreater(40.1 * 20.0, 40.0 * 20.0)

        result = assign_words(self.page, fields, [word])
        # True overlap area puts the word in "left".
        h100 = 20.0 - 16.0 * 40.0 / 80.1
        expected = ((20.0 + h100) / 2.0 * 40.0) / ((20.0 + 4.0) / 2.0 * 80.1)
        self.assertEqual(result["assignments"]["right"], [])
        self.assertEqual([w for w, _ in result["assignments"]["left"]],
                         ["skew"])
        self.assertAlmostEqual(result["assignments"]["left"][0][1], expected,
                               delta=1e-9)

    def test_exact_tie_goes_to_lexicographically_smaller_field_id(self):
        fields = [Field("b", (0.0, 0.0, 100.0, 100.0)),
                  Field("a", (100.0, 0.0, 200.0, 100.0))]
        word = self.word_from_template_rect("w", (50.0, 25.0, 150.0, 75.0))
        result = assign_words(self.page, fields, [word])
        self.assertEqual([w for w, _ in result["assignments"]["a"]], ["w"])
        self.assertAlmostEqual(result["assignments"]["a"][0][1], 0.5,
                               delta=1e-9)
        self.assertEqual(result["assignments"]["b"], [])

    def test_edge_touch_only_is_not_a_candidate(self):
        fields = [Field("a", (100.0, 0.0, 200.0, 100.0))]
        # Shares only the edge x=200 with the field / CropBox.
        word = self.word_from_template_rect("w", (200.0, 0.0, 300.0, 100.0))
        result = assign_words(self.page, fields, [word])
        self.assertEqual(result["assignments"]["a"], [])
        self.assertEqual(result["unassigned"], ["w"])

    def test_input_order_does_not_matter(self):
        fields = [Field("f1", (0.0, 0.0, 90.0, 100.0)),
                  Field("f2", (90.0, 0.0, 200.0, 100.0))]
        words = [
            self.word_from_template_rect("w1", (10.0, 10.0, 80.0, 90.0)),
            self.word_from_template_rect("w2", (100.0, 10.0, 190.0, 90.0)),
            self.word_from_template_rect("w3", (40.0, 25.0, 140.0, 75.0)),
            self.word_from_template_rect("w4", (300.0, 0.0, 400.0, 100.0)),
        ]
        reference = assign_words(self.page, fields, words)
        rng = random.Random(20261001)
        for _ in range(10):
            shuffled_fields = fields[:]
            shuffled_words = words[:]
            rng.shuffle(shuffled_fields)
            rng.shuffle(shuffled_words)
            result = assign_words(self.page, shuffled_fields, shuffled_words)
            self.assertEqual(result, reference)


class InvarianceTest(unittest.TestCase):
    """Coverage semantics and canvas-independence of the assignment."""

    def test_partially_out_of_page_word_keeps_full_area_denominator(self):
        page = Page(crop_box=(0.0, 0.0, 200.0, 100.0), rotate=0,
                    width=200.0, height=100.0, matrix=IDENTITY)
        fields = [Field("f", (100.0, 0.0, 200.0, 100.0))]
        # Word sticks out past the CropBox right edge: full area 100*50,
        # common area 50*50, so coverage must be exactly 0.5 (not 1.0).
        word = Word("w", forward_quad(page, rect_quad(150.0, 25.0,
                                                      250.0, 75.0)))
        result = assign_words(page, fields, [word])
        self.assertEqual([w for w, _ in result["assignments"]["f"]], ["w"])
        self.assertAlmostEqual(result["assignments"]["f"][0][1], 0.5,
                               delta=1e-9)

    def _nontrivial_page(self, width, height, matrix):
        return Page(crop_box=(10.0, 20.0, 110.0, 70.0), rotate=270,
                    width=width, height=height, matrix=matrix)

    def test_canvas_matrix_and_winding_invariance(self):
        base_matrix = ((1.0, 0.2, 3.0), (0.1, 1.0, 5.0), (0.002, -0.001, 1.0))
        base_page = self._nontrivial_page(300.0, 500.0, base_matrix)
        fields = [
            Field("fa", (10.0, 20.0, 60.0, 70.0)),
            Field("fb", (60.0, 20.0, 110.0, 70.0)),
        ]
        tpl_rects = {
            "w1": (15.0, 25.0, 55.0, 65.0),    # fully inside fa
            "w2": (65.0, 25.0, 105.0, 65.0),   # fully inside fb
            "w3": (40.0, 30.0, 80.0, 60.0),    # straddles fa/fb evenly
            "w4": (75.0, 25.0, 145.0, 65.0),   # sticks out of the CropBox
            "w5": (200.0, 200.0, 260.0, 240.0)  # fully outside
        }

        def run(page, reverse=False):
            words = []
            for wid, rect in tpl_rects.items():
                quad = forward_quad(page, rect_quad(*rect))
                if reverse:
                    quad = tuple(reversed(quad))
                words.append(Word(wid, quad))
            return assign_words(page, fields, words)

        reference = run(base_page)

        # Same logical page on a different raster canvas (new W and H).
        other_canvas = run(self._nontrivial_page(777.0, 123.0, base_matrix))
        # M multiplied by a non-zero constant: identical projective map.
        scaled = run(self._nontrivial_page(300.0, 500.0,
                                           scale_matrix(base_matrix, 7.3)))
        # Reversed vertex winding on every word quad.
        reversed_winding = run(base_page, reverse=True)

        for variant in (other_canvas, scaled, reversed_winding):
            self.assertEqual(variant["unassigned"], reference["unassigned"])
            self.assertEqual(variant["assignments"].keys(),
                             reference["assignments"].keys())
            for fid in reference["assignments"]:
                got = variant["assignments"][fid]
                exp = reference["assignments"][fid]
                self.assertEqual([w for w, _ in got], [w for w, _ in exp])
                for (_, gcov), (_, ecov) in zip(got, exp):
                    self.assertAlmostEqual(gcov, ecov, delta=1e-9)

        # Sanity of the reference itself: w3 ties -> "fa"; w4 keeps the
        # full-quad denominator; w5 is unassigned.
        self.assertEqual([w for w, _ in reference["assignments"]["fa"]],
                         ["w1", "w3"])
        self.assertEqual([w for w, _ in reference["assignments"]["fb"]],
                         ["w2", "w4"])
        self.assertEqual(reference["unassigned"], ["w5"])
        self.assertAlmostEqual(
            dict(reference["assignments"]["fb"])["w4"], 0.5, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
