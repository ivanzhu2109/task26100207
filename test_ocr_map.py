"""ocr_map 的自动化测试（仅标准库）。"""
import unittest

from ocr_map import Field, Page, Word, assign_words, map_quad


def forward_point(page, x, y):
    """测试侧独立实现的 模板->OCR 正向变换（不依赖被测库）。"""
    llx, lly, urx, ury = page.crop_box
    u, v = x - llx, y - lly
    w, h = urx - llx, ury - lly
    if page.rotate == 0:
        up, vp, bw, bh = u, v, w, h
    elif page.rotate == 90:
        up, vp, bw, bh = v, w - u, h, w
    elif page.rotate == 180:
        up, vp, bw, bh = w - u, h - v, w, h
    elif page.rotate == 270:
        up, vp, bw, bh = h - v, u, h, w
    else:
        raise ValueError(page.rotate)
    cx = up * page.width / bw
    cy = (bh - vp) * page.height / bh
    (a, b, c), (d, e, f), (g, hh, i) = page.matrix
    den = g * cx + hh * cy + i
    return ((a * cx + b * cy + c) / den, (d * cx + e * cy + f) / den)


def forward_quad(page, points):
    return tuple(forward_point(page, x, y) for x, y in points)


def rect_corners(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def make_word(page, word_id, template_points):
    return Word(id=word_id, quad=forward_quad(page, template_points))


class FullPipelineTest(unittest.TestCase):
    """非零 CropBox + 90 度旋转 + 非等比缩放 + 透视项。"""

    def setUp(self):
        self.page = Page(
            crop_box=(100.0, 200.0, 300.0, 500.0),
            rotate=90,
            width=600.0,
            height=800.0,
            matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1 / 600, 1 / 800, 1.0)),
        )

    def test_map_quad_matches_hand_computed_coordinates(self):
        # 手算（独立于被测函数）：
        # 模板 (100,200)->画布(0,0)->OCR(0,0)
        # 模板 (300,200)->画布(0,800)->OCR(0,400)
        # 模板 (300,500)->画布(600,800)->OCR(200,800/3)
        # 模板 (100,500)->画布(600,0)->OCR(300,0)
        quad = [(0.0, 0.0), (0.0, 400.0), (200.0, 800 / 3), (300.0, 0.0)]
        expected = [(100.0, 200.0), (300.0, 200.0), (300.0, 500.0), (100.0, 500.0)]
        mapped = map_quad(self.page, quad)
        for (ax, ay), (ex, ey) in zip(mapped, expected):
            self.assertAlmostEqual(ax, ex, places=9)
            self.assertAlmostEqual(ay, ey, places=9)
        # 手算内点：模板 (200,350)->画布(300,400)->OCR(150,200)
        (px, py), = map_quad(self.page, [(150.0, 200.0)])
        self.assertAlmostEqual(px, 200.0, places=9)
        self.assertAlmostEqual(py, 350.0, places=9)

    def test_assignment_through_full_pipeline(self):
        fields = [
            Field(id="lower", rect=(100.0, 200.0, 300.0, 350.0)),
            Field(id="upper", rect=(100.0, 350.0, 300.0, 500.0)),
        ]
        words = [
            make_word(self.page, "w1", rect_corners(120, 220, 280, 330)),
            make_word(self.page, "w2", rect_corners(120, 370, 280, 480)),
            make_word(self.page, "w3", rect_corners(120, 300, 280, 480)),
            make_word(self.page, "w4", rect_corners(400, 600, 500, 700)),
        ]
        result = assign_words(self.page, fields, words)
        self.assertEqual([wid for wid, _ in result.assignments["lower"]], ["w1"])
        self.assertEqual([wid for wid, _ in result.assignments["upper"]], ["w2", "w3"])
        self.assertEqual(list(result.unassigned), ["w4"])
        self.assertAlmostEqual(result.assignments["lower"][0][1], 1.0, places=9)
        self.assertAlmostEqual(result.assignments["upper"][0][1], 1.0, places=9)
        self.assertAlmostEqual(result.assignments["upper"][1][1], 13 / 18, places=9)


class RealOverlapTest(unittest.TestCase):
    """真实交叠面积 vs 中心命中 / 包围盒启发式，及并列与边界情形。"""

    def setUp(self):
        # OCR 坐标 = (x, 100 - y)，便于构造
        self.page = Page(
            crop_box=(0.0, 0.0, 200.0, 100.0),
            rotate=0,
            width=200.0,
            height=100.0,
            matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )

    def test_skewed_quad_defeats_center_and_bbox_heuristics(self):
        fields = [
            Field(id="left", rect=(0.0, 0.0, 50.0, 100.0)),
            Field(id="right", rect=(50.0, 0.0, 200.0, 100.0)),
        ]
        # 倾斜细长四边形：质心落在 right，包围盒交叠也是 right 大，
        # 但真实面积交叠 left 占 241/441 > 0.5。
        skewed = ((0.0, 0.0), (45.0, 0.0), (150.0, 10.0), (0.0, 60.0))
        # 面积质心（鞋带公式）x ≈ 50.71，中心命中会错选 right
        crosses = [
            skewed[i][0] * skewed[(i + 1) % 4][1]
            - skewed[(i + 1) % 4][0] * skewed[i][1]
            for i in range(4)
        ]
        area6 = 3 * sum(crosses)
        cx = sum(
            (skewed[i][0] + skewed[(i + 1) % 4][0]) * crosses[i] for i in range(4)
        ) / area6
        self.assertGreater(cx, 50.0)
        bbox_overlap_left = 50.0 * 60.0
        bbox_overlap_right = 100.0 * 60.0
        self.assertGreater(bbox_overlap_right, bbox_overlap_left)  # 包围盒也错
        word = make_word(self.page, "skew", skewed)
        result = assign_words(self.page, fields, [word])
        self.assertEqual([w for w, _ in result.assignments["left"]], ["skew"])
        self.assertEqual(result.assignments["right"], ())
        self.assertAlmostEqual(
            result.assignments["left"][0][1], 241 / 441, places=9
        )

    def test_coverage_tie_picks_lexicographically_smallest_field_id(self):
        # 词对称跨过边界，两侧覆盖率恰好都是 0.5；
        # 字典序更小的 "a_right" 获胜，与传入顺序无关。
        fields = [
            Field(id="b_left", rect=(0.0, 0.0, 50.0, 100.0)),
            Field(id="a_right", rect=(50.0, 0.0, 100.0, 100.0)),
        ]
        word = make_word(self.page, "tie", rect_corners(25, 20, 75, 80))
        result = assign_words(self.page, fields, [word])
        self.assertEqual([w for w, _ in result.assignments["a_right"]], ["tie"])
        self.assertEqual(result.assignments["b_left"], ())
        self.assertAlmostEqual(result.assignments["a_right"][0][1], 0.5, places=12)

    def test_boundary_touch_only_is_not_a_candidate(self):
        fields = [
            Field(id="edge", rect=(0.0, 0.0, 50.0, 100.0)),
            Field(id="home", rect=(50.0, 0.0, 100.0, 100.0)),
        ]
        # 与 edge 仅在 x=50 处共线接触，公共区域面积为零
        word = make_word(self.page, "touch", rect_corners(50, 10, 100, 30))
        result = assign_words(self.page, fields, [word])
        self.assertEqual(result.assignments["edge"], ())
        self.assertEqual([w for w, _ in result.assignments["home"]], ["touch"])

    def test_input_order_does_not_matter(self):
        fields = [
            Field(id="f1", rect=(0.0, 0.0, 50.0, 100.0)),
            Field(id="f2", rect=(50.0, 0.0, 100.0, 100.0)),
            Field(id="f3", rect=(100.0, 0.0, 200.0, 100.0)),
        ]
        words = [
            make_word(self.page, "w1", rect_corners(5, 10, 45, 90)),
            make_word(self.page, "w2", rect_corners(55, 10, 95, 90)),
            make_word(self.page, "w3", rect_corners(120, 10, 180, 90)),
            make_word(self.page, "w4", rect_corners(0, 0, 10, 5)),
        ]
        canonical = assign_words(self.page, fields, words)
        shuffled = assign_words(
            self.page,
            [fields[2], fields[0], fields[1]],
            [words[3], words[2], words[0], words[1]],
        )
        self.assertEqual(canonical, shuffled)
        self.assertEqual([w for w, _ in canonical.assignments["f1"]], ["w1", "w4"])


class InvarianceTest(unittest.TestCase):
    """越界词框分母、画布变换不变性、绕向与矩阵缩放的等价性。"""

    def test_partially_out_of_bounds_word_keeps_full_area_denominator(self):
        page = Page(
            crop_box=(0.0, 0.0, 100.0, 100.0),
            rotate=0,
            width=100.0,
            height=100.0,
            matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        fields = [Field(id="full", rect=(0.0, 0.0, 100.0, 100.0))]
        # 词框右半越出 CropBox：完整面积 60*20=1200，
        # 公共区域仅 20*20=400，覆盖率 1/3 < 0.5 -> 未归属。
        # 若误用裁剪后面积作分母会得到 1.0 而错误归属。
        word = make_word(page, "spill", rect_corners(80, 40, 140, 60))
        result = assign_words(page, fields, [word])
        self.assertEqual(result.assignments["full"], ())
        self.assertEqual(list(result.unassigned), ["spill"])

    def _variant_pages(self):
        base_matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.001, 0.0, 1.0))
        scaled_matrix = tuple(
            tuple(3.7 * v for v in row) for row in base_matrix
        )
        return {
            # 基准：rotate 0 + 透视
            "base": Page((10.0, 20.0, 110.0, 220.0), 0, 200.0, 400.0, base_matrix),
            # M 乘非零常数 3.7，齐次意义下等价
            "scaled_m": Page(
                (10.0, 20.0, 110.0, 220.0), 0, 200.0, 400.0, scaled_matrix
            ),
            # 换画布：rotate 90 + 不同非等比缩放 + 反射与透视
            "rot90": Page(
                (10.0, 20.0, 110.0, 220.0),
                90,
                500.0,
                250.0,
                ((-1.0, 0.0, 500.0), (0.0, 1.0, 0.0), (0.0, 0.002, 1.0)),
            ),
            # 换画布：rotate 180
            "rot180": Page(
                (10.0, 20.0, 110.0, 220.0),
                180,
                350.0,
                700.0,
                ((1.0, 0.3, 0.0), (0.0, 1.0, 0.0), (0.0005, 0.0, 1.0)),
            ),
            # 换画布：rotate 270
            "rot270": Page(
                (10.0, 20.0, 110.0, 220.0),
                270,
                300.0,
                600.0,
                ((1.0, 0.0, 0.0), (0.2, 1.0, 0.0), (0.0, -0.001, 1.0)),
            ),
        }

    def _template_shapes(self):
        return [
            ("w1", rect_corners(15, 30, 55, 100)),      # 全在 fa
            ("w2", rect_corners(65, 30, 105, 100)),     # 全在 fb
            ("w3", rect_corners(40, 130, 90, 200)),     # 在所有字段外
            # 跨过 fa/fb 边界的平行四边形，fa 侧覆盖率 2/3
            ("w4", ((45.0, 25.0), (75.0, 25.0), (65.0, 110.0), (35.0, 110.0))),
        ]

    def _fields(self):
        return [
            Field(id="fa", rect=(10.0, 20.0, 60.0, 120.0)),
            Field(id="fb", rect=(60.0, 20.0, 110.0, 120.0)),
        ]

    def _run(self, page, reverse_winding=False):
        words = []
        for word_id, points in self._template_shapes():
            if reverse_winding:
                points = tuple(reversed(points))
            words.append(make_word(page, word_id, points))
        return assign_words(page, self._fields(), words)

    def _assert_expected(self, result):
        self.assertEqual([w for w, _ in result.assignments["fa"]], ["w1", "w4"])
        self.assertEqual([w for w, _ in result.assignments["fb"]], ["w2"])
        self.assertEqual(list(result.unassigned), ["w3"])
        self.assertAlmostEqual(result.assignments["fa"][0][1], 1.0, places=9)
        self.assertAlmostEqual(result.assignments["fa"][1][1], 2 / 3, places=9)
        self.assertAlmostEqual(result.assignments["fb"][0][1], 1.0, places=9)

    def test_assignment_invariant_across_canvas_transforms(self):
        pages = self._variant_pages()
        reference = self._run(pages["base"])
        self._assert_expected(reference)
        for name, page in pages.items():
            with self.subTest(variant=name):
                result = self._run(page)
                self._assert_expected(result)
                for fid in ("fa", "fb"):
                    for (w0, c0), (w1, c1) in zip(
                        reference.assignments[fid], result.assignments[fid]
                    ):
                        self.assertEqual(w0, w1)
                        self.assertAlmostEqual(c0, c1, places=9)
                self.assertEqual(reference.unassigned, result.unassigned)

    def test_reversed_winding_gives_same_result(self):
        page = self._variant_pages()["rot90"]
        forward = self._run(page)
        reversed_winding = self._run(page, reverse_winding=True)
        self.assertEqual(forward.assignments.keys(), reversed_winding.assignments.keys())
        self.assertEqual(forward.unassigned, reversed_winding.unassigned)
        for fid in forward.assignments:
            f_entries = forward.assignments[fid]
            r_entries = reversed_winding.assignments[fid]
            self.assertEqual([w for w, _ in f_entries], [w for w, _ in r_entries])
            for (_, c0), (_, c1) in zip(f_entries, r_entries):
                self.assertAlmostEqual(c0, c1, places=12)


if __name__ == "__main__":
    unittest.main()
