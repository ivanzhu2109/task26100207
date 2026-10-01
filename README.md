# ocr_map：OCR 词框到模板字段的映射库

仅接收页面元数据、模板字段框与 OCR 词框；不读取 PDF、不调用 OCR，
不做持久化与人工审核。仅依赖 Python 标准库。

## 输入结构

```python
from ocr_map import Page, Field, Word, map_quad, assign_words

page = Page(
    crop_box=(llx, lly, urx, ury),  # 模板坐标中的 CropBox，左下角可非零
    rotate=90,                      # 顺时针角度，仅 0/90/180/270
    width=W, height=H,              # 栅格画布尺寸，右下边界即 (W, H)
    matrix=((a, b, c),              # 3x3 可逆矩阵 M，作用于齐次列向量，
            (d, e, f),              # 透视除法后得到 OCR 坐标
            (g, h, i)),
)

fields = [Field(id="name", rect=(x0, y0, x1, y1)), ...]  # 模板坐标，轴对齐
words  = [Word(id="w1", quad=((x1, y1), (x2, y2), (x3, y3), (x4, y4))), ...]
```

坐标约定：模板为未旋转 PDF 页面的绝对坐标（x 向右、y 向上）；
OCR 画布左上角为原点（x 向右、y 向下）。词框顶点沿边界排列，
起点任意，顺逆时针均可，允许部分或全部越出页面。

## 两个操作

- `map_quad(page, quad)`：把 OCR 四边形逐顶点映回模板坐标
  （逆 M → 逆缩放/翻转 → 逆旋转 → 逆裁切平移），返回真实四边形，不裁剪。
- `assign_words(page, fields, words)`：返回 `AssignmentResult`：
  - `assignments`：字典 `字段ID -> ((词ID, 覆盖率), ...)`，按词 ID 字典序；
  - `unassigned`：未归属词 ID 元组，按字典序。

## 归属规则与数值约定

- 覆盖率 = 词框 ∩ 字段框 ∩ CropBox 的面积 ÷ 未裁剪完整词框面积，
  全部在模板坐标中计算（Sutherland–Hodgman 裁剪 + 鞋带公式）。
- 覆盖率阈值 `COVERAGE_THRESHOLD = 0.5`，比较容差 `TOLERANCE = 1e-9`：
  覆盖率 ≥ 0.5 − 1e-9 才有资格；多候选取覆盖率最大者，
  与最大值相差 ≤ 1e-9 时取字段 ID 字典序最小者。
- 公共区域面积 ≤ 1e-12 视为零面积，永不构成候选（仅边界接触不算）。
- 同一页面仅改变 OCR 画布的缩放/旋转/透视校正并同步变换词框时，
  字段归属与覆盖率保持不变（M 乘非零常数亦等价）。

## 运行测试

```sh
python3 -m unittest test_ocr_map -v
```
