# ocr_map — OCR 词框到模板字段的映射库

纯 Python 标准库实现。输入仅为页面元数据、模板字段框和 OCR 词框；
不读取 PDF、不调用 OCR、不做持久化或人工审核。

## 输入结构

```python
from ocr_map import Page, Field, Word, map_quad, assign_words

page = Page(
    crop_box=(100.0, 200.0, 300.0, 500.0),  # (left, bottom, right, top)，
                                            # 未旋转 PDF 坐标，y 向上，
                                            # 左下角可以非零
    rotate=90,                              # 顺时针角度：0 / 90 / 180 / 270
    width=600.0, height=800.0,              # 栅格画布尺寸，右下角即 (W, H)
    matrix=((1.0, 0.5, 10.0),               # 可逆 3x3 单应矩阵 M，
            (0.0, 1.0, 20.0),               # 作用于齐次列向量，
            (0.001, 0.0, 1.0)),             # 透视除法后得到 OCR 坐标
)

fields = [Field("f1", (100.0, 200.0, 200.0, 350.0)),   # 模板坐标轴对齐矩形
          Field("f2", (200.0, 200.0, 300.0, 350.0))]

words = [Word("w1", ((x0, y0), (x1, y1), (x2, y2), (x3, y3)))]  # OCR 坐标凸四边形
```

模板到 OCR 画布的语义顺序：裁切至 CropBox → 顺时针旋转 Rotate →
将旋转外接矩形独立缩放到 W×H（原点左上、y 向下）→ 施加 M。

## 两个操作

- `map_quad(page, quad)`：把 OCR 四边形逐顶点映回模板坐标，保留真实
  四边形，不裁剪、不取外接框。
- `assign_words(page, fields, words)`：返回
  `{"assignments": {field_id: [(word_id, coverage), ...]},
    "unassigned": [word_id, ...]}`，所有 ID 列表按字符串字典序排列。

归属规则：覆盖率 = 词框 ∩ 字段框 ∩ CropBox 的面积（模板坐标）÷
未裁剪完整词框面积；覆盖率 ≥ 0.5 才有资格；多个候选取最大者，
并列（差 ≤ 1e-9）取字段 ID 字典序最小者；零面积公共区不算候选；
一个词最多归属一个字段。

## 数值误差约定

阈值与并列比较均允许 `1e-9` 误差（`ocr_map.EPSILON`）。面积用鞋带公式、
裁剪用 Sutherland–Hodgman 半平面求交，全部为精确解析计算，不做栅格化
或像素采样。输入应保证所需正逆映射在相关区域内有限且非奇异。

## 运行测试

```sh
python3 -m unittest discover -s tests -v
```
