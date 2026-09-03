# 触底/悬空判定方案

## 任务拆分

当前已完成海缆检出，后续不再建议直接训练位置与损伤的组合检测器。更稳的流程是：

```text
YOLOE 检出海缆 -> 接触状态后处理 -> exposed / suspended
```

其中 `exposed` 表示海缆触底，`suspended` 表示海缆与池底/海床之间存在可见间隙。

## 判定依据

优先使用 mask；没有 mask 时使用 bbox 近似。

- 海缆下边缘与池底边缘距离小于阈值：判定为 `exposed`。
- 海缆下方存在连续可见空隙：判定为 `suspended`。
- 如果水体浑浊或底线不可见：输出 `unknown`，不要强判。

## 当前实现

`scripts/classify_contact.py` 读取检测结果 `detections.json`，在每个海缆框下方搜索池底/海床边缘，通过距离和接触比例判断状态。

```bash
conda run -n yoloe python scripts/classify_contact.py \
  --config configs/contact.yaml
```

快速调参时只跑前 300 帧：

```bash
conda run -n yoloe python scripts/classify_contact.py \
  --config configs/contact.yaml \
  --output-dir outputs/contact_preview \
  --max-frames 300
```

输出：

- `outputs/contact/contact_states.json`
- `outputs/contact/videos/*_contact.mp4`

## 调参建议

- `contact_threshold_px`：触底距离阈值，1080p 下建议从 15-30 开始。
- `search_below_px`：在海缆框下方搜索池底的范围。
- `min_contact_ratio`：沿海缆宽度方向，有多少比例贴近池底才算触底。

现场前应抽取 30-50 帧人工核对，并按相机高度、水池底纹理调参。
