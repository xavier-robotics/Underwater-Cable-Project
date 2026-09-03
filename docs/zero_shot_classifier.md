# SAM3 + YOLOE 无训练分类

## 目标

该路径不调用 Codex/API，也不训练项目数据。它复用 SAM3 的视频分段、海缆
mask 和裁剪，再用 YOLOE 开放词汇提示与固定几何规则输出和 Codex 基准相同
的三分类：

| class_id | class_name | 规则 |
|---:|---|---|
| 0 | `damaged` | 发现明确损伤，忽略位置 |
| 1 | `exposed_intact` | 触底且未发现明确损伤 |
| 2 | `suspended_intact` | 悬空且未发现明确损伤 |

## 处理流程

```text
长视频
  -> SAM3(pipe) 检出与分段
  -> 每个样本 3 帧 + 全景/crop/mask
  -> YOLOE 局部图损伤提示
  -> YOLOE 全景图位置提示
  -> SAM3 mask 面积先验 + 池底间隙规则
  -> 三帧概率融合
  -> 三分类结果与诊断图
```

损伤提示包括 `damaged cable`、`broken cable`、`cut cable`、
`cable with exposed wires` 等，并与 `intact cable`、`smooth cable surface`
比较。位置提示包括触底、平放、悬空和位于池底上方等表达。全部提示和固定阈值
位于 `configs/zero_shot_classifier.yaml`。

损伤优先：任意一帧达到强损伤阈值，或至少两帧达到弱损伤判据，最终类别就是
`damaged`。没有达到损伤条件时，才根据三帧悬空概率平均值区分两个完好类别。

## 运行

复用已经生成的 SAM3 样本：

```bash
scripts/run_sam3_yoloe_pipeline.sh \
  --samples-dir outputs/sam3_codex_pipeline/sample_frames \
  --work-dir outputs/sam3_yoloe_run \
  --skip-split
```

从新长视频完整运行：

```bash
scripts/run_sam3_yoloe_pipeline.sh \
  --input /path/to/long_video.mp4 \
  --work-dir outputs/sam3_yoloe_run \
  --mode scan \
  --force-split
```

同时运行 Codex 基准和无训练方法：

```bash
scripts/run_sam3_codex_pipeline.sh \
  --samples-dir outputs/sam3_codex_pipeline/sample_frames \
  --work-dir outputs/method_comparison \
  --classifier both \
  --skip-split
```

`both` 会生成：

```text
classification/             # Codex 基准
classification_zero_shot/   # SAM3 + YOLOE
comparison/                 # 逐样本一致率
```

## 输出

`classification_zero_shot/` 包含：

- `results.json`、`results.csv`：三分类结果；
- `summary.json`：类别数量与警告数量；
- `review.md`：可读汇总；
- `diagnostics/`：每帧 SAM3 mask、YOLOE 提示框和分数；
- `result_cache.json`：输入、配置和权重未变化时复用；
- `requests.json`：实际参与分类的样本和帧。

## 当前验证

在仓库现有 7 个样本上，无训练路径全部完成分类，与历史 Codex 三分类结果一致
6 个，一致率为 `85.71%`。这只是“与 Codex 基准的一致率”，不是真实标注准确率。
唯一分歧来自一个轻微损伤样本，历史 Codex 结果本身也带有复核标记。

## 限制

- YOLOE 文本分数是开放词汇匹配分数，不是经过本项目标定的缺陷概率；
- 轻微磨损、低对比裂口和浑浊画面仍可能漏检；
- 反光、标记和颜色突变可能产生弱损伤提示；
- 触底/悬空依赖全景信息、SAM3 mask 尺度和池底边缘，机位变化后应重新检查阈值；
- 配置调整属于规则标定，不会更新模型权重。
