# SAM 辅助标注与后处理方案

## 结论

SAM 适合作为本项目的主标注工具：用少量点击或框提示快速得到海缆 mask，再经过后处理生成训练标签。由于当前海缆样本少，优先用 SAM 做高质量伪标注/人工修正，比直接训练 YOLO 更稳。

但 SAM 本身不是缺陷分类模型。它能分割“海缆区域”，不能可靠判断“裸露/悬空”和“破损/非破损”。这两个状态仍需要人工确认，或用规则/小模型在 mask 基础上做辅助判定。

## 推荐路线

1. 用 1080p 抽帧数据 `data/frames_1080p` 做标注源。
2. 在 CVAT、Label Studio、Roboflow 或 SAM2 demo 中用 SAM 生成海缆 mask。
3. 人工确认 mask 边界，并给每个目标指定三类之一：
   - `0 damaged`
   - `1 exposed_intact`
   - `2 suspended_intact`
   只要确认破损，无论海缆裸露还是悬空，都标为 `0 damaged`。
4. 如果后续训练 YOLO 检测模型，用 `scripts/masks_to_yolo.py` 将 mask 转成矩形框标签。
5. 如果后续训练分割模型，保留 mask，使用 YOLO-seg、Mask R-CNN 或 SAM adapter 类方法。

## SAM + 后处理能否直接完成任务

可以做“海缆检测/分割”的初版，但不建议完全自动替代人工标注。

可自动化的部分：
- 根据 SAM mask 得到海缆轮廓和 bbox。
- 过滤过小 mask、过碎 mask。
- 根据细长形态、面积、长宽比筛掉明显非海缆目标。
- 将连续帧 mask 传播到相邻帧，减少逐帧点击。

需要人工或额外模型的部分：
- 破损/非破损：破损区域通常局部、细小，SAM 分割主体不等于识别缺陷。
- 裸露/悬空：需要判断海缆和海床的空间关系，单个 mask 不一定足够。
- 遮挡、浑浊、水草、阴影会让自动 mask 误选背景。

## 与 YOLO 的关系

建议把 YOLO 作为 Plan B 或最终部署模型：

- SAM 阶段：快速产出标签，适合数据少时降低标注成本。
- YOLO 阶段：用 SAM 辅助标注后的数据训练轻量检测器，适合视频批量推理和部署。
- 中期检查：展示 SAM 辅助标注样例、mask 后处理结果、YOLO 备用训练流程，会比只讲 YOLO 更符合小样本现实。

## mask 转 YOLO 框

如果每张图片有一个同名二值 mask，可运行：

```bash
conda run -n yoloe python scripts/masks_to_yolo.py \
  --images-dir data/frames_1080p/mmexport1779156433136 \
  --masks-dir data/sam_masks/mmexport1779156433136 \
  --labels-dir data/annotated_1080p/labels \
  --class-id 0
```

`class-id` 需要按人工判断填写。若同一帧有多个类别，建议在标注平台内直接导出 YOLO 标签，或按类别分别存放 mask 后多次转换再人工合并标签。
