# SAM3 单类破损检测标注（替代旧组合状态流程）

本入口现在只生成局部破损框，类别固定 `0: damage`。
不运行整缆、触底或悬空提示，不输出整缆框、mask、状态类别。
默认提示为 `damage`、`broken sheath`、`cable damage`、`silver patch`；最后一项沿用本项目“亮片视为模拟破损”的定义。
可重复使用 `--prompt` 覆盖整组默认提示，不能传入 cable、pipeline 等整缆概念。

## 运行

在项目根目录激活 sam3 环境后：

```bash
python -m scripts.build_sam3_damage_dataset \
  --input data/processed_images/image_1 \
  --out-dir outputs/sam3_damage_image1 \
  --checkpoint checkpoints/sam3/sam3.pt \
  --confidence 0.45 \
  --max-box-area-ratio 0.35 \
  --max-box-span-ratio 0.85 \
  --save-previews
```

`--input` 接受单图或递归图片目录。`--limit 3` 可先试前三张。
旧命令 `python -m scripts.build_sam3_state_dataset generate ...` 和
`python -m scripts.sam3_segment_candidates ...` 也进入同一个单类流程。
旧的状态参数和 `export --review` 不再使用，生成时直接写标签。
`bash run_scripts/san3_segment.sh --save-previews` 处理之前的测试图。
默认要求输出目录不存在；加 `--append` 可以向现有单类编号数据集追加。
批处理脚本 `sam3_damage_dataset.sh` 已默认启用 `--append`。
不要把输出目录放在输入目录内部，也不要把预览图、对比图作为原图输入。

## 输出

```text
images/0001.jpg
labels/0001.txt
classes.yaml
annotations.json
previews/0001.jpg  # 仅指定 --save-previews 时生成
```

按输入文件路径排序后，全局从 0001 开始编号：0001.jpg、0002.jpg……。
标签和预览使用相同编号，子目录中的同名输入也各分配独立编号。
编号至少四位，超过 9999 后自然扩展为 10000，不截断。
新数据集从 0001 开始；追加时从已有最大编号加一继续，不填补中间缺号。
原文件路径与编号的对应关系记录在 annotations.json 中，追加会保留旧记录并合并新记录。
按原图的绝对路径跳过已处理输入（同一路径即使内容或参数改变也跳过；需要重标请用新输出目录）。
每批提示词和参数分别保存在 runs 中，图片通过 run_id 关联；旧版本记录首次追加时自动补齐。
`--limit` 仍表示本次输入排序后的前 N 张，先限量再跳过已处理图片。
每个输入只生成一张训练图和一个同名 txt，不按提示词或类别复制图片。
JPEG 原文件直接复制；其他格式转成同分辨率 JPG，不裁剪、不缩放、不旋转，
不额外去绿或提亮。带 EXIF 旋转指令的输入会明确报错，避免静默改变坐标。
SAM3 内部输入变换不改变导出原图；模型掩膜恢复到原尺寸后才计算框。

标签每个局部破损框一行：

```text
0 x_center y_center width height
```

坐标归一化为 0–1，仅五列，不写分数、提示词或状态；多个框写入同一文件。
没有通过过滤的框时也写同名空 txt，不生成 class 1。
自动漏检同样会产生空标签，因此仍需检查空标签对应的图片是否确实完好。

`classes.yaml` 内容固定为：

```yaml
nc: 1
names: [damage]
```

它只定义类别，不冒充已经划分好的训练配置。本任务不训练、不划分数据集。
`annotations.json` 记录输入路径、原尺寸、框数量和运行参数，便于检查。
预览只有黄色 damage 框，无整缆框、状态或分数。默认不生成预览图片副本。

批处理脚本默认开启预览，处理 image_1 并写入 outputs/sam3_damage_numbered：

```bash
conda activate sam3
./run_scripts/sam3_damage_dataset.sh
# 覆盖输入、输出目录或其他参数；相对路径以项目根目录为基准。
./run_scripts/sam3_damage_dataset.sh --input data/processed_images/image_2 --out-dir outputs/sam3_damage_image2
```

将 image_2 追加到 image_1 的同一个输出目录，只覆盖输入参数即可：

```bash
./run_scripts/sam3_damage_dataset.sh --input data/processed_images/image_2
```

例如已有 0001–0113，追加后从 0114.jpg / 0114.txt 开始。
已有图片、标签和预览不会被覆盖；重复运行同一输入不会再次加载模型或生成重复记录。
程序会检查已有图片和标签是否与 annotations.json 一一对应，并拒绝同时向同一目录写入的进程。
若中断恰好发生在写图片与更新记录之间，可能留下未记录文件，此时追加会明确报错；
先核对并恢复记录与文件的一致性，或换新目录。不会静默覆盖残留文件。
JSON 记录采用原子替换，避免写入中断损坏已有记录。

## 过滤

- `--confidence`（兼容别名 `--damage-confidence`），默认 0.45。
- `--max-box-area-ratio`：框面积/图像面积上限，默认 0.35；不是 mask 面积。
- `--max-box-span-ratio`：框宽/图宽或框高/图高达到此值就丢弃，默认 0.85，用于排除贯穿画面的长条框。
- `--min-area-ratio`：候选 mask 最小面积比，默认 0.0002。
- `--iou-threshold`：重复框过滤阈值，默认 0.65；同一亮片的嵌套碎片保留外层局部框。

图像跨度是无需整缆检测的启发式过滤，不能保证排除所有较短的整缆误检；
可根据实际数据调低面积/跨度上限。不要为了有输出而把阈值降到产生大量误检。

## 小亮片漏检的调参

新增 `--prompt-threshold '提示词=阈值'`，重复参数可为不同提示词独立设阈值；
提示词不在原列表时自动加入。降低特定提示词的阈值不会降低其他提示词的阈值。
这些分数仅用于筛选，YOLO txt 仍然只含类别 0 和四个坐标。

`run_scripts/sam3_damage_dataset.sh` 现在默认使用以下组合：

```bash
--confidence 0.45 \
--prompt-threshold 'silver ring=0.05' \
--prompt-threshold 'silver patch on black pipe=0.10' \
--max-box-area-ratio 0.10 \
--max-box-span-ratio 0.50
```

在当前第 14 张中，silver ring 返回约 0.062 的小亮片候选；第 23 张的
silver patch on black pipe 候选约为 0.134。原统一 0.45 阈值会滤掉这些结果。
这里沿用亮片视为模拟破损的标注口径。更低阈值可能增加反光/标志牌误检，
因此脚本同时收紧局部框面积和跨度上限；近距离较大的真实破损也可能被这些上限过滤，需抽查。
这是针对漏检示例的试验参数，不代表完整数据集已通过人工验收。

**改参数后，向原目录追加不会重标旧图**。请使用新的输出目录重新运行：

```bash
./run_scripts/sam3_damage_dataset.sh --input data/processed_images/image_1 --out-dir outputs/sam3_damage_recall
./run_scripts/sam3_damage_dataset.sh --input data/processed_images/image_2 --out-dir outputs/sam3_damage_recall
./run_scripts/sam3_damage_dataset.sh --input data/processed_images/image_3 --out-dir outputs/sam3_damage_recall
```

保持输入目录顺序及文件集合不变，编号与原批次对应。旧数据集保持原样，方便比较。
