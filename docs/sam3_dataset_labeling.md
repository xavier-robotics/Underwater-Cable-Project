# 增强图像数据集：选择 SAM3 辅助标注及运行方法

> 当前单类破损标注入口已改造，运行方法见 [SAM3 单类破损检测标注](sam3_state_dataset.md)。
> 下文为旧整缆三分类方案的历史记录，其候选脚本参数和导出步骤不适用于当前单类入口。

## 1. 框架决定

**选择 SAM3 生成海缆实例 mask，人工修正轮廓并确认类别，最后导出 YOLO 检测标签。** 保留 mask，后续需要时可以另行制作分割数据集。当前目标是制作可信的训练数据，SAM3 的提示分割适合承担轮廓候选生成工作。

SAM3 支持文本和视觉提示分割；YOLOE 同样支持检测、分割和开放词汇提示，并着重实时效率。两者都能辅助标注。这里选择 SAM3，是结合项目已有候选导出脚本及本批数据规模作出的工程判断，**并非已经在这 195 张图上测得 SAM3 精度高于 YOLOE**。能力说明见 [SAM3 官方说明](https://github.com/facebookresearch/sam3) 和 [YOLOE 官方说明](https://github.com/THU-MIG/yoloe)。

| 环节                     | 本次选择      | 原因                                                   |
| ------------------------ | ------------- | ------------------------------------------------------ |
| 找到海缆、生成轮廓       | SAM3          | 项目已有逐图生成 mask、框、预览图的脚本                |
| 修正漏检、边界和重复实例 | 人工审核      | 自动候选不等于最终标签                                 |
| 损伤、触底、悬空三分类   | 人工确认      | 涉及细小损伤与空间关系，不能直接把文本匹配分数当作真值 |
| 后续训练和部署           | YOLO 检测器等 | 标签输出格式与生成候选的框架相互独立                   |

现有 `scripts/run_sam3_yoloe_pipeline.sh` 是视频/样本级零样本分类入口，调用 SAM3 + YOLOE 分类流程；它不是对这三个普通图片目录直接导出人工审核训练标签的入口。`scripts/train_yolo.py` 使用 `ultralytics.YOLO`，默认模型为 `yolov8n.pt`，不要因项目存在 `yoloe/` 就把所有训练都理解成 YOLOE 训练。

## 2. 输入和类别规则

2026-09-20 检查到的输入：

| 目录                              | JPG 数量 | 文件名中的源视频标识      |
| --------------------------------- | -------: | ------------------------- |
| `data/processed_images/image_1` |      113 | `mmexport1779157136529` |
| `data/processed_images/image_2` |       37 | `mmexport1779157141457` |
| `data/processed_images/image_3` |       45 | `mmexport1779157142588` |
| 合计                              |      195 | 三个源视频标识            |

只处理这三个目录。父目录的 `comparisons/`、`comparison_overview.jpg` 是对比材料，不加入训练集。`image_1/2/3` 是输入分组名称，**不代表类别 0/1/2**。

类别以 [`configs/classes.yaml`](../configs/classes.yaml) 为准：

| ID | 名称                 | 人工判定口径                                   |
| -: | -------------------- | ---------------------------------------------- |
|  0 | `damaged`          | 海缆有明确损伤；损伤优先，不再按触底或悬空拆分 |
|  1 | `exposed_intact`   | 可见海缆触底、裸露，且未见明确损伤             |
|  2 | `suspended_intact` | 可见海缆悬空，且未见明确损伤                   |

以一根可区分的海缆实例的可见整体作为标注对象，损伤类别也框住该海缆，不只框裂口。多根海缆分别标注；同一根海缆的多个提示结果只保留一个实例。反光、绳索、标记不自动视为损伤。遮挡或画质导致无法确认状态时先暂缓；不得把“看不清”直接标成完好。增强可能改变颜色证据，必要时对照原图或源视频核实。

同一根完好海缆同时包含明显触底段和悬空段时，当前类别配置未定义整根实例的优先规则。本流程先将该图设为 `excluded` 并记录“混合位置状态”，待统一标注口径后再纳入；不要临时拆成两个类别或自行按多数面积决定。若已确认损伤，仍按损伤优先标为 0。

## 3. 环境准备

以下命令都从项目根目录执行，使用独立 SAM3 环境。此次只编写操作文档，未安装环境、下载权重或运行全量推理。检查时未发现名为 `sam3` 的 Conda 环境，也未发现默认路径 `ckpts/sam3/sam3.pt`；已有其他位置的可用环境或权重可以复用。

```bash
cd /home/leo/breeze/hailan_project/uw_detection
conda create -n sam3 python=3.12 -y
conda activate sam3

# 本地 sam3/README.md 所列 CUDA 12.8 PyTorch 安装方案。
# 先在实际运行终端用 nvidia-smi 确认驱动支持；不适配时按 PyTorch 官方选择安装包。
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./sam3
python -m pip install 'numpy<2' opencv-python-headless pillow pyyaml einops decord 'setuptools<81'
```

本地模型构建代码使用 `pkg_resources`，所以这里保留提供该接口的 setuptools。不要盲目套用 `constraints-sam3.txt` 中另一组 Torch/CUDA 版本来覆盖此环境。GPU 环境与模型加载仍需通过下面的单图试运行确认。

SAM3 权重需先在 [facebook/sam3](https://huggingface.co/facebook/sam3) 申请并获得访问权限，然后下载；使用现有权重时跳过下载并修改后续 `--checkpoint` 路径。

```bash
hf auth login
hf download facebook/sam3 sam3.pt --local-dir ckpts/sam3
python -c "import torch; from sam3.model_builder import build_sam3_image_model; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

使用 `--device cuda` 前应看到 `CUDA: True`。CPU 可通过 `--device cpu` 尝试，但不是本批处理的首选。

## 4. 先做一张图，再做全部候选

### 4.1 单图试运行

下面的图片确实存在于当前输入目录。提示词先只描述海缆主体，类别稍后人工判断。

```bash
python scripts/sam3_segment_candidates.py \
  --input data/processed_images/image_1/mmexport1779157136529_f001110_t00037.00.jpg \
  --checkpoint ckpts/sam3/sam3.pt \
  --out-dir outputs/sam3_annotation_pilot \
  --prompt pipe --prompt cable --prompt 'underwater cable' \
  --confidence 0.30 --device cuda
```

检查输出中的 `*_all_candidates.jpg`、`masks/*.png` 和 `candidates.json`。`--confidence 0.30` 是起点，不是此数据已验证的最优阈值。漏检时可尝试 0.15 或 0.10，并比较误检；面积过滤也会剔除过小或占据画面过大的目标。先在三个目录各抽几张，包括损伤、遮挡和悬空画面，确认效果，再全量执行。

现有脚本的 `--input` 只接受单张图或视频，不接受图片目录；也没有交互式点击修正界面。需要点击、框提示或边界编辑时，应另用支持这些操作的标注界面，不能给此脚本添加不存在的参数。

### 4.2 批处理三个目录

在已激活的 `sam3` 环境复制执行。这个文档内的 Python 循环调用已有脚本，不需要新建业务脚本。

```bash
python - <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path('data/processed_images')
out = Path('outputs/sam3_annotation_candidates')
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
for group in ('image_1', 'image_2', 'image_3'):
    images = sorted(p for p in (root / group).iterdir()
                    if p.is_file() and p.suffix.lower() in exts)
    assert images, f'没有图片: {group}'
    assert len({p.stem for p in images}) == len(images), '同组存在同名不同扩展名图片'
    for image in images:
        target = out / group / image.stem
        if (target / 'candidates.json').exists():
            print('跳过已有候选:', image)
            continue
        subprocess.run([
            sys.executable, 'scripts/sam3_segment_candidates.py',
            '--input', str(image), '--out-dir', str(target),
            '--checkpoint', 'ckpts/sam3/sam3.pt',
            '--prompt', 'pipe', '--prompt', 'cable',
            '--prompt', 'underwater cable', '--confidence', '0.30',
            '--device', 'cuda',
        ], check=True)
PY
```

每张图独立目录，避免不同图片的候选互相覆盖。此方式每张图重新加载模型，速度不如常驻模型批处理，但可直接复用现有脚本。断点续跑会跳过已有 `candidates.json`；改变提示词、阈值或输入图片后应换一个输出根目录，并在后续清单生成步骤同步修改路径，避免复用旧结果。

## 5. 人工审核与实例清单

### 5.1 生成待审核清单

全量候选完成后执行。清单中的每个图片默认 `pending`，每个候选默认不保留，避免未经审核的预测自动进入训练集。

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path('data/processed_images')
base = Path('outputs/sam3_annotation_candidates')
dest = Path('outputs/sam3_annotation_review.json')
assert not dest.exists(), '审核清单已存在；不要覆盖人工修改'
rows = []
for group in ('image_1', 'image_2', 'image_3'):
    for image in sorted((root / group).iterdir()):
        if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        folder = base / group / image.stem
        candidates = json.loads((folder / 'candidates.json').read_text())
        rows.append({
            'image': str(image), 'group': group,
            'source_video': image.stem.split('_f')[0],
            'status': 'pending', 'note': '',
            'objects': [{'candidate_id': c['id'], 'keep': False,
                         'class_id': None, 'mask': str(folder / c['mask_path'])}
                        for c in candidates],
        })
dest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
print('待审核图片数:', len(rows), '清单:', dest)
PY
```

### 5.2 逐图编辑清单

用文本编辑器打开 `outputs/sam3_annotation_review.json`，同时查看原尺寸增强图和对应候选叠加图：

1. 确认所有海缆实例都已找到，排除池底、阴影、绳索和重复候选。
2. 对正确实例设置 `keep: true`，填写 `class_id: 0`、`1` 或 `2`。
3. 如果 mask 有误，用支持二值 mask 编辑的工具修正；另存单通道 PNG（背景 0、目标 255），尺寸必须与输入图一致，把 `mask` 改成该文件路径。同一实例被遮挡成数块时，可保留在同一张实例 mask 中。
4. 漏检实例需要手工补画完整的实例 mask，并新增一个对象记录，可用 `candidate_id: "manual_1"`。不要仅保留模型找到的实例而忽略其他海缆。
5. 全图检查完成后设置 `status: "approved"`。真正没有目标的图设置 `status: "negative"`，所有对象 `keep` 为 false。无法可靠判断则设置 `status: "excluded"` 并在 `note` 写原因。

`pending` 会阻止导出；没有候选的图片也必须人工判定负样本或补标。SAM3 的分数只用于候选排序，不能代替三分类审核。清单是人工审核接口，项目当前没有为这一步提供现成 GUI。

## 6. 划分数据并导出 YOLO 检测标签

这批图的文件名表明它们来自三个视频。**按源视频分组划分，不能把相邻帧随机分到训练和验证集。** 如果不同视频实际拍的是同一根样件、同一连续实验，也应合并成一个独立分组。原图与对应增强图必须归入同一集合。

先审核类别分布，再在下面 `split_by_source` 字典中填写各视频所属集合（`train`、`val` 或 `test`）。默认留空，要求显式决定，不把某个输入目录默认当作测试集。只有三个视频时，三路划分可能导致某些类别在训练集或验证集缺失；可先做 train/val，补采独立测试视频。若某类别仅出现在一个视频，当前数据不能支持该类的可靠跨视频评估，应补采，而不是通过随机拆帧制造指标。

导出脚本直接读取**审核后的实例 mask**计算整体外接框：一个实例产生一行，支持同图多个类别、人工补标、负样本，并为文件名增加组前缀。输出是 YOLO **检测**标签，非 YOLO-seg 多边形标签。

```bash
python - <<'PY'
from pathlib import Path
from collections import Counter
import json
import shutil
import cv2
import numpy as np
import yaml

split_by_source = {
    'mmexport1779157136529': '',  # 审核类别与拍摄关系后填 train / val / test
    'mmexport1779157141457': '',
    'mmexport1779157142588': '',
}
rows = json.loads(Path('outputs/sam3_annotation_review.json').read_text())
names = yaml.safe_load(Path('configs/classes.yaml').read_text())['names']
out = Path('data/yolo_sam3_reviewed_v1')
assert not out.exists(), '输出目录已存在，请换版本名称，避免混入旧标签'
assert all(r['status'] in {'approved', 'negative', 'excluded'} for r in rows), '仍有待审核图'
assert all(s in {'train', 'val', 'test'} for s in split_by_source.values()), '先填写视频划分'
counts = Counter()
image_counts = Counter()
seen = set()
jobs = []
for row in rows:
    if row['status'] == 'excluded':
        continue
    image = Path(row['image'])
    frame = cv2.imread(str(image))
    assert frame is not None, f'无法读取图片: {image}'
    h, w = frame.shape[:2]
    split = split_by_source[row['source_video']]
    stem = row['group'] + '__' + image.stem
    assert stem not in seen, f'输出重名: {stem}'
    seen.add(stem)
    kept = [o for o in row['objects'] if o['keep']]
    assert (row['status'] == 'negative' and not kept) or (row['status'] == 'approved' and kept)
    lines = []
    for obj in kept:
        cls = obj['class_id']
        assert type(cls) is int and cls in names, f'非法类别: {cls}'
        mask = cv2.imread(obj['mask'], cv2.IMREAD_GRAYSCALE)
        assert mask is not None and mask.shape == (h, w), f'mask 缺失或尺寸不符: {obj["mask"]}'
        assert set(np.unique(mask)).issubset({0, 255}), 'mask 必须是 0/255 二值图'
        ys, xs = np.where(mask > 0)
        assert xs.size, '实例 mask 为空'
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        box = ((x1 + x2) / (2*w), (y1 + y2) / (2*h), (x2 - x1) / w, (y2 - y1) / h)
        lines.append(str(cls) + ' ' + ' '.join(f'{v:.6f}' for v in box))
        counts[(split, cls)] += 1
    image_counts[split] += 1
    jobs.append((image, split, stem, lines))

print('各集合图片数:', dict(image_counts))
print('各集合实例数:', dict(counts))
assert image_counts['train'] and image_counts['val'], 'train / val 均需有审核数据'
assert all(counts[('train', cls)] for cls in names), '训练集缺少类别，请调整分组或补采'
for cls in names:
    if not counts[('val', cls)]:
        print('注意：验证集缺少类别，无法评估该类:', cls, names[cls])

# 所有输入检查通过后才开始写出。
for image, split, stem, lines in jobs:
    idir, ldir = out / 'images' / split, out / 'labels' / split
    idir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, idir / (stem + image.suffix.lower()))
    (ldir / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
config = {'path': str(out.resolve()), 'train': 'images/train', 'val': 'images/val', 'names': names}
if image_counts['test']:
    config['test'] = 'images/test'
(out / 'dataset.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
shutil.copy2('outputs/sam3_annotation_review.json', out / 'review_manifest.json')
(out / 'split_by_source.json').write_text(json.dumps(split_by_source, indent=2) + '\n')
print('已导出:', out)
PY
```

示例目录结构：

```text
data/yolo_sam3_reviewed_v1/
  dataset.yaml
  review_manifest.json
  split_by_source.json
  images/train/...
  images/val/...
  labels/train/...
  labels/val/...
```

每行标签为 `class_id center_x center_y width height`，坐标归一化到 0～1；人工确认负样本对应空 TXT。候选目录和修正后的 mask 应一起归档，清单中的 mask 路径仍指向这些文件。

不要直接将 `outputs/.../masks` 交给现有 `scripts/masks_to_yolo.py`：候选文件名不是原图名；该脚本还按连通区域拆框、一次只能指定一个类别、写同名标签会覆盖。上述导出步骤避开了这些接口不匹配。也不要再对导出的数据运行 `scripts/make_yolo_dataset.py`：它会逐图随机划分，且缺少标签时写空文件，不适合本批连续帧的正式划分。

## 7. 训练前复核与可选训练

训练前逐图叠加检查最终框和类别，尤其检查多实例、遮挡断开的海缆、细小损伤，以及增强产生的伪纹理。确认三类训练实例均存在，验证集的类别覆盖足以支持预期指标，并记录未覆盖的类别。第一次实验建议保留三类样例的人工复核记录。

已有安装好 Ultralytics 的训练环境时，在该环境运行项目训练入口；无需把 SAM3 环境当作训练环境。下面是普通 YOLOv8 检测训练示例，模型权重可能需要下载：

```bash
python scripts/train_yolo.py \
  --data data/yolo_sam3_reviewed_v1/dataset.yaml \
  --model yolov8n.pt --epochs 100 --imgsz 960 --batch 8 \
  --device 0 --project runs/cable_detection --name sam3_reviewed_v1
```

显存不足时先降低 batch。保留 mask 并不表示此命令会训练分割模型；训练 YOLO-seg 需要另外导出多边形标签并使用相应分割模型。

## 8. 本文验证范围

已核对当前输入目录、类别映射、现有脚本参数和输出格式。文档内命令需要在准备好环境、权重并完成人工审核后执行；本文没有报告这批图上的模型精度或吞吐量，也没有把自动候选当作已完成的真实标注。
