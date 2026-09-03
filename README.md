# UW Detection

水下海缆检测与缺陷状态识别项目。当前实现了从视频抽帧、YOLO 数据集组织、训练和推理的最小闭环。

## 环境

SAM3 方案使用已有的 `sam3` 环境。注意不要直接 `pip install opencv-python`，最新 wheel 会把 `numpy` 升到 2.x，而 `sam3` 要求 `numpy<2`。

如果已经误装，按下面命令修复：

```bash
conda run -n sam3 python -m pip uninstall -y opencv-python opencv-python-headless numpy
conda run -n sam3 python -m pip install --no-cache-dir -c constraints-sam3.txt "numpy==1.26.4" "opencv-python-headless==4.8.1.78"
conda run -n sam3 python -m pip install --no-cache-dir -c constraints-sam3.txt --force-reinstall "setuptools==80.9.0"
conda run -n sam3 python -m pip check
conda run -n sam3 python -c "import cv2, numpy, sam3; print(cv2.__version__, numpy.__version__)"
```

GB10 / DGX Spark 上需要 CUDA 13.0 版 PyTorch：

```bash
conda run -n sam3 python -m pip install --no-cache-dir --force-reinstall \
  -c constraints-sam3.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.12.0+cu130" \
  "torchvision==0.27.0+cu130"
```

后续在 `sam3` 环境安装任何 pip 包时，都带上约束文件：

```bash
conda run -n sam3 python -m pip install --no-cache-dir -c constraints-sam3.txt <package>
```

`opencv-python-headless` 同样提供 `import cv2`，这里只做视频读写和图像处理，不需要 GUI 组件。

YOLOE/传统脚本环境：

```bash
conda activate yoloe
pip install --no-cache-dir -r requirements.txt
```

或重建环境：

```bash
conda env create -f environment.yml
conda activate yoloe
```

训练依赖单独安装：

```bash
pip install -r requirements-train.txt
```

说明：当前机器是 aarch64/Jetson 类环境，`ultralytics` 会依赖 `torch/torchvision`。如果普通 pip 安装 torch 卡住，先使用本机已有的 `isaacsim` 环境验证训练流程，或安装 NVIDIA/Jetson 对应的 PyTorch wheel 后再装 `requirements-train.txt`。

## 1. 视频抽帧

当前新版视频为 1080p，已抽帧到：

```text
data/frames_1080p
```

重新生成：

```bash
conda run -n yoloe python scripts/extract_frames.py \
  --video-dir /home/nvidia/DATA/UW/video \
  --out-dir data/frames_1080p \
  --every-sec 1.0
```

脚本不做 resize，输出帧保持原视频分辨率。

旧版 720p 抽帧命令：

```bash
conda run -n yoloe python scripts/extract_frames.py \
  --video-dir /home/nvidia/DATA/UW/video \
  --out-dir data/frames \
  --every-sec 1.0
```

快速抽样检查：

```bash
conda run -n yoloe python scripts/extract_frames.py \
  --video-dir /home/nvidia/DATA/UW/video \
  --out-dir data/frames_preview \
  --every-sec 2.0 \
  --max-frames-per-video 10
```

## 2. 标注

将筛选后的图片放入：

```text
data/annotated/images
```

对应 YOLO 标签放入：

```text
data/annotated/labels
```

类别定义见 `configs/classes.yaml`：

- `0 damaged`
- `1 exposed_intact`
- `2 suspended_intact`

损伤类别优先于位置：只要确认破损，无论海缆触底还是悬空，都标为 `0 damaged`。

## 3. 生成训练集

```bash
conda run -n yoloe python scripts/make_yolo_dataset.py \
  --images-dir data/annotated_1080p/images \
  --labels-dir data/annotated_1080p/labels \
  --out-dir data/yolo_1080p \
  --val-ratio 0.2
```

## 4. 训练

```bash
conda run -n yoloe python scripts/train_yolo.py \
  --data configs/dataset_1080p.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --device 0
```

训练输出在 `runs/yoloe/yolov8n_cable`。

如 `yoloe` 暂未装好 `torch/ultralytics`，可临时用已有 torch 环境运行同一脚本：

```bash
conda run -n isaacsim python scripts/train_yolo.py \
  --data configs/dataset.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --device cpu
```

## 5. 推理展示

```bash
conda run -n yoloe python scripts/predict_yolo.py \
  --weights runs/yoloe/yolov8n_cable/weights/best.pt \
  --source data/frames \
  --project outputs/predict \
  --name yoloe
```

中期方案说明见 `docs/midterm_plan.md`。

## 6. 长水池扫描视频选代表帧

如果后续使用视觉 Agent 做离线样本级判断，先从视频中为每个样本缓存 3 张代表帧。每个样本只进行一轮分类，不再对低置信度样本追加复核。

当前视频已经被预处理成“每个视频只有一个样本”，推荐先用 SAM3 的 `pipe` 文本提示检出管线并按 mask 裁剪：

```bash
conda run -n sam3 python scripts/select_sample_frames.py \
  --input /home/nvidia/DATA/UW/video \
  --out-dir outputs/sample_frames_sam3_pipe \
  --detector sam3 \
  --crop-source sam3-mask \
  --sam3-prompt pipe \
  --sam3-conf 0.05 \
  --cable-threshold 0.04 \
  --sam3-checkpoint ckpts/sam3/sam3.pt \
  --sam3-dtype float32 \
  --frames-per-sample 3
```

项目当前使用从 ModelScope 下载到本地的 SAM3 checkpoint：

```bash
--sam3-checkpoint ckpts/sam3/sam3.pt
```

YOLOE 的 `beam` 文本提示可作为备选方案：

```bash
conda run -n yoloe python scripts/select_sample_frames.py \
  --input /home/nvidia/DATA/UW/video \
  --out-dir outputs/sample_frames_yoloe_beam \
  --detector yoloe \
  --crop-source yoloe-mask \
  --yoloe-weights ckpts/yoloe/yoloe-v8l-seg.pt \
  --mobileclip-path mobileclip_blt.ts \
  --yoloe-labels beam \
  --imgsz 640 \
  --conf 0.5 \
  --iou 0.4 \
  --frames-per-sample 3
```

先做短测试可加：

```bash
--max-frames 300
```

正式长水池连续扫描时，一个视频里会包含多个间隔 1m 的样本，改用 `scan` 模式：

```bash
conda run -n sam3 python scripts/select_sample_frames.py \
  --input /home/nvidia/DATA/UW/video/long_scan.mp4 \
  --out-dir outputs/sample_frames_scan \
  --detector sam3 \
  --crop-source sam3-mask \
  --sam3-prompt pipe \
  --sam3-conf 0.05 \
  --cable-threshold 0.04 \
  --sam3-checkpoint ckpts/sam3/sam3.pt \
  --sam3-dtype float32 \
  --mode scan \
  --sample-every-sec 1.0 \
  --min-segment-sec 2.0 \
  --merge-gap-sec 2.0 \
  --frames-per-sample 3
```

脚本不缩放保存图像，输出仍为原始 1080p 帧。每个视频会生成：

```text
outputs/sample_frames/<video_name>/
├── frame_metrics.csv
├── manifest.json
└── S001/
    ├── S001_01_*.jpg
    ├── S001_02_*.jpg
    └── S001_03_*.jpg
```

上述阈值针对当前 ModelScope `sam3.pt` 和水池视频标定。更换 checkpoint 或拍摄条件后应重新抽样检查置信度。在 `scan` 模式下，如果同一个样本被切成多段，优先增大 `--merge-gap-sec`，例如 `2.5`；如果相邻两个真实样本被合并，减小 `--merge-gap-sec` 或提高 `--cable-threshold`。如果提前知道长视频里有几个样本，可加 `--expected-samples N`。

## 7. SAM3 + Agent 闭环

完整入口默认使用中期验收模式：SAM3 自动发现长视频中的样本段，把每个样本的多帧全景/局部证据拼成一张图，再使用 Codex Pro 做零样本分类：

```bash
scripts/run_sam3_codex_pipeline.sh \
  --input /home/nvidia/DATA/UW/video \
  --work-dir outputs/final_pool_run \
  --mode scan
```

验收策略位于 `configs/closed_loop.yaml`：

- 默认模型为 `gpt-5.5`；
- 每个样本只向 Agent 提供一张 `contact-sheet`；
- 每个 Codex 批次最多 6 个样本，批次独立缓存并支持断点续跑；
- 每个样本固定使用 3 帧并只分类一轮，低置信度和不确定判断不会再次调用 Codex；
- `position` 使用有效帧多数票；
- `damage=damaged` 只需至少一帧存在高置信度明确损伤，其他视角没有看到损伤不算冲突；
- 最终采用三分类：`0 damaged`、`1 exposed_intact`、`2 suspended_intact`；有损伤时位置不影响类别；
- Agent 只生成 `results.json`，其余 CSV、Markdown 和校验报告由 Python 生成；
- 只要 Agent 给出合法三分类就作为最终结果；质量问题写入 `warnings.json`，不会进入复核队列；
- 相同工作目录、图片、请求和模型会复用每个批次的 `result_cache.json`，不会重复调用 Agent；
- Pro 限额、网络或 Codex 批次异常时立即停止，已完成批次缓存保留；原命令重跑即可续跑；
- 同一工作目录已经存在 `sample_frames/manifest.json` 时会复用 SAM3 分段；输入或参数变化后用 `--force-split` 强制重建。

需要更快的轻量模型时，显式指定：

```bash
scripts/run_sam3_codex_pipeline.sh \
  --input /home/nvidia/DATA/UW/video \
  --work-dir outputs/final_pool_fast \
  --closed-loop-config configs/closed_loop_fast.yaml \
  --model gpt-5.4-mini
```

只检查请求、不调用 Codex：

```bash
scripts/run_sam3_codex_pipeline.sh \
  --samples-dir outputs/final_pool_run/sample_frames \
  --work-dir outputs/closed_loop_check \
  --skip-split \
  --prepare-only
```

主要分类输出包括：

```text
classification/
├── attempt_1/
│   ├── evidence/
│   ├── batches/
│   │   ├── batch_001/
│   │   └── batch_002/
│   ├── requests.json
│   ├── results.json
│   └── validation.json
├── closed_loop_state.json
├── results.json
├── results.csv
├── review.md
├── warnings.json
├── unclassified.json
└── quality_check.md
```

## 8. SAM3 + YOLOE 无训练分类

不调用 Agent/API 的离线路径复用 SAM3 的样本分段和 mask，使用 YOLOE 多文本
提示、池底间隙规则和三帧融合输出相同的三分类：

```bash
scripts/run_sam3_yoloe_pipeline.sh \
  --samples-dir outputs/sam3_codex_pipeline/sample_frames \
  --work-dir outputs/sam3_yoloe_run \
  --skip-split
```

同时运行 Codex 基准和无训练方法并生成逐样本对比：

```bash
scripts/run_sam3_codex_pipeline.sh \
  --samples-dir outputs/sam3_codex_pipeline/sample_frames \
  --work-dir outputs/method_comparison \
  --classifier both \
  --skip-split
```

开放词汇提示和固定阈值位于 `configs/zero_shot_classifier.yaml`。当前 7 个样本
全部跑通，与历史 Codex 三分类结果一致 6 个（`85.71%`）；这是方法一致率，
不是真实标注准确率。实现和限制见 `docs/zero_shot_classifier.md`。

## 9. SAM3-only 无训练分类

只使用 SAM3 模型的离线路径分为两遍：第一遍用 `pipe` 完成视频切分和目标
mask，第二遍在每个样本的 3 张代表帧上运行损伤、完好、绳索反例及白色
PVC/T 形支架提示。白标签只位于海缆两端，所以两端各 20% 的正向候选不参与
破损投票，只有中间 60% 能形成破损证据；绳索遮挡也不计破损。当前验收使用
当前 SAM3-only 模式采用悬空优先：强损伤提示可单帧命中，弱损伤提示需要跨帧
位置一致；位置分支复用 SAM3 `pipe` mask，通过端点附近的垂直亮色支撑杆区分
裸露/悬空，不增加支架 prompt 推理。判为悬空时最终输出类别2；非悬空时再由
损伤结果决定类别0或类别1。

```bash
scripts/run_sam3_only_pipeline.sh \
  --samples-dir outputs/sam3_codex_pipeline/sample_frames \
  --work-dir outputs/sam3_only_run \
  --skip-split
```

从新视频完整运行：

```bash
scripts/run_sam3_only_pipeline.sh \
  --input /path/to/long_video.mp4 \
  --work-dir outputs/sam3_only_run \
  --mode scan \
  --force-split
```

该路径不加载 YOLOE，也不调用 Agent/API。SAM3 提示和阈值位于
`configs/sam3_only_classifier.yaml`，详细流程和限制见
`docs/sam3_only_classifier.md`。当前历史 7 个样本与 Codex 基准一致
`7/7`；这只是基准一致率，不是真实标注准确率。

早期提示词经过 8 样本/24 帧扫测；中期俯拍数据又加入了海缆端部、绳索和
支撑杆几何规则。当前正式提示、分组、阈值和空间排除规则以
`configs/sam3_only_classifier.yaml` 为准；历史候选与排名保留在
`configs/sam3_prompt_search.yaml` 和
`outputs/sam3_prompt_search_grouped/prompt_report.md` 中。

## 10. SAM3 + YOLOE 低帧率实时视频

低帧率入口在每个采样时刻使用 SAM3 生成海缆 mask，YOLOE 默认每 3 秒
更新一次三分类结果，其余采样帧复用当前分类状态。默认输出 1 FPS 中文可视化，
白色标签、绳索和支架作为排除证据；悬空帧中的白色支架不会绘制为海缆 mask。

```bash
conda run -n sam3 python scripts/runtime_sam3_yoloe.py \
  --input /home/nvidia/DATA/UW/mid/clip \
  --output-dir outputs/runtime_mid_clip \
  --target-fps 1
```

主要输出为 `results.json`、`runtime_report.json`、`videos/` 和 `evidence/`。
`runtime_report.json` 同时记录 SAM3、YOLOE、渲染耗时及是否满足源视频实时性。
当前 GB10 实测 1 FPS 可以持续实时，2 FPS 尚不能稳定实时，因此正式默认值为
1 FPS。模型加载只发生一次，报告同时给出冷启动时间和不含加载的稳态时间。

## SAM 辅助标注

小样本阶段建议优先用 SAM 生成海缆 mask，再人工指定三类状态。mask 可通过 `scripts/masks_to_yolo.py` 转成 YOLO 框标签。详细方案见 `docs/sam_labeling_plan.md`。
