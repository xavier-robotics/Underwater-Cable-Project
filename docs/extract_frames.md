# 视频抽帧使用说明

项目已有程序：[`scripts/extract_frames.py`](../scripts/extract_frames.py)。启动脚本：[`run_scripts/extract_frames.sh`](../run_scripts/extract_frames.sh)。

## 直接运行

在项目根目录执行，使用当前激活的 Python 环境（例如你已有的 `sam3` 环境）。需要 OpenCV 和 tqdm；缺少依赖时安装：

```bash
python -m pip install opencv-python-headless tqdm
```

把视频放进项目的 `video/` 文件夹，然后运行：

```bash
bash run_scripts/extract_frames.sh
```

默认每 **1 秒**抽取一张，从第 0 帧开始，保存到 `data/frames/`，保留视频原始分辨率，不执行颜色增强。

## 指定目录和抽帧间隔

例如处理服务器上的视频，每 0.5 秒抽一张：

```bash
bash run_scripts/extract_frames.sh \
  --video-dir /home/nvidia/DATA/UW/video \
  --out-dir data/frames_05s \
  --every-sec 0.5
```

先试跑，每个视频最多保存 10 张：

```bash
bash run_scripts/extract_frames.sh \
  --max-frames-per-video 10 \
  --out-dir data/frames_test
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--video-dir` | 项目 `video/` | 输入目录，会递归扫描子目录；不能直接填写单个视频文件 |
| `--out-dir` | 项目 `data/frames/` | 输出目录 |
| `--every-sec` | `1` | 抽帧间隔，单位秒，请填写正数 |
| `--max-frames-per-video` | 不限制 | 每个视频最多保存的图片数，请填写正整数 |

支持 `.mp4`、`.mov`、`.avi`、`.mkv`、`.m4v`。脚本使用当前环境的 `python`，不依赖 SAM3 模型、权重或 GPU。相对路径均相对于项目根目录解释。

## 输出结果

假设输入为 `video/demo.mp4`：

```text
data/frames/
├── demo/
│   ├── demo_f000000_t00000.00.jpg
│   └── ...
└── manifest.json
```

文件名记录原视频名、帧编号和时间（秒）；`manifest.json` 记录视频 FPS、分辨率、保存数量等信息。抽帧间隔按 FPS 换算成整数帧步长，因此时间可能存在取整偏差；读取不到 FPS 时，现有程序每 30 帧抽一张，文件名时间记为 0。

重复运行会覆盖同名图片，但不会清理旧图片。改变间隔时建议换输出目录。不同子目录中的同名视频（包括去掉扩展名后同名、空格替换为下划线后同名）会共用输出子目录，应分开运行并指定不同输出目录。

也可直接运行原 Python 程序；它原本的默认输入是服务器路径，所以建议显式传入目录：

```bash
python scripts/extract_frames.py \
  --video-dir video --out-dir data/frames --every-sec 1
```
