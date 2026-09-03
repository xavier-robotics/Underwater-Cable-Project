# 专用水下相机标定工具链

本目录面向完整封装、直接安装到机器人的专用水下相机，只求解两类参数：

1. 相机在实际使用介质中的有效内参；
2. `base_link` 到 `camera_optical_frame` 的固定机械外参。

工具不再要求防水舱尺寸、端口法向、厚度或折射率，也不再单独估计舱体折射
几何。相机的镜头、密封窗和内部装配被视为一个不可拆分的成像单元。

> 这里采用的是“在标定介质中有效的中心投影模型”。水中内参必须用水下数据
> 标定，不能把空气内参直接当成水下内参。如果独立验证显示画面边缘或不同
> 距离存在明显系统误差，说明设备不能被普通针孔/鱼眼模型充分描述，需要厂家
> 的光学模型或重新引入非中心折射模型。

## 环境与安装

项目使用 Python、NumPy、OpenCV、SciPy 和 PyYAML。当前机器使用 `sam3`
conda 环境：

```bash
conda activate sam3
python -m pip install --no-build-isolation --no-deps -e .
camera_calibration --version
```

也可以不激活环境直接运行：

```bash
conda run -n sam3 python -m camera_calibration --help
```

## 网络摄像头采集

`camera_capture` 可从 RTSP 摄像头提供浏览器实时预览，并保存单张图片、定时视频或
固定间隔的图片序列。
密码优先从 `CAMERA_PASSWORD` 环境变量读取；未设置时自动读取项目内仅供本机
使用的 `.camera_password` 文件。该文件已被 Git 忽略，不会提交到代码库。也可用
`CAMERA_PASSWORD_FILE` 指向其他密码文件。密码不会作为命令行参数出现在进程
列表中。

当前直连摄像头可这样配置：

```bash
export CAMERA_HOST=192.168.1.123
export CAMERA_USERNAME=admin
```

抓拍一张、录制 60 秒视频、以及每秒保存一张共 30 张：

```bash
conda run -n sam3 python -m camera_calibration.capture_cli snapshot
conda run -n sam3 python -m camera_calibration.capture_cli record --seconds 60
conda run -n sam3 python -m camera_calibration.capture_cli frames --interval 1 --count 30
```

安装本项目后也可以把上述命令中的
`python -m camera_calibration.capture_cli` 换成 `camera_capture`，或直接运行
`python scripts/capture_camera.py`。默认 RTSP 端口为 `554`，流路径为
`/h264_stream`，传输方式为 TCP；可用 `--port`、`--stream-path`、`--transport`
覆盖。默认输出目录是 `outputs/camera_capture/`，视频保持摄像头原始分辨率，
当前 MP4 录制不包含音频。

在 Spark 本机浏览器实时预览：

```bash
camera_capture preview
```

命令连接成功后会显示 `http://127.0.0.1:8765/`，在浏览器打开该地址即可；按
`Ctrl+C` 停止。默认网页预览为 10 FPS、最大宽度 1280 像素，以降低延迟和 CPU
占用，可用 `--fps`、`--max-width`、`--quality` 和 `--web-port` 调整。这些设置
只影响网页预览，不改变抓拍和录像的原始分辨率。

默认仅监听本机，网页本身不带登录认证。如果确需由同一局域网内的另一台设备
访问，可显式运行：

```bash
camera_capture preview --bind 0.0.0.0
```

然后访问 `http://Spark的局域网IP:8765/`；仅应在可信网络中使用此模式。

指定输出文件或保存 PNG 序列：

```bash
camera_capture snapshot --output outputs/camera_capture/check.png
camera_capture record --seconds 10 --output outputs/camera_capture/check.mp4
camera_capture frames --format png --interval 0.5 --count 20
```

工具默认不会覆盖已有文件；确需覆盖时显式添加 `--overwrite`。长时间采集前应
先持久化直连网口的静态地址，并校正摄像头时间。

## 推荐顺序

```text
1. 相机保持最终镜头、焦点、分辨率和安装状态；
2. 在实际使用的水体中拍摄多距离、多倾角标定板视频；
3. 标定水下有效内参；
4. 使用已知 T_base_target 的工装数据标定机械外参；
5. 使用独立数据验证内参与机械外参；
6. 加载两份结果，把像素射线转换到 base_link。
```

命令为：

```bash
camera_calibration calibrate-intrinsics --config config/intrinsics.yaml
camera_calibration calibrate-extrinsics --config config/extrinsics.yaml
camera_calibration validate --config config/validation.yaml
```

配置文件：

- `config/project.yaml`：项目、相机和坐标系摘要；
- `config/intrinsics.yaml`：水下内参视频、质量门槛和输出；
- `config/extrinsics.yaml`：已知工装位姿、机械边界和全局优化；
- `config/validation.yaml`：独立验证数据和验收阈值。

配置中的相对路径以配置文件所在目录为基准。所有长度输入使用米，报告可额外
显示毫米。

## 1. 水下有效内参

默认配置包含：

```yaml
camera:
  camera_name: underwater_camera
  model: pinhole
  image_width: 1920
  image_height: 1080
  fixed_focus: true
  calibration_environment: underwater
```

`calibration_environment` 会写入 `intrinsics.yaml`。只有相机实际在空气中使用
时才应改成 `air`。

建议拍摄 30–90 秒视频：

- 标定板中心遍历画面中心、四边和四角；
- 包含实际工作范围内的近、中、远距离；
- 包含正对、左右倾斜和上下倾斜；
- 避免运动模糊、反光、严重欠曝及连续静止画面；
- 标定板始终完整可见；
- 固定焦点、曝光模式、分辨率、裁切和电子变焦。

支持视频、图片目录和实时摄像机。视频先按时间间隔取候选帧，再依据检测、
清晰度、曝光、面积、位置、倾角和重复程度选择 20–60 帧。Checkerboard 优先
使用 `findChessboardCornersSB`，不可用时回退到传统检测和亚像素优化。

普通镜头使用 `cv2.calibrateCamera`，鱼眼使用 `cv2.fisheye.calibrate`，两种
参数格式和去畸变接口不会混用。最多执行两轮异常帧剔除；有效帧不足时命令
失败，不会生成伪造结果。

输出：

```text
output/intrinsics/
├── intrinsics.yaml
├── camera_info.yaml
├── report.md
├── observations.csv
├── coverage.png
├── detected/
├── rejected/
├── undistorted/
└── reprojection/
```

## 2. 机械外参

全项目统一使用：

```text
p_A = T_A_B * p_B
```

OpenCV PnP 返回 `T_camera_target`，所以每张图像的候选严格按下式计算：

```text
T_base_camera = T_base_target * inverse(T_camera_target)
```

每个工装数据集必须提供完整 `T_base_target`，包括三维平移和姿态。仅知道
“标定板距离相机 0.5 m”不足以确定六自由度变换。配置还必须明确标定板原点和
轴方向；默认原点为第一个内部角点，`+x` 沿列、`+y` 沿行、`+z` 构成右手系。

建议至少使用 3–6 个已知工装位姿，每个位姿保留 5–10 张清晰图像。为了让
内参和图像模型一致，机械外参数据也应在内参标定所声明的介质与相机设置下
采集。

平面 PnP 使用 IPPE 并保留所有正深度候选，通过重投影误差、跨帧外参一致性
和机械位置范围联合消歧。多帧候选使用鲁棒平移和 SO(3)/四元数均值融合，再
只优化全局 `T_base_camera`。

默认子坐标系是 `camera_optical_frame`：`+x` 向右、`+y` 向下、`+z` 向前。
`camera_link` 是机械坐标系，不会被静默当成光学坐标系。如果已知
`T_camera_link_camera_optical`，配置后会额外输出：

```text
base_link → camera_link → camera_optical_frame
```

输出：

```text
output/extrinsics/
├── extrinsics.yaml
├── report.md
├── observations.csv
├── detected/
├── rejected/
└── reprojection/
```

`extrinsics.yaml` 同时保存 `T_base_camera` 和严格计算的逆变换
`T_camera_base`。

## 运行时组合

运行时只加载两份结果：

```python
from camera_calibration import UnderwaterCameraCalibration

calibration = UnderwaterCameraCalibration.load(
    "output/intrinsics/intrinsics.yaml",
    "output/extrinsics/extrinsics.yaml",
)

ray_camera = calibration.pixel_to_camera_ray(960.0, 540.0)
ray_base = calibration.pixel_to_base_ray(960.0, 540.0)
print(ray_base.origin, ray_base.direction)
```

处理顺序为：

```text
像素
→ 使用水下有效内参去畸变并反投影
→ 从相机投影中心生成单位射线
→ 使用 T_base_camera 转换到 base_link
```

机械外参只描述刚体安装关系，不会被内参替代。输出射线是中心投影模型下的
有效水下射线。

## 验证与合成验收

`validate` 分别检查：

- 内参焦距/主点合理性、中心和角落重投影误差、去畸变预览；
- 外参方向、逆变换、物理位置、重投影和平移/旋转残差；
- 中心及四角像素能否生成归一化 `base_link` 射线。

任何强制阈值不满足时，验证结果写为 `failed` 并返回非零退出码。

不依赖真实图片的完整合成验收：

```bash
conda run -n sam3 python scripts/run_synthetic_acceptance.py \
  --output /tmp/uw_calibration_acceptance
```

脚本在本地生成 AVI，依次运行三个 CLI 命令，比较已知内参/外参真值，检查
报告、CSV、YAML 和调试图片，并加载两份结果生成一条 `base_link` 射线。

单元测试与静态语法检查：

```bash
conda run -n sam3 python -m unittest discover -s tests -v
conda run -n sam3 python -m compileall -q camera_calibration tests scripts
```

## 如何判断结果可用

- `coverage.png` 应覆盖中心、边缘和角落，不应只集中在画面中央；
- 内参验证误差不应随半径、距离或倾角出现明显系统变化；
- 不同工装位置应得到一致的 `T_base_camera`；
- 独立数据应满足任务所需的像素和机械误差；
- 更换焦点、分辨率、裁切、镜头、相机安装或使用介质后必须重新标定。

常见失败原因包括：内部角点行列填反、方格尺寸不准、标定板不完整、连续重复
帧、覆盖不足、运动模糊、水下反光或浑浊、视频分辨率与配置不一致、四元数未
归一化、`T_base_target` 方向错误，以及把 `camera_link` 当作光学坐标系。

当前项目 `ros_version: none`，所以 ROS topic/rosbag 采集会明确报错；应先导出
为无损图片或视频。ChArUco 接口保留 `cv2.aruco` 能力检查，但默认配置未提供
完整板规格，因此不会猜测参数。
