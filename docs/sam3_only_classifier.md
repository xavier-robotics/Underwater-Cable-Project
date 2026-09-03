# SAM3-only 无训练分类

## 目标

该路径不加载 YOLOE、不调用 Codex/API，也不训练项目数据。唯一使用的学习模型
是本地 `ckpts/sam3/sam3.pt`；OpenCV 只负责图像读写和 mask 几何规则。

输出类别与其他路径相同：

| class_id | class_name | 规则 |
|---:|---|---|
| 0 | `damaged` | 非悬空，并达到损伤判据 |
| 1 | `exposed_intact` | 触底且没有达到损伤判据 |
| 2 | `suspended_intact` | 判为悬空，损伤结果不再覆盖该类别 |

## SAM3 与规则流程

```text
长视频
  -> SAM3(pipe)：海缆出现区间、mask、crop、每个样本 3 帧
  -> SAM3(损伤/完好/绳索提示)：局部损伤证据
  -> OpenCV(SAM3 mask 主轴 + 端点支撑杆)：位置证据
  -> 海缆两端排除 + 绳索排除 + 支撑杆几何规则
  -> 三帧融合和三分类
```

第一遍沿用 `scripts/select_sample_frames.py`。分类阶段对每张 crop 只计算一次
SAM3 图像特征，然后依次切换多个损伤文本提示。悬空判定直接复用第一遍的
`pipe` mask，不再运行支架文本提示。

## 损伤判定

当前损伤提示分为三组：

- 金属损伤标记：`metal patch on pipe`；
- 管体损伤：`damaged pipe`、`torn pipe`；
- 材料外露：`exposed material`。

中期水池验收使用 `direct_prompt_threshold`，只有 `metal patch on pipe` 直接
参与最终损伤判定；其余词保留在诊断证据中，便于以后加入真实护套破损数据后
重新标定。

白色编号标签只位于海缆两个物理端部。因此不再让“标签提示”和“破损提示”
竞争分数，而是用第一遍 `pipe` mask 的主轴建立 0–1 位置坐标：外侧两端各
20% 是端部标签区，中间 60% 是有效破损区。正向候选只要落在端部就直接忽略；
只有中段候选能投破损票。这样“两端有异常但中间完好”的样本一定排除破损类。

跨过、绑住或遮住管体的绳子、细线和绿色水下绳索仍是非损伤反例。只有绳索
mask 与正向候选空间重合且分数达到阈值时才压制候选，画面其他位置出现绳子
不会否决真实中段破损。

早期候选词在 8 个样本、24 帧上扫测过 66 个组合；当前中期配置又针对
30 张带真值图像收敛为上述短提示。它同时保留模拟贴片和真实外皮损伤的语义，
而不是只使用抽象的 `damaged`。

同一证据组允许配置多个近义词，但组内只取最高分；只有不同证据组之间才可能
增加一致性分，防止 `silver patch`、`metallic patch` 等近义词重复计算。
组得分必须先达到 `0.020` 才能参与一致性加分，多个低分噪声不能拼成损伤。

SAM3 返回的正向候选需要满足：

1. mask 面积位于配置范围内；
2. 候选 mask 与第一遍 `pipe` mask 的重合率达到阈值；
3. 候选至少覆盖一定比例的第一遍 `pipe` mask，排除边缘小伪影；
4. 一般正向区域最多覆盖第一遍管体 mask 的 35%，防止把“整根黑管”误当
   成 `torn pipe`；
5. 候选中心必须位于海缆长轴中间 20%–80%，两端候选直接忽略；
6. 与绳索区域重合的候选会被排除；
7. 同组近义词只取最高分，达到门槛的不同证据组同时命中才增加一致性分。

整条海缆 mask 即使被损伤提示命中，也只记录为诊断信息，默认不能
直接触发损伤。原因是 SAM3 的全局提示分无法可靠区分“识别到一条海缆”和
“整条海缆确实破损”。

视频中 `metal patch on pipe >= 0.05` 可单帧直接确认；`0.026–0.05` 的弱响应
必须至少两帧在海缆主轴相近位置重复出现。单张 GT 图使用 `0.0275` 门槛。
这是针对水管上金属贴片模拟损伤的验收配置，通用真实护套破损仍需另建独立
测试集验证，不能把本组调参结果直接当作跨场景泛化精度。

## 位置判定

`position.enabled: true`。中期视频是俯视机位，不能可靠使用“底部间隙”或
管体面积推断高度；实验中的悬空样本由靠近海缆端部的横向支撑杆托起。因此
位置分支在 SAM3 `pipe` mask 上计算主轴和两个端点，并用 OpenCV 搜索亮色长线。

一条线必须同时满足：

1. 与海缆主轴近似垂直；
2. 靠近海缆任一端点，而不是穿过海缆中段；
3. 大部分线段位于 pipe mask 外；
4. 长度、亮度对比和 mask 细长度达到配置阈值；
5. 综合几何分数至少为 `0.11`。

任一采样帧命中支撑杆，样本位置记为 `suspended`；否则记为 `exposed`。
该规则没有额外模型调用。最终类别采用悬空优先：位置为 `suspended` 时直接
输出类别2；只有位置为 `exposed` 时，才由损伤结果在类别0和类别1之间选择。
`suspended_intact` 名称为兼容现有数据 schema 保留，当前规则中的实际含义是
“悬空优先类别”。

已有逐帧结果可以不运行 GPU，直接改为“一帧命中即破损”：

```bash
conda run -n sam3 python scripts/reaggregate_sam3_any_hit.py \
  --input-results /path/to/results.json \
  --out-dir /path/to/any_hit_results
```

## 运行

复用已有 SAM3 样本：

```bash
scripts/run_sam3_only_pipeline.sh \
  --samples-dir outputs/sam3_codex_pipeline/sample_frames \
  --work-dir outputs/sam3_only_run \
  --skip-split
```

处理一个新长视频：

```bash
scripts/run_sam3_only_pipeline.sh \
  --input /path/to/long_video.mp4 \
  --work-dir outputs/sam3_only_run \
  --mode scan \
  --force-split
```

只生成待处理清单、不加载模型：

```bash
scripts/run_sam3_only_pipeline.sh \
  --samples-dir outputs/sam3_codex_pipeline/sample_frames \
  --work-dir outputs/sam3_only_check \
  --skip-split \
  --prepare-only
```

## 输出

`classification_sam3_only/` 包含：

- `results.json`、`results.csv`：最终三分类和逐帧证据；
- `summary.json`：类别计数；
- `review.md`：可读结果表；
- `diagnostics/`：第一遍 pipe mask 轮廓、第二遍提示 mask、框和分数；其中
  `*_damage_full.jpg` 是在原始完整帧上叠加的局部损伤 mask，展示时会过滤
  覆盖海缆主体比例过大的提示结果；`*_support_geometry.jpg` 用绿色显示通过
  阈值的端点支撑杆，用橙色显示低于阈值的候选线；
- `requests.json`：实际参与分类的样本和帧；
- `result_cache.json`：输入、配置、脚本和 checkpoint 未改变时复用结果。

## 中期真值

- `/home/nvidia/DATA/UW/mid/gt/sampleNN_CLASS.jpg` 的后缀是真值：
  `0=damaged`、`1=exposed_intact`、`2=suspended_intact`；
- 项目输出的 `class_id` 已与真值后缀统一，可以直接比较；评测脚本仍优先按
  `class_name` 读取，以兼容修改前生成的旧结果；
- 30 个样本的真值分布为：破损 19、裸露 9、悬空 2；
- 真值只用于独立评测，不作为分类器运行时输入。评测命令：

```bash
conda run -n sam3 python scripts/evaluate_midterm_results.py \
  --results outputs/sam3_only_mid_clip_20260813/classification_sam3_only_final/results.json \
  --gt-dir /home/nvidia/DATA/UW/mid/gt
```

## 历史提示词扫测结论

本节记录早期 8 样本数据上的搜索过程，不代表当前中期数据的最终规则。当前
损伤与位置提示、标签/绳索排除和白色支架规则均以本文前述章节及
`configs/sam3_only_classifier.yaml` 为准。

完整候选与结果位于 `configs/sam3_prompt_search.yaml` 和
`outputs/sam3_prompt_search_grouped/prompt_report.md`。

复用已有 GPU 扫测缓存重新生成排名：

```bash
conda run -n sam3 python scripts/tune_sam3_prompts.py \
  --input outputs/sam3_codex_pipeline/sample_frames \
  --input outputs/new_video_20260729_zero_shot/sample_frames \
  --reference-results outputs/sam3_codex_pipeline/classification/results.json \
  --reference-results outputs/new_video_20260729_zero_shot/classification_zero_shot/results.json \
  --out-dir outputs/sam3_prompt_search_grouped \
  --evaluate-only
```

只有新增/删除候选词或更换输入帧时才移除 `--evaluate-only` 重新进行 GPU
扫测。

- 定位：`cylindrical object` 在已有正样本上的平均 mask IoU 最高
  （0.589），`pipe` 为 0.566，`cable` 为 0.561；但通用词产生的区域更大，
  当前又缺少真正无海缆视频验证误检，因此正式第一遍仍使用更具体的 `pipe`。
- 损伤：`silver patch` 的样本级分数最高，但会在贴纸不可见的第 174 帧产生
  弱局部响应，因此没有直接部署。最终组合在 8 个样本上为 `8/8`，两个完好
  样本共 6 帧零阳性，且每个损伤样本至少有 2 帧阳性。
- 贴纸帧：`metal patch on black pipe` 在第 609、1044 帧命中，在第 174
  帧低于门槛；`broken cable jacket` 提供对应的真实结构损伤描述。
- 位置：`cable`、`pipe`、`beam`、`hose`、`tube` 都是 `7/8`，没有
  词族改善结果；保留语义最准确的 `cable`。

报告中的 raw search-score leader 只用于诊断，不直接部署。数据仅 8 个样本，
如果完全按样本内分数选择，会出现 `intact beam` 这类语义不合理但碰巧高分
的组合；正式配置额外要求语义合理、完好帧零误触发和损伤跨帧稳定。

`7/7` 是与 Codex 基准的一致率，不是有人工真值支撑的准确率。样本数量仍然
很少，更换机位、水质、光照和海缆外观后需要重新检查。

## 限制

- SAM3 是开放词汇分割模型，不是经过项目标定的损伤分类器；
- 不同文本提示的分数不是严格校准后的类别概率；
- 过小 mask、画面边缘和反光可能产生伪提示，因此必须结合第一遍 pipe mask；
- 细小划痕和低对比裂口可能没有独立 mask；
- 当前悬空规则依赖实验用白色 PVC/T 形支架；更换支架颜色或结构需要更新
  几何/亮度阈值；如果真实海上悬空场景没有可见支撑结构，需要改用跨帧视差、
  声呐或深度信息，不能直接套用当前水池 trick；
- 配置调整只改变固定规则，不会更新 SAM3 权重。
