# Agent 闭环分类

当前流程把“看图分类”和“质量记录”拆成两个独立角色：

1. SAM3 从视频中定位管线、划分样本段，并缓存每段 3 张代表帧。
2. 把每帧的 `context` 全景和 `detail` mask/crop 拼成一张 `contact-sheet`。
3. 样本按每批最多 6 个拆分，每个 Codex 批次独立执行、校验和缓存。
4. Agent 每个样本只读取一张拼图，并且只负责生成 `results.json`。
5. `validate_agent_results.py` 独立检查结果并记录质量警告，但不触发第二轮分类。
6. 只要 `position` 和 `damage` 合法，就接受对应三分类作为最终结果；错判风险由使用方接受。
7. 低置信度、弱证据和帧间分歧写入 `warnings.json`，不进入人工复核队列。
8. 请求、图片和模型未变化时，控制器直接复用批次缓存结果。
9. Codex 批次返回非零状态时立即停止，保留之前的批次缓存，原命令重跑即可续跑。
10. 主流水线默认复用同一工作目录中的 SAM3 manifest；需要重算时使用 `--force-split`。

校验项包括：

- `position`、`damage` 枚举是否合法；
- `class_id` 和 `class_name` 是否符合三类映射；
- 两项置信度是否都达到默认阈值 `0.65`；
- `needs_review` 是否为真会记录为警告，但不改变最终分类；
- Agent 是否为每个 `frame_idx` 返回一个结构化投票，以及投票是否覆盖全部帧；
- `position` 是否达到配置的多数票比例并与最终类别一致；
- `damage=damaged` 是否至少有一帧达到明确损伤阈值；
- 只要 `damage=damaged`，是否统一映射到 `class_id=0, class_name=damaged`，而不再区分位置；
- 其他视角的 `no_visible_damage` 不会否定已有的明确损伤证据；
- 请求帧数、图片文件、`attempt_id` 和 `image_paths` 是否完整一致。

策略配置：

```yaml
max_attempts: 1
confidence_threshold: 0.65
position_majority_threshold: 0.6
clear_damage_threshold: 0.75
retry_policy: none
finalization_policy: best_effort
cache_results: true
agent_batch_size: 6

rounds:
  - max_frames_per_sample: 3
    min_frames: 3
    evidence_mode: contact-sheet
```

默认 `configs/closed_loop.yaml` 与 `configs/closed_loop_acceptance.yaml` 都是单轮尽力分类策略。需要进一步节省 Pro 额度时，使用 `configs/closed_loop_fast.yaml` 并指定 `--model gpt-5.4-mini`。技术故障不会被伪装成分类结果：缺失或非法的聚合类别会写入 `unclassified.json`。
