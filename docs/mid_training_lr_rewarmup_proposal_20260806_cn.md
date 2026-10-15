# 中途学习率 Re-warmup 备选方案（2026-08-06）

## 0. 当前状态

本文只保存一个后续可选实验方案，**尚未修改代码、尚未启用，也不改变当前主训练的
optimizer 或 scheduler**。先等待当前训练和固定评测结果，再决定是否实施。

当前相关对象为：

- 代码仓库：
  `/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02`
- 当前 run：
  `/mnt/project_abd/klp/proteinx_output/protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler_20260804_140441`
- 可作为未来分叉点的 raw checkpoint：
  `checkpoints/24999.pt`
- 对应 EMA checkpoint：
  `checkpoints/24999_ema_0.999.pt`

对 `24999.pt` 的状态核查结果为：

| 状态 | checkpoint 中的值 |
| --- | ---: |
| checkpoint `step` | `24999` |
| resume 后下一次 trainer update | `25000` |
| scheduler `last_epoch` | `25000` |
| optimizer 当前 LR | `9.5e-4` |
| scheduler `_last_lr` | `9.5e-4` |

因此未来如果从这里启动 re-warmup，起点必须读取 checkpoint 中的实际
`9.5e-4`，不能根据文件名假定仍为 `1e-3`。

## 1. 背景和目的

当前主训练在前 1,000 个 optimizer updates 从 0 warmup 到 `1e-3`，并在
25,000-update 边界按既有 AF3-like scheduler 乘 `0.95`。该 25,000-update
边界按 global target batch 512 的数据曝光量，对应 AF3 global batch 256 下的
50,000-update 边界。

AF3 论文与附件只公开了固定的 `50,000 steps × 0.95` 调度，没有证据说明该
衰减由 validation plateau 触发。AF3 的 validation 用于训练集采样权重调整、
early stopping 和最佳 checkpoint 选择，而不是公开的动态 LR 控制：

- 主文：<https://www.nature.com/articles/s41586-024-07487-w>
- Supplementary Methods 5.4、5.7：
  <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>

本方案的目的不是宣称当前 LR 太低，而是预先定义一个可复现、连续且可回滚的
中途升 LR 方法。只有后续固定 evaluation 支持尝试更高 LR 时才实施。

## 2. 推荐的 re-warmup 轨迹

首选方案是一次性 cosine up-ramp：

```text
scheduler update 25000: 9.5e-4
25000 -> 26000:         cosine 平滑升到 1.10e-3
26000 -> 50000:         保持 1.10e-3
```

设：

- `S` 为 re-warmup 起始 scheduler update；
- `W=1000` 为成功 optimizer updates 数；
- `lr0` 为 checkpoint 中实际 LR；
- `lr_peak=1.10e-3`。

在 `0 <= k <= W` 时：

```text
lr(k) = lr0 + 0.5 * (lr_peak - lr0) * (1 - cos(pi * k / W))
```

cosine up-ramp 在起点和终点的斜率都为 0，比直接跳变或带端点折角的线性上升更
平滑。这里的 re-warmup 也不同于初始 warmup：它从当前 LR 连续上升，绝不重新
降到 0。

### 2.1 目标 LR 档位

| 目标 LR | 相对 `9.5e-4` | 定位 |
| ---: | ---: | --- |
| `1.0e-3` | `+5.26%` | 风险低，但信号可能太小，没有必要专门做复杂实验 |
| `1.1e-3` | `+15.79%` | 推荐的第一个有效实验档位 |
| `1.2e-3` | `+26.32%` | 只适合后续独立实验 |
| `1.3e-3` | `+36.84%` | 不应直接用于当前主线 |

如果未来仍希望达到 `1.3e-3`，应先完成 `1.10e-3` 阶段并取得稳定训练和固定评测
证据，再用第二段 1,000--2,000 update ramp 逐步升到 `1.2e-3`，最后才考虑
`1.3e-3`。不采用从 `9.5e-4` 一步跳到 `1.3e-3` 的做法。

## 3. 必须保留的训练状态

实施 re-warmup 时只改变 LR policy，以下状态必须完整恢复，不能重置：

- raw model parameters；
- EMA `0.999` shadow parameters；
- Adam 一阶、二阶 moments 和 optimizer step；
- GradScaler；
- trainer/global step；
- scheduler 成功更新计数；
- 各 rank RNG 状态；
- dataloader、sampler 和下一条训练样本 cursor。

重置 Adam 或 scheduler step 会把“只测试 LR 曲线”变成另一个优化器实验，也会
破坏与原训练轨迹的可比性。

## 4. 实现约束

当前 `AlphaFold3LRScheduler` 只支持一次初始 warmup 和固定阶梯衰减。strict full
resume 会先恢复 optimizer，再加载 checkpoint 中完整的 scheduler state。因此：

1. 不能只在 AIStudio launcher 中覆盖 `--lr`；旧 checkpoint state 会覆盖普通配置。
2. 不能通过增大 `warmup_steps` 或重置 scheduler 实现中途升温。
3. 需要新增显式、可写入 `state_dict` 的 piecewise re-warmup policy，或向现有
   scheduler 增加等价且经过测试的参数。
4. re-warmup 必须锚定 checkpoint 中的 `scheduler.last_epoch`。scheduler 只在
   optimizer 真正成功更新时前进，若发生 optimizer skip，trainer step 与 scheduler
   step 可能不同。
5. migration 必须记录并校验 `start_scheduler_step`、`lr0`、`lr_peak`、`W` 和
   ramp 后策略；resume contract 不一致时应 fail closed。
6. 应新建一次性 migration launcher，不能原地改变现有 step-21999 或 strict-resume
   launcher 的训练身份。

未来实现预计涉及：

- `protenix/utils/lr_scheduler.py`：piecewise re-warmup 曲线及持久化状态；
- `runner/train.py`：加载旧 scheduler 后的一次性、严格校验的 LR-policy migration；
- `tests/test_lr_schedule.py`：边界、连续性、skip 语义和 resume round-trip；
- 新的 migration launcher 和对应文档/测试。

## 5. 训练健康与固定评测要求

实施前必须保留不可变的起始 raw/EMA checkpoint。训练启动后至少检查：

- non-finite loss/gradient；
- optimizer skip；
- total gradient norm、p50/p95 和 clip 比例；
- total、diffusion、distogram 和 confidence 各项 loss；
- 200/500-update 平滑趋势，而不是单个 update 的尖峰。

建议固定 evaluation 节点为：

```text
S
S + 600
S + 1200
S + 3000
```

EMA decay 为 `0.999`，半衰期约 693 updates，约 3,000 updates 后才基本反映新的
训练轨迹。因此不能只用 ramp 刚结束时的一个 EMA checkpoint 判断效果。

出现以下情况时应回滚到不可变起始 checkpoint：

- NaN、non-finite gradient 或异常 optimizer skip；
- gradient norm/clip 频率持续显著恶化；
- 500-update 平滑 total loss 相对 ramp 前窗口持续升高约 10%--15%；
- 完整固定 evaluation 在多个 checkpoint 上稳定变差，而非单点波动。

## 6. 启用前的决策条件

当前决定是继续观察，不实施 re-warmup。至少等待：

1. 25,000 边界前后的完整训练 loss、gradient health 和速度数据；
2. 完整的固定 1,315-target evaluation，而不是未完成 target 前缀；
3. 至少多个衰减后 EMA checkpoint，以区分正常波动和持续平台/退化；
4. 对 update 22,000 开始的 residual/ge200 catalog 与 soft-cluster 数据迁移影响已有
   足够观察，避免把数据策略和 LR 策略混在一起解释。

如果现有 `9.5e-4` 轨迹仍稳定改善，则不需要为了“提高学习速度”而自动启用更高
LR。只有证据显示优化速度受限且训练健康有余量时，才从同一不可变 checkpoint 创建
明确命名的 `1.10e-3` re-warmup 分支。

## 7. 本轮明确不做的事情

- 不修改当前运行任务；
- 不修改任何已有 launcher 默认值；
- 不实现或启用 scheduler 代码；
- 不重置 optimizer、EMA、step 或采样状态；
- 不直接把 LR 升到 `1.3e-3`；
- 不因保存本方案而中断当前训练。
