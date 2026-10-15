# Initial 阶段学习率评测观察与决策（2026-08-05）

## 1. 决策摘要

截至本记录冻结时，**主训练不手工提高学习率，也不重置 optimizer/scheduler**。
保持已经验收的学习率轨迹：

- 当前继续使用 `lr=1e-3`；
- 完成第 25,000 次 optimizer update 后，由既有 scheduler 自动乘 `0.95`；
- 第 25,001 次 update 开始实际使用 `9.5e-4`；
- 不在主线上试验 `1.3e-3` 或 AF3 的 `1.8e-3`。

这个决定不是因为高学习率一定会变差，而是因为当前固定 evaluation 仍显示长期改善，
没有证据证明 `1e-3` 限制了收敛；同时，step 22000 刚发生 catalog 与 soft-cluster
采样迁移，尚无完整的迁移后 evaluation。此时改学习率会把数据策略和优化器策略两个
变量混在一起，无法归因。

## 2. 观察对象与冻结边界

### 2.1 当前训练

- 代码仓库：
  `/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02`
- 当前 run：
  `/mnt/project_abd/klp/proteinx_output/protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler_20260804_140441`
- 本次健康快照：完成到约 update `24149`，`lr=1e-3`；
- 最新健康窗口中没有 NaN、non-finite gradient、optimizer skip 或 gradient clip；
- 当前 optimizer 是 Adam，`beta=(0.9, 0.95)`，global target batch 为 `512`；
- 模型仍处于 crop 384、diffusion batch 48 的 initial-training 形态。

### 2.2 固定 evaluation

本次判断使用：

`/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_every600_alltokens1315/latest20_completed_evaluation_cn.md`

评测口径为：

- 锁定的 1,315-target PXM-Legacy AF3-like all-token proxy；
- EMA decay `0.999` checkpoint；
- cycle 4；
- 每个 target 生成 5 samples，每个 sample 运行 200 diffusion steps；
- template cutoff 为 2021-09-30；
- 七个子类使用 `top1_best5_pct`；
- `Macro-7` 对七类等权；
- `Micro` 对所有有效类别记录按记录数加权；
- `complex gPDE rank-1` 使用全复合物 gPDE 选出的 rank-1 结构。

冻结报告只有 u10200 至 u21600 共 20 个完整 checkpoint。u22200、u22800、u23400
和 u24000 当时仍在运行，不能使用未完成 target 前缀计算趋势。尤其需要注意：新的
residual/ge200 catalog 和 gamma=0.2 soft-cluster 从 update 22000 才开始，因此本报告
**尚未包含任何完整的新数据阶段 checkpoint**。

## 3. 总体 evaluation 趋势

### 3.1 长期趋势仍然向上

| 指标 | u10200 | u21600 | 净变化 | OLS 每 1,000 updates | OLS R² |
|---|---:|---:|---:|---:|---:|
| Macro-7 | 62.58 | 64.58 | +2.00 | +0.217 | 0.676 |
| Micro | 71.92 | 73.70 | +1.78 | +0.174 | 0.915 |
| complex gPDE rank-1 | 82.10 | 83.32 | +1.22 | +0.114 | 0.931 |

Micro 和 complex gPDE 的长期趋势稳定，R² 都超过 0.9。Macro 的 R² 较低，主要是
小样本核酸接口类别的 checkpoint 间波动较大，而不是所有类别同时停止学习。

从最早 5 个 checkpoint 到最近 5 个 checkpoint，均值变化为：

- Macro-7：`62.574 -> 64.330`，增加 `1.756`；
- Micro：`72.188 -> 73.686`，增加 `1.498`；
- complex gPDE rank-1：`82.320 -> 83.306`，增加 `0.986`。

因此当前更准确的描述是：**总体结构质量继续改善，但近期速度放缓并进入高位平台波动**，
不是训练发散，也不是已经证明的验证集退化。

### 3.2 u21600 是高位单点回撤，不足以触发 LR 修改

| 指标 | u21000 | u21600 | 单点变化 |
|---|---:|---:|---:|
| Macro-7 | 65.18 | 64.58 | -0.60 |
| Micro | 73.95 | 73.70 | -0.25 |
| complex gPDE rank-1 | 83.41 | 83.32 | -0.09 |

基于相同 1,315 targets 的逐 target 配对 bootstrap：

- Micro 差值约 `-0.246`，95% CI 为 `[-0.513, +0.003]`；
- complex gPDE 差值约 `-0.084`，95% CI 为 `[-0.234, +0.066]`；
- 两个稳定主指标的区间均覆盖零，不能确认真实退化；
- Macro 差值约 `-0.603`，95% CI 为 `[-1.087, -0.153]`，但主要由小样本
  RNA-protein 类别驱动。

此外，u20400、u21000、u21600 三点均值为：

- Macro-7：`64.753`；
- Micro：`73.820`；
- complex gPDE rank-1：`83.383`。

该三点均值作为 step 22000 数据迁移前的参考基线，比只盯住 u21000 单点更稳健。

## 4. 子类层面的关键观察

### 4.1 Macro-7 的噪声来源

u21000 的有效类别记录数为：

| 类别 | 有效记录数 |
|---|---:|
| intra-protein | 1,163 |
| protein-protein | 681 |
| ligand-protein | 396 |
| DNA-protein | 88 |
| intra-DNA | 81 |
| RNA-protein | 46 |
| intra-RNA | 31 |

Macro-7 对这些类别等权，因此只有 31 或 46 条记录的核酸类别，与 1,163 条记录的
intra-protein 在 Macro 中权重相同。Macro 适合暴露弱项，但不适合单独判断全模型
是否应该改变学习率。

u21000 到 u21600 的 Macro 回撤中：

- RNA-protein：`36.51 -> 33.71`，下降 `2.80`，约贡献总 Macro 回撤的三分之二；
- DNA-protein：下降 `0.85`；
- protein-protein：下降 `0.42`；
- ligand-protein：下降 `0.33`；
- intra-protein 基本不变；
- intra-DNA 和 intra-RNA 略有提高。

RNA-protein 只有 46 条记录，且该类别在历史 checkpoint 中曾出现
`37.98 -> 31.85 -> 36.51 -> 33.71` 的明显波动。因此不应把 u21600 的 Macro
下降直接解释为学习率不足或训练已经退化。

### 4.2 局部结构趋稳，接口仍在继续学习

u10200 至 u21600 的长期 OLS 改善速度约为：

- intra-protein：每 1,000 updates `+0.089`；
- protein-protein：每 1,000 updates `+0.256`；
- ligand-protein：每 1,000 updates `+0.243`。

从 u10200 到 u21600，七个类别的 endpoint 全部为正：

- intra-protein `+1.22`；
- protein-protein `+2.32`；
- ligand-protein `+2.48`；
- intra-DNA `+0.80`；
- DNA-protein `+2.24`；
- intra-RNA `+2.42`；
- RNA-protein `+2.50`。

这与 AF3 initial training 中“局部结构先趋稳、接口能力继续较长时间改善”的现象
一致。当前证据支持继续 initial-stage 训练，而不是因为训练 loss 平滑下降变慢就立即
提高学习率。

## 5. Confidence ranking 是值得单独关注的瓶颈

跨这些 checkpoint：

- complex best-of-5 比 gPDE rank-1 通常高约 `0.75-0.94` 个百分点；
- gPDE rank-1 相对随机选一个 sample 通常只提高约 `0.03-0.23` 个百分点。

这说明候选结构本身在持续改善，但 confidence ranking 从 5 个候选中挑出最好结构的
能力仍较弱，尚未充分缩小 oracle-best 与 rank-1 的差距。另一方面，evaluation 中的
pDE loss、pLDDT loss 和 resolved loss 都有长期下降，不能简单称为 confidence head
整体退化。

该现象应作为独立诊断项处理；提高全模型 LR 不会有选择地解决 ranker 问题，也不能
用它作为当前升 LR 的理由。

## 6. 与 AF3 学习率的关系

AF3 公开的 initial-training 配置是：Adam、`beta=(0.9, 0.95)`、前 1,000 updates
线性 warmup 到 `1.8e-3`，之后每 50,000 updates 乘 `0.95`，global target batch
为 256，crop 384，diffusion batch 48。

当前模型使用相同数量级和相同 warmup 思路，但不是严格复现：

- 当前 peak LR 为 `1e-3`，是 AF3 的 55.6%；
- 当前 global target batch 为 512，是 AF3 的两倍；
- 当前 scheduler 每 25,000 updates 衰减一次；
- 当前模型约 529.9M 参数，`c_z=256`、64 层 Pairformer；AF3 是
  `c_z=128`、48 层 Pairformer；
- 数据组成、loss 归一化和具体训练实现也不同。

因此 AF3 的 `1.8e-3` 只能说明 `1e-3` 与其处于同一数量级，不能证明当前模型需要
提高学习率。模型更大本身也不是提高 LR 的原则性理由。

官方来源：

- Nature 主文：<https://www.nature.com/articles/s41586-024-07487-w>
- Supplementary Methods 5.2/5.4：
  <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>

## 7. 正式决定

### 7.1 当前主线

1. 不修改当前 launcher、optimizer 或 scheduler；
2. 不在主训练中途把 LR 提高到 `1.3e-3` 或 `1.8e-3`；
3. 继续使用 `1e-3` 到 update 25000；
4. 允许既有 scheduler 在 update 25000 后自然进入 `9.5e-4`；
5. 保存 u21000 EMA 作为当前已知综合最佳候选，但不据此停止训练；
6. 不使用尚未完成的 evaluation target 前缀做模型或 LR 决策。

### 7.2 下一次重新评估的证据要求

首先等待至少三个完整的 soft-cluster 新阶段 checkpoint：

- u22200；
- u22800；
- u23400；

最好再纳入 u24000 和 u24600。随后观察衰减后的：

- u25200；
- u25800；
- u26400。

重新考虑学习率至少需要满足以下条件：

1. 使用完整 1,315-target evaluation，而不是部分完成结果；
2. Micro 和 complex gPDE 的三点平滑同时转为下降，而不是仅 Macro 下降；
3. 下降超过正常 checkpoint 波动，并在多个连续 checkpoint 上复现；
4. 分库和分子类型分析排除数据迁移、soft-cluster 采样及小样本核酸类别的影响；
5. 训练 loss、gradient health 与固定 EMA evaluation 提供一致证据。

如果以后需要验证更高 LR，只能从同一个 raw checkpoint 创建独立 A/B 分支，保留
完整 optimizer/RNG/sampler 状态并显式记录 scheduler 迁移；主线不能原地修改。
在触发上述条件前，不启动该 A/B。

## 8. 证据路径

- 自动汇总报告：
  `/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_every600_alltokens1315/latest20_completed_evaluation_cn.md`
- u21000 逐 target 原始记录：
  `/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_every600_alltokens1315/af3_pxm_full_combined_formal_u21000_a1_20260804_141917/logs/eval_samples_rank0.jsonl`
- u21600 逐 target 原始记录：
  `/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_every600_alltokens1315/af3_pxm_full_combined_formal_u21600_a1_20260804_184600/logs/eval_samples_rank0.jsonl`
- 当前训练 metrics：
  `/mnt/project_abd/klp/proteinx_output/protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler_20260804_140441/logs/train_metrics.jsonl`
- 当前冻结配置：
  `/mnt/project_abd/klp/proteinx_output/protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler_20260804_140441/config.yaml`
- step 21999 catalog 迁移合同：
  `docs/catalog_replacement_step21999_resume_contract_cn.md`
- soft-cluster 实现与 checkpoint 迁移：
  `docs/soft_cluster_sampling_gamma02_checkpoint_migration_cn.md`

