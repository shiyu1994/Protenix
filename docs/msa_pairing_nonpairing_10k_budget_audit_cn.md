# 训练 MSA 四项风险审计：深度、source 预算、pairing 模式与去重

## 1. 文档目的

本文记录当前训练代码中四项可能影响模型效果的 MSA 行为，供工程师确认其是否符合设计意图。其中重点
展开 paired/unpaired raw MSA 的 10k source 预算竞争，另外三项在第 11 节简要记录。

核心问题不是简单的“paired 10k + unpaired 10k，最后裁剪到 16,384”，而是：当前配置会把
`pairing.a3m` 同时读入 paired raw pool 和 unpaired raw pool；在 unpaired raw pool 内，
`pairing.a3m` 又排在真正的 `non_pairing.a3m` 前面，并与后者共享同一个 10k 预算。因此，深度较大的
`pairing.a3m` 可能让 `non_pairing.a3m` 完全没有机会进入训练特征。

本文只记录和分析现状，尚未修改实现。

## 2. 当前结论

当前存在三层不同的 MSA 深度预算：

1. paired raw pool 独立上限 10,000；
2. unpaired raw pool 独立上限 10,000；
3. species pairing 和 cleanup 后，最终 assembled MSA 总深度不超过 16,384，其中 paired 最多 8,192。

“paired raw 10k”和“unpaired raw 10k”本身是独立预算，前者不会直接从后者扣减。但是，当前 unpaired
source 配置是 `pairing-non_pairing`，加载器会先用 `pairing.a3m` 消耗 unpaired 的 10k 预算，再考虑
`non_pairing.a3m`。所以从 MSA 来源组成上看，pairing-source rows 确实会抢占 genuine non-pairing rows。

此外，在最终 assembled MSA 中，paired rows 还会占用 16,384 总预算，剩余容量才提供给 unpaired rows。

## 3. 生效配置

当前训练 run：

```text
/mnt/project_abd/klp/proteinx_output/
protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler_20260804_140441/config.yaml
```

其中 MSA 配置为：

```yaml
max_input_paired_msa_depth: 10000
max_input_unpaired_msa_depth: 10000
max_size:
  train: 16384
  test: 16384

prot_pairing_dbs:
  - pairing
prot_non_pairing_dbs:
  - pairing-non_pairing
```

这里的 `pairing-non_pairing` 不是一个单独的文件名。加载器会按 `-` 拆分，得到以下有序 source 列表：

```text
pairing.a3m
non_pairing.a3m
```

## 4. 代码实际数据流

### 4.1 Raw source 读取

入口位于：

```text
protenix/data/msa/msa_featurizer.py
MSASourceManager.fetch_msas()
```

paired 和 unpaired 分别维护：

```text
paired_remaining   = 10000
unpaired_remaining = 10000
```

实际读取顺序可以概括为：

```text
                         ┌─ pairing.a3m，最多 10k ──> paired raw pool
同一条蛋白质序列的目录 ─┤
                         └─ pairing.a3m
                            然后 non_pairing.a3m ───> unpaired raw pool，共享 10k
```

unpaired 分支每读入一个 source 后都会执行：

```text
unpaired_remaining -= raw_msa.depth
```

当 `unpaired_remaining <= 0` 时直接停止，不再读取后面的 source。对应实现位于：

- `protenix/data/msa/msa_featurizer.py:845-865`
- 配置传入位置：`protenix/data/msa/msa_featurizer.py:2291-2303`

读取阶段使用 `dedup=False`。后续虽然会对 unpaired merge 做去重，但被前缀 10k 截断而没有读入的
`non_pairing.a3m` 不会重新参与去重或补位。因此，pairing-source 中的重复行也可能先消耗 raw 预算，
进一步降低实际 unique/Neff。

### 4.2 Species pairing 和 cleanup

在标准 species-paired 模式下，代码先进行跨链物种配对，然后从每条链的 unpaired 特征中删除已经进入
paired block 的相同行：

```text
pair_chains_by_species(...)
cleanup_unpaired_features(...)
```

对应位置：

- `protenix/data/msa/msa_featurizer.py:1134-1141`
- `protenix/data/msa/msa_utils.py:663-747`
- `protenix/data/msa/msa_utils.py:750-764`

`cleanup_unpaired_features()` 只执行删除，不会回到磁盘读取此前因 10k 预算而跳过的
`non_pairing.a3m`，因此 cleanup 后释放出的容量不会被 genuine non-pairing rows 回填。

### 4.3 最终 16,384 总预算

最终组装阶段使用：

```text
max_paired = 16384 // 2 = 8192
final_unpaired <= 16384 - final_paired
```

对应位置：

- `protenix/data/msa/msa_featurizer.py:1135`
- `protenix/data/msa/msa_featurizer.py:1155-1169`
- `protenix/data/msa/msa_featurizer.py:1213-1216`

所以最终确实满足：

```text
final_paired + final_unpaired <= 16384
```

但这个最终约束不能修复 raw source 读取阶段对 `non_pairing.a3m` 的饿死；它只会进一步让 paired 和
unpaired 在最终总预算内竞争。

## 5. 具体示例

假设某条抗体链原始文件深度为：

```text
pairing.a3m      = 12,000
non_pairing.a3m  = 1,500
```

当前 raw 读取行为近似为：

```text
paired raw pool:
  pairing-source     10,000

unpaired raw pool:
  pairing-source     10,000
  non-pairing-source      0
```

随后 species pairing 可能从 paired pool 中选择若干行；cleanup 会从 unpaired pool 删除相同序列，
但不会用未读取的 1,500 条 genuine non-pairing rows 回填。

更一般地，真正 non-pairing source 在 raw 阶段能够保留的行数上界近似为：

```text
loaded_pair_as_unpair = min(raw_pairing_depth, 10000)

retained_non_pairing <= min(
    raw_non_pairing_depth,
    max(0, 10000 - loaded_pair_as_unpair),
)
```

因此，只要 `raw_pairing_depth >= 10000`，当前 source 顺序预测
`retained_non_pairing = 0`。

## 6. 与抗体-抗原统计的关系

### 6.1 用户提供的 104-target 效果联表

用户提供的 104 个有完整 MSA/效果统计的抗体-抗原 target 中，原始抗体链 MSA 深度中位数为：

| 链与来源 | 中位数 | 范围 |
| --- | ---: | ---: |
| H pairing | 10,716 | 8,099–13,578 |
| H non-pairing | 1,521 | 898–2,019 |
| L pairing | 11,302 | 8,767–13,289 |
| L non-pairing | 1,510 | 999–3,045 |

H 和 L 的 pairing 中位数都超过 10k，因此至少从深度条件上看，大量抗体链满足“pairing-source 可能耗尽
unpaired 10k、genuine non-pairing 保留为 0”的条件。

这能解释为什么只看最终 MSA 总深度可能看不到问题：最终仍可由大量 pairing-source rows 填得很深，
但来源组成、unique sequence 数、Neff 和 species 多样性已经发生变化。

目前这仍是代码机制加统计深度得到的强假设，而不是逐 target 已验证事实。需要增加 source-level
instrumentation 才能确认每条链究竟保留了多少 genuine non-pairing rows。

注意：当前 workspace 中未找到这份 104-target FT749/U21000 delta 与 final paired/unpaired 联表的原始
CSV/JSON；本节数值来自用户提供的分析结果，暂时无法在本地重新联表复核。不要把下面独立统计的
528-target PXM-22to25-Ab-Ag 与该 104-target cohort 混为一组。

### 6.2 PXM-22to25-Ab-Ag 全量 528-target 的 source 配额审计

已对 PXM-22to25-Ab-Ag 全量 528 targets、835 个抗体 sequence entity occurrences 执行只读的
depth-only source 配额模拟。该模拟严格应用以下 raw-row 预算顺序：

```text
paired pool:   pairing.a3m，最多 10,000 行
unpaired pool: pairing.a3m 优先，剩余额度才读 non_pairing.a3m，共最多 10,000 行
```

结果如下：

| 抗体角色 | 实体数 | pairing 中位数 | non-pairing 中位数 | non-pairing homolog 读取 0 条 | source 完整读取 | homolog 读取上界 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 | 835 | 10,438 | 1,513 | 522（62.5%） | 43（5.1%） | 18.4% |
| H | 544 | 10,180 | 1,509 | 296（54.4%） | 40（7.4%） | 22.6% |
| L | 279 | 11,035 | 1,505 | 215（77.1%） | 3（1.1%） | 11.7% |
| scFv | 12 | 12,103 | 2,695 | 11（91.7%） | 0（0.0%） | 5.0% |

Target 级结果：

- 344/528（65.2%）至少有一个抗体实体完全读不到 genuine non-pairing；
- 275/528（52.1%）所有抗体实体都完全读不到 genuine non-pairing；
- 510/528（96.6%）至少有一个抗体实体损失 genuine non-pairing；
- 只有 18/528（3.4%）所有抗体实体都能完整读入 genuine non-pairing。

可复现脚本和产物：

- `scripts/evaluation/summarize_pxm22to25_abag_training_msa_budget.py`
- `tests/test_pxm22to25_abag_training_msa_budget.py`
- `/mnt/project_abd/klp/proteinx_eval_data/PXM-22to25-Ab-Ag/analysis/training_msa_budget_pairing_first_10k/antibody_entity_training_raw_budget.csv`
- `/mnt/project_abd/klp/proteinx_eval_data/PXM-22to25-Ab-Ag/analysis/training_msa_budget_pairing_first_10k/summary.json`
- `/mnt/project_abd/klp/proteinx_eval_data/PXM-22to25-Ab-Ag/analysis/training_msa_budget_pairing_first_10k/README.md`

这组结果确认 source-order 风险在全量数据上不是少数边缘情况，但仍不是 checkpoint 回退的因果证据。
它只模拟 raw source 配额，尚未逐条复现 query validation fallback、prefix 去重、species pairing、cleanup
和最终 16,384 裁剪；同时这 528 targets 没有与上述 104-target delta 表做交集联结。

这里的 `raw_depth` 和 10k cap 都包含每个 A3M 的 row 0 query。表中的读取比例已扣除 query，但仍是
merge/dedup 前的读取上界，不能直接解释为最终保留 homolog 数。

### 6.3 当前 RawMsa 的全量精确校验

为排除 depth-only 公式的 parser 偏差，又对 835 occurrences 对应的 600 个唯一抗体序列/MSA 目录做了
全量 `RawMsa.from_a3m_file(..., depth_limit=10000, dedup=False)` 校验，共检查 1,200 个 A3M 文件：

- 1,200/1,200 个文件的首条记录都是 query；
- `pairing.a3m` 和 `non_pairing.a3m` 的 query feature validation 均为 600/600 通过，fallback 为 0；
- CSV raw depth 与磁盘 record 数 1,200/1,200 一致；
- pairing-first 余额公式与当前 loader 的实际读取深度在 600/600 个唯一 source 上一致。

唯一 source 口径下，366/600（61.0%）完全不读取 `non_pairing.a3m`。其余 234 个 source 共读取
169,957 条 query 之外的 non-pairing homolog；按当前 first-wins merge/dedup 后实际保留 159,830 条，
另有 10,127 条因 source 内重复或与 pairing-source 重叠而删除。换言之：

```text
完整 non-pairing homolog       942,075
10k source 预算允许读取         169,957（18.04%）
pairing-first merge/dedup 后保留 159,830（16.97%）
```

这确认了在当前数据上 query fallback 不是混杂因素；而且“先被 raw 预算挡住”是主要损失，读入后再被
dedup/overlap 删除是第二层损失。这里仍未执行跨链 species pairing、cleanup 和最终 assembled 裁剪。

按 835 个 target-entity occurrences 加权时，精确的 homolog 流量为：

| 角色 | occurrences | raw hits | 10k 后读入 | merge/dedup 后保留 | 读入/raw | 保留/raw | 读入后再删除 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 | 835 | 1,305,730 | 240,103 | 225,281 | 18.39% | 17.25% | 14,822 |
| H | 544 | 820,071 | 185,092 | 173,090 | 22.57% | 21.11% | 12,002 |
| L | 279 | 455,305 | 53,496 | 50,709 | 11.75% | 11.14% | 2,787 |
| scFv | 12 | 30,354 | 1,515 | 1,482 | 4.99% | 4.88% | 33 |

全部 14,822 条读入后删除又可拆为：8,368 条与 pairing-source 重叠，6,454 条是 non-pairing
prefix 内部重复。provenance 模拟的最终序列和顺序已与当前
`RawMsa.merge(..., deduplicate=True, depth_limit=10000)` 在 600/600 个唯一 source 上逐条断言一致。

Target 级按所有抗体实体合计，275/528（52.1%）最终保留的 genuine non-pairing homolog 为 0；
其余 253/528 都是部分保留，没有任何 target 完整保留全部 raw non-pairing homolog。至少一个抗体
实体保留量为 0 的仍是 344/528（65.2%）。

### 6.4 与“默认推理中 final paired 较多时回退更大”是否矛盾

两者表面上方向相反，实际并不构成直接矛盾，但也不能据此宣称 10k source 竞争已经解释了回退。

需要区分两个口径：

1. `raw H/L pairing depth` 是单条抗体链的 `pairing.a3m` 文件深度；
2. 默认 inference 的 `final paired depth` 是所有 active protein chains 经过 species overlap、跨链配对、
   cleanup 和裁剪后形成的全复合物 paired row block。

二者不是同一个变量。单链有 13k pairing hits，并不意味着 H、L、抗原链之间一定有大量相同 species
可供跨链配对；反过来，单链 raw depth 较低的 target 也可能具有较高的跨链 species overlap。

当前 raw H/L pairing depth 与 checkpoint delta 基本无相关，说明“抗体链 pairing 文件越深，回退越大”
并不成立。而且大量 H/L 链都已经达到或超过 10k，source starvation 对这些 target 可能是一个近似饱和的
共同暴露，未必能解释 target 之间谁退化得更多。

一种尚待验证、但方向上可以自洽的机制是：

```text
训练阶段：
pairing-source 很深
  -> unpaired 10k 中 genuine non-pairing 被挤出
  -> 模型看到的来源多样性/Neff 下降

默认 inference：
完整 species pairing + 无 raw 10k cap + 深 MSA
  -> final paired fraction 高的 target 与训练分布错配更明显
  -> 这些 target 可能更容易暴露模型回退
```

在这个解释中，推理时 `final paired` 较高是“训练—推理组成错配的标记”，不是已经证明的直接有害因素。
同时，final paired 和 final unpaired 在 16,384 总预算下机械竞争，因此 `paired` 的负相关和 `unpaired`
的正相关本质上主要是同一条组成信号。

所以当前应采用以下保守结论：

- 10k/source-order 行为是真实存在的代码机制；
- 它可能造成训练 MSA 来源组成失真；
- 当前统计尚未证明它是 U21000 回退的原因；
- 它更不能证明“paired MSA 本身越多越有害”。

要连接这两类统计，必须对每个 target 同时测量：训练 loader 实际保留的 genuine non-pairing rows、
unique/Neff、跨链 species overlap、默认 inference final paired fraction，以及 checkpoint delta。

## 7. 默认 inference 的边界

本问题首先是当前训练 featurizer 的 raw source 预算问题，不是预测后的 evaluate/scoring 脚本问题。

默认 inference：

```yaml
msa_pair_as_unpair: true
```

它会把 paired-source hits 合并进 unpaired 候选池，同时保留原始 paired MSA 并继续进行 species pairing。
但是默认 `InferenceMSAFeaturizer` 最后调用无 10k raw cap 的 `FeatureAssemblyLine()`，因此不会复现训练时
“pairing-source 先耗尽 unpaired 10k、随后不再读取 non_pairing”的同一读取限制。

相关位置：

- `configs/configs_inference.py:35`
- `protenix/data/inference/infer_dataloader.py:170-176`
- `protenix/data/msa/msa_featurizer.py:2777-2796`

默认 inference 仍有最终 paired/unpaired 对 16,384 总预算的竞争，但没有训练中的两个 raw 10k cap。
因此潜在风险是训练—推理分布错配：训练时 genuine non-pairing 可能被饿死，推理时却输入完整、深得多且
来源组成不同的候选池。

## 8. 建议工程审计项

建议先只加统计，不立即改变训练行为。对每个 protein chain 记录：

```text
raw_pairing_depth
raw_non_pairing_depth
loaded_pairing_into_paired_pool
loaded_pairing_into_unpaired_pool
loaded_non_pairing_into_unpaired_pool
post_merge_unique_paired_depth
post_merge_unique_unpaired_depth
species_paired_selected_depth
cleanup_removed_unpaired_depth
final_paired_depth
final_unpaired_depth
paired_duplicate_fraction
unpaired_duplicate_fraction
paired/unpaired Neff
species count / species entropy
```

优先回答以下问题：

1. 在 104 个抗体-抗原 target 的 H/L/VHH 链中，有多少条链的 genuine non-pairing 保留量为 0？
2. cleanup 平均删除多少 unpaired rows？删除后有多少空余预算没有回填？
3. pairing-source 的重复率是否显著消耗了 pre-dedup 10k cap？
4. genuine non-pairing 保留量、Neff 或 species entropy 是否比最终行数更能解释 U21000 相对 FT749 的回退？
5. 此行为是否是设计意图，还是只想实现“paired 10k、genuine unpaired 10k”？

## 9. 建议最小单元测试

构造一个临时 MSA 目录，至少覆盖以下用例：

| 用例 | pairing 深度 | non-pairing 深度 | 当前预期行为 |
| --- | ---: | ---: | --- |
| A | 12,000 | 1,500 | unpaired pool 中 non-pairing 为 0 |
| B | 8,000 | 3,000 | unpaired pool 先取 pairing，再最多取 2,000 non-pairing |
| C | 0 | 3,000 | unpaired pool 取 3,000 non-pairing |
| D | pairing 前缀含大量重复 | 3,000 | 验证 pre-dedup cap 是否造成 unique depth 丢失且不回填 |

另增加一个跨链测试，验证 species pairing cleanup 删除 unpaired rows 后不会从后续 source 回填。

## 10. 可选修复方向

以下方案会改变训练数据分布，需要先由工程和建模侧确认设计意图：

1. 将 unpaired source 顺序改为 `non_pairing-pairing`，优先保留 genuine non-pairing；
2. 为 pairing-as-unpaired 和 genuine non-pairing 设置独立子预算，merge/dedup 后再裁剪；
3. 先读取两个 source，做全量或流式 dedup，再应用统一 10k unique-row cap；
4. species cleanup 后从未使用的 genuine non-pairing source 回填；
5. 保留当前行为，但明确命名和记录 source composition，避免把最终 `unpaired` 行数误解为 genuine
   `non_pairing.a3m` 深度。

如果优先目标是以最小改动验证效果，建议先实现 instrumentation，再对同一批 target 比较：

```text
当前 pairing-first
vs.
non-pairing-first
vs.
独立 source quota
```

比较时应固定 checkpoint、target、MSA 文件、seed、JSON 顺序和 inference world size。

## 11. 另外三个关联 MSA 问题

### 11.1 U21000 训练 MSA 深度与默认 inference 明显错配

U21000 resolved run config 为：

```yaml
sample_cutoff:
  train: 2048
  test: 16384
```

模型每次调用 MSA module 时随机选择：

```text
K ~ Uniform{1, ..., min(N_msa, sample_cutoff)}
```

对于当前抗体-抗原统计中的中位候选池 `N_msa=13,188`：

```text
U21000 训练：E[K] = 1,024.5
默认 inference：E[K] = 6,594.5
```

FT749 的已执行训练档案是 `sample_cutoff.train=16384`，即训练时同样近似
`K ~ Uniform{1, ..., N_msa}`。因此，U21000 默认推理时面对的平均 MSA 深度约为自身训练平均值的 6.4 倍，
而 FT749 没有同等级别的深度错配。

代码/配置位置：

- U21000 resolved config：本文件第 3 节所列 run 的 `config.yaml:1114-1116`
- 随机 K：`protenix/model/utils.py:384-402`
- FT749 已执行档案：`docs/sampling_immune_heavy_20260603_executed_run_cn.md:369-398`

这是四项中确定性最高的训练—推理分布差异，但是否导致结构质量回退仍需固定 checkpoint 的 K-depth
sweep 或共同起点训练消融验证。

### 11.2 标准 species-paired 训练暴露从 100% 降至约 60%

U21000 resolved run config 使用三个互斥 MSA mode：

```text
10%  query-only / partial-query-only
30%  pair-as-unpaired shuffle
60%  standard species-paired
```

互斥 categorical 选择见 `protenix/data/pipeline/dataset.py:790-810`。当前 run 的概率为：

```yaml
query_only_msa_probability: 0.1
pair_msa_as_unpaired_probability: 0.3
```

FT749 已执行版本使用标准 species pairing；默认 inference 也保留 paired block 并执行标准 species
pairing。因此，当前模型只有约 60% 的相关训练 occurrence 与默认 inference 的 pairing 模式直接一致。

潜在风险是模型对跨链 paired block 的利用能力下降，尤其可能在默认 inference final paired fraction 较高的
target 上暴露。但 30% shuffle 也可能抑制错误物种配对、产生正则化收益，所以影响方向不能只靠代码确定。

建议记录每个训练 occurrence 的实际 mode，并做保持其他条件不变的 `60% vs 100% species-paired`
continuation 对照。

### 11.3 当前训练 paired raw merge 不去重

当前训练代码显式使用：

```python
RawMsa.merge(
    p_msas,
    deduplicate=False,
    depth_limit=max_input_paired_msa_depth,
)
```

位置：`protenix/data/msa/msa_featurizer.py:2332-2337`。

原始 FT749 训练代码使用：

```python
RawMsa.merge(p_msas)
```

而 `RawMsa.merge()` 的默认参数是 `deduplicate=True`。原始调用位置为
`../Protenix_train/protenix/data/msa/msa_featurizer.py:494-500`。

如果 pairing source 中存在较多重复序列，当前行为可能让重复行消耗：

- pre-dedup paired 10k raw cap；
- 每 species 配额和最终 paired row block；
- 模型实际随机采样的 K 个槽位；

从而使行数很深但 unique depth/Neff 较低。另一方面，默认 inference 的 paired parser 本身也不去重，
所以这一修改在训练—推理一致性上可能有正面作用；净影响方向目前未知。

建议首先统计 `duplicate_fraction`、unique paired depth、Neff、species entropy，以及重复行最终进入 paired
block 和每个 recycle 采样的比例，再决定是否恢复 dedup。

## 12. 四项问题的当前优先级

| 优先级 | 问题 | 已确认内容 | 尚未确认内容 |
| ---: | --- | --- | --- |
| 1 | train cutoff 2048 vs inference 16384 | 配置和深度分布错配已确认 | 是否直接造成回退 |
| 2 | unpaired 10k 内 pairing-first | source 抢占机制已确认 | target 实际保留量及效果贡献 |
| 3 | 仅约 60% standard species-paired | mode 概率及默认 inference 差异已确认 | 破坏还是正则化收益 |
| 4 | paired merge `deduplicate=False` | 新旧代码差异已确认 | 实际重复率及净效果方向 |

以上四项都是 MSA 侧候选机制，不能排除 FT749 的 immune-heavy 数据配方与 U21000 广谱训练配方差异等
更强的非 MSA 混杂因素。
