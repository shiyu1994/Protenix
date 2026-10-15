# AlphaFold 3 测试集定义与当前 checkpoint 评测方案

## 结论

AlphaFold 3 没有公开 Fig. 2d 训练曲线所用的完整 1,220-target
model-selection validation manifest，因此无法在“完全同一批 ID”上复现该图。
当前仓库使用公开的 PXM-Legacy RecentPDB 低同源集合及其 RNA/DNA 子集，构造一份
可复现的 7-category AF3-like proxy：

- smoke：12 个 target，先验证完整链路；
- full：1,093 个本地可读取且不超过 1,024 token 的 target；
- 每个 target 固定 5 个 diffusion sample；
- `quick` 使用 20 diffusion steps，只用于排错；
- `formal` 使用 200 diffusion steps，才用于 checkpoint 间正式比较。

这个 proxy 可以比较我们不同 checkpoint 的能力，但不能冒充 AF3 未公开的
1,220-target 集，也暂时不能作为严格的未见数据泛化结论。

评测保留当前 Protenix 的标准 test-stage paired MSA 行为，但底层 A3M 通过逐文件
hash-lock 固定；真实 template 不再在线构建，而是从独立的
`2021-09-30` cutoff、固定 top4 evaluation cache 加载。完整输入冻结合同见
[PXM-Legacy 评测的冻结 MSA 与 Template 输入](af3_pxm_locked_msa_template_inputs_cn.md)。
H100 对训练中全部 EMA checkpoint 按每 600 updates 持续评测的交接说明见
[H100 Worker：每 600 Updates 评测全部 EMA Checkpoint](af3_pxm_every600_checkpoint_inference_worker_cn.md)。
MSA 原始输入最多 16,384 行，
模块以固定 seed 从
`1...min(actual_depth, 16384)` 随机确定深度并无放回取行；所以同一 manifest、
seed 和单卡顺序可复现，但不等于每个 target 固定用满 16,384 行。

## AF3 实际使用了哪些集合

### Fig. 2d 训练曲线

AF3 Supplementary Methods 5.7/5.8 描述的 model-selection validation set：

- PDB release 在 `2021-09-30` 之后、`2023-01-13` 之前；
- 最终 target 不超过 2,048 tokens；
- 从 chain/interface cluster 中抽样并做人工排除；
- 1,220 个 targets；
- 2,333 个低同源 interfaces；
- 2,099 个低同源 chains。

官方没有公开：

- 1,220 个 PDB ID；
- cluster 抽样 seed；
- 被人工排除的 target ID；
- 精确 chain/interface scoring manifest；
- Fig. 2d 每个点的原始数值。

所以只能重建规则相近的集合，不能声称 exact reproduction。

### 论文最终 benchmark

论文还报告了几类独立 benchmark：

- Recent PDB evaluation set；
- PoseBusters v1；
- CASP15 RNA；
- 抗体、蛋白–核酸、蛋白–配体等 RecentPDB 子集。

它们与 Fig. 2d 的 1,220-target model-selection validation set 不是同一个集合。
论文默认最终推理通常使用 5 seeds × 每 seed 5 diffusion samples，共 25 个候选；
Fig. 2d 的训练期曲线则是“全局 gPDE top-1 与 best-of-5 的平均”。

官方来源：

- [AlphaFold 3 Nature 论文](https://www.nature.com/articles/s41586-024-07487-w)
- [AlphaFold 3 官方代码](https://github.com/google-deepmind/alphafold3)
- [PXMeter 官方仓库](https://github.com/bytedance/PXMeter)
- [PXMeter 可用数据集说明](https://github.com/bytedance/PXMeter/blob/main/docs/available_datasets.md)

## 当前本地 proxy 如何得到

原始 PXM-Legacy：

- RecentPDB：1,569 个 PDB；
- RNA-Protein：25 个 PDB；
- dsDNA-Protein：38 个 PDB；
- 三者并集：1,620 个 PDB。

清单生成脚本：

```text
scripts/evaluation/prepare_af3_pxm_legacy_subset.py
```

它执行以下确定性过滤：

1. 与本地
   `recentPDB_low_homology_maxtoken1536.csv` 取 PDB-ID 交集；
2. 要求本地同时存在原始 mmCIF 和 recentPDB assembly pkl；
3. 要求 `num_tokens <= 1024`；
4. 审计与 `weightedPDB_before210930` 的直接 PDB-ID 重叠；
5. 排除当前 runtime 无法正确汇总评分类型的 `7to1`；
6. 按 token 数和 PDB ID 稳定排序；
7. 生成 full 与每类至少两个 target 的 smoke manifest。

当前结果：

| 项目 | 数量 |
| --- | ---: |
| PXM 三集合并集 | 1,620 |
| 存在于本地 RecentPDB index | 1,317 |
| 本地 index 缺失 | 303 |
| 本地结构文件不完整 | 1 |
| 完整但超过 1,024 token | 222 |
| 当前 runtime 不支持的评分类型 | 1（`7to1`） |
| 最终 full manifest | 1,093 |
| 与 weightedPDB 的直接 PDB-ID 重叠 | 0 |
| smoke manifest | 12 |

manifest 位于：

```text
/mnt/project_abd/klp/proteinx_eval_data/manifests/pxm_legacy_af3_like/
```

完整统计：

```text
/mnt/project_abd/klp/proteinx_eval_data/manifests/pxm_legacy_af3_like/summary_le1024.json
```

当前本地 index 可评估 7 类：

- `intra_prot`
- `prot_prot`
- `ligand_prot`
- `intra_dna`
- `dna_prot`
- `intra_rna`
- `rna_prot`

没有可用的 `intra_ligand` chain 行，因此该类保持缺失，不能用 0 伪造第八条曲线。
这里的 target ID 来自 PXM 清单，但 7 类 chain/interface scoring rows 取自本地
Protenix RecentPDB index。它不是 PXMeter 对五套公开 benchmark 的正式联合聚合：
严格 PXM 评测应把 RecentPDB、RNA-Protein、dsDNA-Protein、AF3-AB 和
PoseBusters 分开运行；PoseBusters 使用 ligand RMSD/validity，不能冒充
`intra_ligand LDDT`。

## 指标实现

实现位置：

```text
protenix/metrics/lddt_metrics.py
```

每个 target：

1. 用全复合物 gPDE 最低的 sample 作为同一个全局 top-1；
2. 每条 chain/interface 独立取 5 个 sample 中的 best；
3. 分数为 `(top1 + best5) / 2`；
4. 同一 target 内相同类型的多条 entry 先求均值；
5. 再跨 target 求平均，日志使用 0–100 分制。

蛋白/配体 LDDT 邻域半径为 15 Å；核酸作为第一个原子方向时为 30 Å。接口同时保留
两个方向，因此 protein–DNA/RNA 不受 index 中 chain 顺序影响。空接口直接跳过，
不写入 0 或 NaN。

指标 key 示例：

```text
ema0.999_lddt/af3_proxy/intra_prot/top1_best5_pct
ema0.999_lddt/af3_proxy/prot_prot/top1_best5_pct
ema0.999_lddt/af3_proxy/rna_prot/top1_best5_pct
```

注意：当前是 target-balanced AF3-like proxy。PXMeter 正式 benchmark 的
cluster-weighted 聚合、严格结构映射和同实体链最大-LDDT permutation 仍应通过
独立 PXMeter CIF evaluator 完成。

## 启动命令

必须使用一张不属于 128-GPU 训练任务的独立 H100。launcher 会检查
`CUDA_VISIBLE_DEVICES` 只暴露一张卡，并在非 dry-run 时拒绝已经占用超过
2,048 MiB 的 GPU。

当前已验证可读取的 checkpoint：

```text
/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/checkpoints/7399_ema_0.999.pt
```

### 先做 dry-run

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

CUDA_VISIBLE_DEVICES=0 \
DRY_RUN=1 \
EMA_CHECKPOINT=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/checkpoints/7399_ema_0.999.pt \
EVAL_SCOPE=smoke \
EVAL_MODE=quick \
bash eval_af3_pxm_legacy_ema_checkpoint.sh
```

### 12-target quick smoke

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

CUDA_VISIBLE_DEVICES=0 \
EMA_CHECKPOINT=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/checkpoints/7399_ema_0.999.pt \
EVAL_SCOPE=smoke \
EVAL_MODE=quick \
bash eval_af3_pxm_legacy_ema_checkpoint.sh \
  > af3_pxm_smoke_u7400.log 2>&1
```

### 12-target 200-step formal smoke

只把上一条命令改为：

```text
EVAL_MODE=formal
```

### 1,093-target full formal

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

CUDA_VISIBLE_DEVICES=0 \
EMA_CHECKPOINT=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/checkpoints/7399_ema_0.999.pt \
EVAL_SCOPE=full \
EVAL_SUBSET=combined \
EVAL_MODE=formal \
bash eval_af3_pxm_legacy_ema_checkpoint.sh \
  > af3_pxm_full_formal_u7400.log 2>&1
```

不要直接跳到 full formal。应先完成：

1. quick smoke；
2. formal smoke；
3. 核对每类 metric 都是 finite；
4. 再提交 full formal。

## 输出与成功标准

输出根目录：

```text
/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_legacy_proxy/
```

实际 run 目录会自动追加时间戳。重点文件：

```text
logs/eval_metrics.jsonl
logs/eval_samples_rank0.jsonl
logs/eval_worst_samples.jsonl
manifest/run_status.json
config.yaml
```

成功要求：

- checkpoint preflight 显示 PF64/cz256/cm128/diffusion-cs384 全部匹配；
- `sample_count` 等于 manifest target 数；
- `run_status.json` 为成功完成；
- 七类 `top1_pct / best5_pct / top1_best5_pct` 都是 finite；
- 没有随机替换、CUDA OOM、template/MSA parser error 或缺失 target；
- `formal` 日志明确显示 `N_sample/N_step=5/200`。

## 泛化性限制

当前 full manifest 与 weightedPDB cutoff 集合的直接 PDB-ID 重叠为 0，但这不足以
证明完全未见：

- 尚未对 AFDB/MGnify 3,000 万蒸馏序列做 exact-sequence 去重；
- 尚未对蒸馏库做序列同源性过滤；
- template 搜索可能暴露相关结构。

因此现阶段结果应命名为“公开 RecentPDB 能力/回归评测”。若要用于严格泛化结论，
下一步必须基于训练使用记录和全部蒸馏库建立 exact-sequence + homology exclusion
manifest。
