# AFDB + MGnify 蒸馏库按最终代表长度整簇分区

本文记录如何把现有 AFDB/MGnify full runtime index 离线拆成两个互不重叠、
可独立训练采样的 compact runtime index：

- Short：最终簇代表的 `canonical_length <= 200`；
- Long：最终簇代表的 `canonical_length > 200`。

这里的“最终簇代表”严格指 `clusters.csv.gz` 中的 `representative_row_id` 和
`representative_id`，不是原始 MMseqs TSV 每块的第一行，也不是任意成员自己的长度。
分区决定每簇只做一次；同一原始 AFQ/MGN 簇的所有成员必须进入同一个输出。
cluster authority 始终是当前 AFQ/MGN sequence `clusters.csv.gz`；名字中含 Foldseek 的
`short_le200.paths.tsv` 只提供从 CIF→PKL 审计得到的 canonical-length `<=200` path set，
不会把 sequence cluster 替换成 Foldseek cluster。

## 1. 数据语义

分区工具保持以下合同：

- 原 full runtime index 不修改；
- 原始 `cluster_id` 原样保留；
- 每个簇内部的成员顺序不变；
- Short 和 Long 内部的 row ID、`first_row_ids` 分别重新连续编号；
- 两个输出的 row 数之和等于 full 的 row 数；
- 两个输出的 cluster 数之和等于 full 的 cluster 数；
- 两个输出不共享任何 cluster；
- 每个输出仍然支持“先均匀选择 cluster，再在 cluster 内均匀选择 member”；
- CIF 仍只保存 `source_root_id + cif_relpath`，不展开绝对路径；
- 不读取任何 CIF、A3M 或 PKL。

因此，某个 `<=200` 成员可能因为它所在簇的最终代表 `>200` 而进入 Long；反过来也
一样。这是“按代表长度整簇分区”的预期行为，不是遗漏或错误。

## 2. 输入

生产输入固定为：

```text
full runtime
/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_runtime_index_20260725

最终 cluster catalog
/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724/clusters.csv.gz

canonical_length <= 200 的已审计相对路径
/mnt/project_abd_hs/klp/databases/afdb_mgnify_foldseek_le200_tm0.8_cov0.90_20260725/manifest/short_le200.paths.tsv

对应 stats/provenance
/mnt/project_abd_hs/klp/databases/afdb_mgnify_foldseek_le200_tm0.8_cov0.90_20260725/manifest/short_le200.stats.json

权威 canonical_length 转换审计
/mnt/project_abd_hs/klp/databases/afdb_mgnify_bioassembly_plan_v3_18x100_20260725/distillation_bioassembly_indices.csv.gz

转换审计完成标记
/mnt/project_abd_hs/klp/databases/afdb_mgnify_bioassembly_plan_v3_18x100_20260725/_FINALIZE_SUCCESS.json
```

工具要求 short paths 的 stats 同时满足：

- 阈值正好是 200；
- 输入行数与 full runtime 一致；
- selected/excluded 计数闭合；
- path 行数、字节数和内容 SHA-256 一致；
- `roots.json` SHA-256 与 full runtime 一致；
- 没有未知、重复或不安全相对路径。

已有 `short_le200.paths.tsv` 是从全库 CIF→PKL 转换审计的
`canonical_length` 流式生成的。工具用它做长度集合，不会重新扫描结构文件。
生产 stats 本身只记录转换审计的路径和大小，所以构建器还会读取
`_FINALIZE_SUCCESS.json`，并重新计算 2.5 GB gzip 转换审计的 SHA-256；只有它与
完成标记中的 `merged_audit_sha256` 一致时才继续。该 SHA 和完成标记 SHA 都进入
pair ID，避免同 roots、同行数但 canonical length 内容不同的审计被静默接受。

## 3. 构建命令

生产全库构建已于 2026-07-25 完成，两个兄弟 runtime 已通过 builder 内部 artifact、
pair receipt 和 full-runtime 精确并集验证。以下保留正式构建命令用于复现：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/\
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

PY=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
FULL=/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_runtime_index_20260725
CLUSTERS=/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724/clusters.csv.gz
SHORT_MANIFEST=/mnt/project_abd_hs/klp/databases/afdb_mgnify_foldseek_le200_tm0.8_cov0.90_20260725/manifest
SHORT_OUT=/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_short_le200_runtime_index_20260725
LONG_OUT=/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_long_gt200_runtime_index_20260725

"$PY" scripts/distillation/build_length_partitioned_runtime_indices.py build \
  --full-runtime-dir "$FULL" \
  --clusters-csv "$CLUSTERS" \
  --short-paths "$SHORT_MANIFEST/short_le200.paths.tsv" \
  --short-paths-stats "$SHORT_MANIFEST/short_le200.stats.json" \
  --canonical-audit-index \
    /mnt/project_abd_hs/klp/databases/afdb_mgnify_bioassembly_plan_v3_18x100_20260725/distillation_bioassembly_indices.csv.gz \
  --short-output-dir "$SHORT_OUT" \
  --long-output-dir "$LONG_OUT" \
  --max-representative-length 200 \
  --progress-every-clusters 1000000
```

两个输出必须位于同一父目录。工具先在同一文件系统的隐藏 staging 目录中完整构建并
校验两个输出，最后才用目录 rename 发布。任一目标目录已经存在时都会直接拒绝；
没有 `--overwrite`。异常退出会清理本次 staging；进程被强制杀死时可能留下隐藏
staging，但不会被后续运行或训练当作已发布 runtime index。

两个兄弟目录无法由 POSIX 文件系统在一个 rename 中同时出现；工具采用“两个目录都
完成并验收后再依次 rename，第二次 rename 失败则回滚第一次”的成对发布。若进程恰好
在两个 rename 之间被 `SIGKILL` 或机器掉电，可能只留下一个目标目录。此时另一目录
缺失，训练 profile 和 pair verify 都必须失败；重跑也会因为已有目标而 fail closed。
不要直接使用或覆盖这个单边目录，应先核对 pair ID 后把不完整发布移出目标名称，再
重新构建。

## 4. 输出和 provenance

两个输出各自包含：

```text
manifest.json
partition_pair.json
roots.json
row_offsets.u64
source_root_ids.u8
cif_relpaths.bin
sampling/
├── cluster_ids.u32
├── first_row_ids.u64
└── cluster_sizes.u32
```

它们都是可由现有 `DistillationRuntimeIndex` 直接读取的自包含 runtime 目录。
`manifest.json.partition` 记录分区角色、判定规则、stable cluster ID 合同和 pair ID。
两个目录中的 `partition_pair.json` 必须逐字节相同，其中记录：

- full runtime manifest SHA-256；
- `clusters.csv.gz` SHA-256；
- short paths 与 stats SHA-256；
- canonical-length 转换审计及其 `_FINALIZE_SUCCESS.json` SHA-256；
- 阈值和最终代表判定规则；
- Short/Long 的 row、cluster 数；
- 按代表分区后短/长成员交叉计数和 mixed cluster 数；
- 两个分区的总覆盖计数。

pair ID 是上述输入身份和阈值的确定性哈希。修改任一输入或阈值都会得到不同 pair ID。

### 4.1 生产发布身份与计数

生产结果为：

| 分区 | cluster | row | short member | long member | mixed cluster |
|---|---:|---:|---:|---:|---:|
| Short (`<=200`) | 14,327,216 | 15,425,587 | 15,393,410 | 32,177 | 26,392 |
| Long (`>200`) | 16,163,340 | 18,073,064 | 10,877 | 18,062,187 | 7,503 |
| 合计 | 30,490,556 | 33,498,651 | 15,404,287 | 18,094,364 | 33,895 |

表中的 short/long member 是成员自己的 canonical-length 类别；partition 仍只由最终代表决定，所以
两边都允许 mixed cluster。合计恰好恢复 full runtime 的 30,490,556 个 cluster 和
33,498,651 行。

不可变生产身份：

```text
pair_id:
9c87d52dd446f622be689b7a09611f302d3791ac2448ecd86f8aee2488fb8734

partition_pair.json SHA256:
1b47dd0a7f15840b4a3f27ad67e1d821c954b263ee58c4c110c6c45a880a5476

Short manifest.json SHA256:
cab1953754d9c45302d8ab359006b21352d332628d767584471778433431ff1c

Long manifest.json SHA256:
b935c0d623d1fc2982a1449fab7ffed50b259653e11fb4c41c7bda131f535d9a
```

两个 runtime 中的 `partition_pair.json` 逐字节相同。训练和 strict resume 必须同时看到
Short role、Long role、阈值 200、相同 pair ID 和相同 receipt SHA；只出现一边或任一 identity
不匹配都必须失败。

## 5. 发布后验证

严格验证两个输出自身 SHA-256、pair receipt，以及它们相对 full runtime 的精确
cluster/path 并集：

```bash
"$PY" scripts/distillation/build_length_partitioned_runtime_indices.py verify \
  --short-runtime-dir "$SHORT_OUT" \
  --long-runtime-dir "$LONG_OUT" \
  --full-runtime-dir "$FULL" \
  --verify-recorded-sources
```

生产 build 的内部全量验证已经通过；独立 post-publish
`verify --verify-recorded-sources` 也已完成并通过，复核得到同一 exact counts、pair ID 和
receipt。以后移动或重新发布目录时仍必须重跑该命令；它会重新读取已发布目录和 recorded source，
不以 build 进程的内存状态代替发布后校验。

验证会检查：

- 两个 runtime 的所有 artifact 大小、dtype、offset、sampling interval 和 SHA-256；
- 两边的 pair ID、receipt 内容与 receipt SHA-256 一致；
- stable cluster ID 对 full runtime 恰好形成无重叠并集；
- 每个 cluster 的 size、root ordinal、CIF 相对路径和成员顺序与 full 完全一致；
- full runtime、cluster CSV、short paths 和 stats 仍与构建时记录的 stat/SHA-256 一致。
- canonical-length 转换审计和完成标记仍与构建时记录的 stat/SHA-256 一致。

不要只比较目录大小或只看构建进程退出码。配置中已经注册目录不等于数据通过验收；只有上面的
verify 正常结束后，v3 launcher 才能正式使用两个 runtime。

## 6. v3 训练接入

当前 profile：

```text
weighted_pdb_afdb_mgnify_short_long_disordered_v3
```

使用有序 dataset/outer mass：

| dataset | raw mass |
|---|---:|
| `weightedPDB_before210930_v20260101` | 0.5 |
| `afdb_mgnify_short_distillation_v20260725` | 0.005 |
| `afdb_mgnify_long_distillation_v20260725` | 0.495 |
| `openfold3_disordered_v20260725` | 0.02 |

默认 runtime 路径就是第 3 节的 `SHORT_OUT` 和 `LONG_OUT`。移动目录时，在所有训练节点同时设置：

```bash
export PROTENIX_SHORT_DISTILLATION_RUNTIME_INDEX_DIR=/new/shared/path/short_runtime
export PROTENIX_LONG_DISTILLATION_RUNTIME_INDEX_DIR=/new/shared/path/long_runtime
```

v3 对四个 dataset 统一使用 MSA query-only/unpaired-shuffle/species-paired =
`0.1 / 0.3 / 0.6`。命中 30% unpaired-shuffle 时，单个 crop-visible physical protein chain
固定请求 segment；至少两个 protein chain 时以独立 occurrence seed 50%/50% 请求 segment 或
whole-physical-chain。精确 MSA mode policy 是
`categorical_crop_aware_partial_query_only_all_protein_multichain_segment_or_whole_shuffle_species_paired_v4`。
Short/Long 每条记录只有一个 sibling A3M，运行时明确按 direct-unpaired MSA
加载，因此 unpaired raw budget 为 16,384；默认的 paired/unpaired
10,000/10,000 仍适用于 PDB 和 disordered。旧双 10,000 实现的 schema-7
training-data contract SHA-256
`b76b782fee58b6353992d2ebf8c4138173bf013f5dd7d05344abeaf739a8f55b`
不再适用于该新数据策略；新任务以 checkpoint 中记录的最终 resolved
data-contract SHA-256 为准。

16 个训练节点执行相同 scratch 命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/\
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v3_crop384_db48_disordered_v3_50k_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_disordered_v3.sh \
  > "disordered_v3.node${LOG_NODE_ID}.log" 2>&1
```

严格 resume：

```bash
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<v3_source_run> \
RESUME_STEP=999 \
RUN_NAME=protenix_v3_crop384_db48_disordered_v3_resume1000_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash resume_crop384_db48_ddp_disordered_v3.sh
```

v1/v2 checkpoint 不能 strict-resume 到 v3；两个 partition runtime、pair identity、四项 weights、
MSA policy 和 Short/Long unpaired=16,384 都属于合同。任何使用旧
Short/Long unpaired=10,000 的 checkpoint 也不能 trajectory-exact
strict-resume 到新合同。

真实 DataLoader smoke 已读取 Short 最后一行并生成 27-token crop、`N_msa=3`，读取 Long 第 0 行
并生成 384-token crop、`N_msa=64`。四源 sampler 的 200,000 次确定性抽样计数为
`97,914 / 948 / 97,136 / 4,002`，与归一化期望一致。

## 7. 资源特征

构建器不使用 pandas，也不打开 3,350 万个 CIF。主要工作是：

- 顺序读取约 6.6 亿字节 short paths，建立约 1,540 万项的内存字典；
- 顺序哈希约 2.5 GB canonical-length gzip 审计；
- 顺序解压约 3,049 万行 `clusters.csv.gz`；
- mmap 读取 full runtime；
- 顺序写出两个 runtime，其总有效载荷约等于 full runtime；
- 构建前后执行 artifact SHA-256 和完整并集校验。

生产机内存充足时，内存主要由 short-path 字典占用；预期是数 GB，不会复制全库
CSV 到 pandas。共享存储吞吐通常比 CPU 核数更关键，单进程顺序构建即可；不要把
同一输出 pair 在多台机器上并发启动。

## 8. 当前测试

定向测试：

```bash
"$PY" -m pytest -q \
  tests/test_distillation_runtime_index.py \
  tests/test_length_partitioned_runtime_indices.py
```

fixture 明确覆盖：

- 最终代表不位于簇第一行；
- Short 代表簇包含 Long 成员；
- Long 代表簇包含 Short 成员；
- 长度边界 200 进入 Short；
- stable cluster ID 与簇内成员顺序保持；
- 两个 compact row 空间分别连续；
- 任一目标已存在时 fail closed；
- short path 不属于 full runtime 时拒绝发布；
- `representative_id` 与 `representative_row_id` 不一致时拒绝；
- 输出 artifact 被篡改后 verify 失败。
