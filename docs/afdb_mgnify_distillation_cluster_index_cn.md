# AFDB + MGnify 蒸馏库聚类索引、运行时索引与训练接入

## 1. 目的与边界

本文固定 AFDB 与 MGnify 预测结构蒸馏库的以下数据契约：

- 哪些目录和 manifest 是结构、A3M 与模型 ID 的权威来源；
- 如何解释并修复聚类 TSV；
- 如何生成训练兼容、路径可迁移且可审计的全库 CSV；
- 如何从全库 CSV 一次生成 mmap 运行时索引；
- 如何在训练期做可恢复的临时分层随机采样；
- 全库索引和训练期加载必须满足哪些正确性与性能约束。

当前数据已经完成 CIF canonical sequence 与 A3M query sequence 的库外校验，也已经预先完成全库 PKL
转换。本流程不再逐样本重复做序列一致性检查；训练期只读取 catalog 的 compact 运行时视图、现成
PKL 和同 stem A3M。

> **2026-07-25 实现状态：** 33,498,651 个 CIF 已全部转换为同目录
> `bioassembly/<stem>.pkl.gz`，失败数为 0。训练方案已从“预生成冻结 schedule”
> 调整为“compact mmap + 训练期确定性临时随机”；不再生成 10 万步样本表。
> 下文第 7–13 节记录当前生效实现，早期冻结 schedule 只保留为设计背景。

## 2. 已核实的输入快照

聚类输入为：

```text
/mnt/project_abd_hs/klp/databases/result_cluster.AFQ_MGN_only.tsv
```

2026-07-24 对该文件进行了不加载全文件到内存的流式全量审计：

| 项目 | 核实值 |
|---|---:|
| 文件大小 | 1,794,961,159 bytes |
| 数据行数 | 33,498,651 |
| 列数 | 2 |
| header | 无 |
| 空字段或列数异常 | 0 |
| 连续聚类块数 | 30,490,556 |
| AFQ 代表簇数 | 20,572,755 |
| MGN 代表簇数 | 9,917,801 |
| 单成员簇 | 28,467,298 |
| 多成员簇 | 2,023,258 |
| 最大簇大小 | 2,696 |

两列语义是 MMseqs 风格的：

```text
original_representative_id<TAB>member_id
```

相同的第一列在文件中构成连续块，但不同聚类块不是全局字典序。每个连续块恰好包含一条 `original_representative_id == member_id` 的 self 行，没有发现缺 self 或重复 self 的聚类块。不能假定 self 行总是该块第一行：

- 30,408,631 个簇的 self 位于第一行，其中包含全部单成员簇；
- 79,454 个簇的 self 位于最后一行但不在第一行；
- 2,471 个簇的 self 位于中间。

因此，流式解析只能按第一列变化切分聚类块，不能把每块第一条成员误当成原代表。

### 2.1 成员与簇来源分布

| 成员来源 | 行数 |
|---|---:|
| AFQ | 23,176,298 |
| MGN | 10,322,353 |
| 合计 | 33,498,651 |

| 聚类组成 | 簇数 |
|---|---:|
| 纯 AFQ | 20,197,932 |
| 纯 MGN | 9,460,919 |
| AFQ/MGN 混合 | 831,705 |

混合簇中，374,823 个以 AFQ 为原代表，456,882 个以 MGN 为原代表。成员边的全量分布为：

| 原代表来源 → 成员来源 | 行数 |
|---|---:|
| AFQ → AFQ | 22,657,500 |
| AFQ → MGN | 379,981 |
| MGN → AFQ | 518,798 |
| MGN → MGN | 9,942,372 |

混合簇必须作为一个聚类整体处理，不能先按 AFDB/MGnify 拆成两个互不相关的索引。

### 2.2 簇大小分布

| 簇大小 | 簇数 |
|---|---:|
| 1 | 28,467,298 |
| 2 | 1,583,714 |
| 3–5 | 379,179 |
| 6–10 | 48,193 |
| 11–50 | 11,881 |
| 51–100 | 220 |
| 101–1000 | 69 |
| >1000 | 2 |

单成员簇占 93.3643%，平均每簇 1.098657 个成员。最大簇的原代表为 `AFQ|AF-A0A1B1R7N2-F1-model_v4`。

## 3. 权威数据根与模型清单

全库 CSV 不重复保存绝对路径。每行通过 `source_root_id + cif_relpath` 定位 CIF；根目录映射由 provenance manifest 固定。

| `source_root_id` | `source_db` | 权威根目录 | 当前模型数 |
|---:|---|---|---:|
| 0 | `afdb` | `/mnt/project_abd_hs/klp/databases/afdb/msas` | 22,803,271 |
| 1 | `afdb` | `/mnt/project_abd_hs/klp/databases/afdb/models_unique_20260724` | 373,027 |
| 2 | `mgnify` | `/mnt/project_abd_hs/biran/data/mgnify/msas` | 10,322,353 |

三处模型数之和正好是 33,498,651，与聚类 TSV 的成员行数完全闭合：

```text
22,803,271 + 373,027 + 10,322,353 = 33,498,651
```

其中 373,027 个 `models_unique_20260724` 模型不是应删除的“聚类孤儿”，而是 AFDB 合并后单独保存的有效模型。它们恰好解释了只扫描旧 `afdb/msas` 时看到的 AFQ 数量差：

```text
23,176,298 - 22,803,271 = 373,027
```

AFDB 的历史合并与校验清单来自：

```text
/mnt/project_abd_hs/klp/databases/afdb/afdb_model_rename_20260723/afdb_model_rename.mapping.tsv
/mnt/project_abd_hs/klp/databases/afdb/afdb_model_merge_20260724/model_manifest.tsv.gz
```

第二份 merge manifest 中的 final CIF 路径覆盖 rename mapping 中相同 model ID 的旧路径。实际 catalog 构建统一枚举上表三个最终根目录中的 CIF；这样 AFDB 和 MGnify 使用同一套精确文件名规则，也不需要先把 14 GB 级 AFDB mapping 再加载一遍。MGnify 的 MGYP ID 不能推导叶目录，因此更必须保留枚举得到的相对路径。

对三个根目录，A3M 均由 CIF 同目录、同 stem 替换扩展名得到，因此全库 CSV 不另存 `a3m_relpath`：

```text
path/to/model.cif  ->  path/to/model.a3m
```

本次 catalog 构建不逐文件 `stat` A3M，也不再读取或比较其中的序列；A3M 路径只按上述规则推导。CIF/A3M 配对和序列一致性采用已经完成的库外校验结果。训练期若文件在此后被移动或删除，应由读取/缓存层快速报错，而不是静默换样本。

## 4. ID 规则

聚类 ID 必须原样保存，不能丢失 namespace、大小写、isoform 或模型版本。

### 4.1 AFDB

```text
AFQ|AF-<accession>-F<fragment>-model_v<version>
```

已核实的主要版本为 `model_v4` 和 `model_v6`。`<accession>` 本身可能包含连字符，例如：

```text
AFQ|AF-P58753-3-F1-model_v6
```

因此不能用“按 `-` 固定切成若干段”的方式解析 AFDB ID，也不能把 `_v6` 归一成 `_v4`。对应文件 stem 是去掉 `AFQ|` 后的完整字符串。

### 4.2 MGnify

```text
MGN|MGYP<digits>
```

对应文件 stem 是去掉 `MGN|` 后的完整字符串。MGYP ID 不能用数值区间可靠推导其 `pNNN` 或 subshard 目录，必须使用已经枚举出的相对路径。

### 4.3 内部唯一键

`source_protein_id` 使用聚类 TSV 中包含 namespace 的完整 ID。它是 catalog join、重复检测和
provenance 审计的权威蛋白键：

- 不删除 `AFQ|` 或 `MGN|`；
- 不改变大小写；
- 不改变 `model_v4`/`model_v6`；
- 不将 accession 或 MGYP 数字单独用作全局键。

## 5. 全库 CSV 契约

全库 CSV 的前 19 列必须保持当前官方 `distill-sabdab.csv` 与数据集代码使用的顺序。旧版预处理文档中包含 `assembly_id` 的表格不是本索引的列顺序依据。

### 5.1 标准训练列

| 顺序 | 列名 | 蒸馏单体行约束 |
|---:|---|---|
| 1 | `entity_1_id` | `1` |
| 2 | `chain_1_id` | `A` |
| 3 | `mol_1_type` | `prot` |
| 4 | `cluster_1_id` | 原始连续聚类块的 0-based 整数序号 |
| 5 | `entity_2_id` | 单体行为空 |
| 6 | `chain_2_id` | 单体行为空 |
| 7 | `mol_2_type` | 单体行为空 |
| 8 | `cluster_2_id` | 单体行为空 |
| 9 | `cluster_id` | 与 `cluster_1_id` 相同的紧凑整数 |
| 10 | `pdb_id` | CIF 的原始 stem，不改变大小写或模型版本 |
| 11 | `release_date` | `9999-12-31` |
| 12 | `num_tokens` | 全库 catalog 固定写 `-1`；实际长度由训练读取 PKL 后得到 |
| 13 | `num_prot_chains` | `1` |
| 14 | `resolution` | `-1.0` |
| 15 | `type` | `chain` |
| 16 | `mol_type_group` | `intra_prot` |
| 17 | `sub_mol_1_type` | `prot` |
| 18 | `sub_mol_2_type` | 单体行为空 |
| 19 | `eval_type` | `intra_prot` |

整数 `cluster_id` 不随代表替换而变化：它等于原 TSV 中连续代表块的出现序号；如果一个簇被全部删除，该序号留下空洞。完整的原代表与修复后代表只在 `clusters.csv.gz` 和清洗后的 MMseqs TSV 中保存，避免在 33,498,651 行训练索引中重复长字符串。`pdb_id` 必须唯一、稳定，并完整保留 AFDB 的 v4/v6 后缀。

### 5.2 直接附加的 provenance 列

标准 19 列之后按以下顺序追加 6 列：

| 顺序 | 列名 | 定义 |
|---:|---|---|
| 20 | `row_id` | 最终 catalog 中稳定、唯一的整数行号 |
| 21 | `source_db` | `afdb` 或 `mgnify` |
| 22 | `source_root_id` | 引用第 3 节三行根目录表 |
| 23 | `cif_relpath` | 相对于 `source_root_id` 的 POSIX CIF 路径 |
| 24 | `source_protein_id` | 聚类 TSV 中带 `AFQ|`/`MGN|` 的完整 ID |
| 25 | `cluster_size` | 缺失过滤和代表修复后，该簇的存活成员数 |

完整列顺序因此是：

```text
entity_1_id,chain_1_id,mol_1_type,cluster_1_id,
entity_2_id,chain_2_id,mol_2_type,cluster_2_id,
cluster_id,pdb_id,release_date,num_tokens,num_prot_chains,
resolution,type,mol_type_group,sub_mol_1_type,sub_mol_2_type,
eval_type,row_id,source_db,source_root_id,cif_relpath,
source_protein_id,cluster_size
```

路径字段必须满足：

- 使用 `/` 分隔；
- 不是绝对路径；
- 不包含 `..` 路径逃逸；
- 以 `.cif` 结尾；
- 与根目录拼接后唯一定位一个 CIF；
- A3M 可由同 stem 的 `.a3m` 确定。

根目录绝对路径、输入 manifest 路径、文件大小、mtime、内容摘要、生成工具版本和 catalog 摘要应写入单独的 provenance manifest，不能在 33,498,651 行中重复保存。

## 6. 缺失过滤与代表替换

尽管当前三个权威根的数量与聚类成员数已经完全闭合，构建工具仍必须保留缺失容错。存在性以构建出的路径 catalog 为准，不应在解析聚类 TSV 时对每一行重新进行随机 `stat`。

对每个连续的原代表聚类块依次执行：

1. 用完整 `source_protein_id` 与路径 catalog join。
2. 删除 CIF inventory 中不存在的成员，并写入缺失审计输出。这里不重复检查已在库外验证过的 A3M。
3. 如果没有成员存活，删除整个簇并写入空簇审计输出。
4. 如果原代表仍然存活，继续使用原代表。
5. 如果原代表不存在，从原 TSV 顺序中的第一个存活成员确定性地选择新代表。
6. 该簇所有存活行的 `cluster_id` 和 `cluster_1_id` 均写原始聚类块的紧凑整数序号；最终代表写入簇级审计和清洗 TSV。
7. 按存活成员数重新计算并填写 `cluster_size`。
8. 保持原聚类块顺序和块内存活成员顺序，再分配稳定 `row_id`。

代表替换必须在 `clusters.csv.gz` 中显式标记；簇级记录至少包含：

- 原代表；
- 新代表；
- 原簇大小；
- 存活簇大小；
- 被删除成员数；
- 替换原因。

如果同一个 `source_protein_id` 出现在多个簇或同一簇中重复出现，构建必须 fail closed 并报告，不能静默选择其中一条。合法的 `representative == member` self 行不属于错误重复。

## 7. 训练运行时索引

33,498,651 行、25 列的全库 CSV 是构建期和审计期的 catalog，不是每个训练 rank 都应完整常驻内存的运行时对象。

如果 16 个节点、每节点 8 卡的 128 个 rank 都用 pandas 读取全表，会造成：

- 全库索引在每个 rank 中重复物化；
- 节点内 8 份重复内存，集群内 128 份重复内存；
- 所有 rank 同时读取共享文件，形成启动期元数据与带宽风暴；
- pandas 字符串对象的实际内存远高于 CSV 文件字节数；
- rank 启动时间和内存占用随全库规模增长。

当前实现由以下一次性工具流式扫描 CSV：

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/distillation/build_runtime_distillation_index.py build \
  --input-index \
  /mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724/distillation_index.csv.gz \
  --output-dir \
  /mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_runtime_index_20260725
```

只在 catalog 内容变化时重建。普通训练启动和断点续训都不能运行这条构建命令。

发布目录包含：

- `row_offsets.u64`：每一行相对路径在字符串块中的起止偏移；
- `source_root_ids.u8`：每一行使用哪个数据根；
- `cif_relpaths.bin`：连续存储的 CIF 相对路径；
- `sampling/cluster_ids.u32`、`first_row_ids.u64`、`cluster_sizes.u32`；
- `roots.json` 和带完整大小、SHA-256、schema、源表身份的 `manifest.json`。

`DistillationRuntimeIndex` 构造时只读取两个小 JSON；数组在第一次访问时只读 mmap。DataLoader
通过 fork 或 spawn 建立 worker 后会按进程重新打开映射。mmap 不会把 2.1 GB 内容复制成 Python
对象；同一节点内的 rank/worker 还可复用操作系统页缓存。`BaseSingleDataset` 检测到
`runtime_index_dir` 后完全跳过 `pandas.read_csv(indices_fpath)`。

## 8. 训练期临时分层随机采样

当前不生成冻结 10 万步 schedule。每个 sampler epoch 按 seed 和 epoch 临时生成 10,240 个
global target 位置，随后由各 rank 按 stride 取自己的位置。16×8=128 rank 时，每个 rank 每个
epoch 恰好得到 80 个 micro-step，即 20 个 gradient-accumulation update。

混合训练使用两层数据源采样：

1. 按 outer mass 选择现有 weighted PDB 或 AFDB+MGnify 蒸馏库；默认二者各 `0.5`。
2. PDB 子集沿用原 weighted sampler。
3. 蒸馏子集先在 30,490,556 个簇中均匀选择一个簇，再在该簇成员中均匀选择一行。

该规则让每个簇获得相同的第一阶段概率，再让同一簇内成员等概率；它不等价于直接对
33,498,651 行做 member-uniform。采样有放回，不要求一个 epoch 覆盖全库。

sampler state 保存 `sampler_epoch + next_local_offset + consumed_global_batches`。相同 seed、epoch、
world size、权重、runtime manifest 和恢复 cursor 会重现相同 target 流；strict resume 合同同时绑定
runtime manifest SHA-256、源 catalog SHA-256、行数、簇数与 sampler fingerprint。修改数据权重、
runtime index 或拓扑后不能 strict-resume 旧 checkpoint。

## 9. 构建工具与输出

工具入口：

```text
scripts/distillation/build_clustered_distillation_catalog.py
```

当前快照的全量命令为：

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/distillation/build_clustered_distillation_catalog.py \
  --cluster-tsv /mnt/project_abd_hs/klp/databases/result_cluster.AFQ_MGN_only.tsv \
  --source-root 0,AFQ,/mnt/project_abd_hs/klp/databases/afdb/msas \
  --source-root 1,AFQ,/mnt/project_abd_hs/klp/databases/afdb/models_unique_20260724 \
  --source-root 2,MGN,/mnt/project_abd_hs/biran/data/mgnify/msas \
  --output-dir /mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724 \
  --scan-workers 128 \
  --compresslevel 1 \
  --progress-every-clusters 1000000 \
  --expect-count inventory_entries=33498651 \
  --expect-count input_rows=33498651 \
  --expect-count input_clusters=30490556 \
  --expect-count surviving_rows=33498651 \
  --expect-count surviving_clusters=30490556 \
  --expect-count removed_members=0 \
  --expect-count dropped_clusters=0 \
  --expect-count replaced_representatives=0 \
  --expect-count source_rows.afdb=23176298 \
  --expect-count source_rows.mgnify=10322353
```

`--expect-count` 是发布门槛而不是统计提示：任一值不符，staging 会被删除且目标目录不会发布。输出目录采用 staging 后原子提交；非空目标目录会被拒绝，避免覆盖已有结果。工具产生：

1. **`distillation_index.csv.gz`：全库 catalog**
   - 标准 19 列加 6 个 provenance 列；
   - 路径均为 `source_root_id + cif_relpath`；
   - 一行对应一个存活蛋白。
2. **`roots.json` 与 `manifest.json`：根目录与输入 provenance**
   - 三个 root ID；
   - 聚类 TSV 和 AFDB 两份权威 manifest 的身份信息；
   - MGnify 枚举快照；
   - 构建工具、参数、时间和摘要。
3. **`manifest.json` 中的构建摘要**
   - 输入/输出成员、簇和各 source 数量；
   - 缺失、替换代表、删除空簇和重复错误数量；
   - 簇大小分布。
4. **`removed_members.tsv.gz`：缺失成员 audit**
   - 所有被过滤的 source protein ID 及原因。
5. **`clusters.csv.gz`：存活簇及代表修复 audit**
   - 所有原代表、最终代表、首行位置、代表行位置、原始/存活大小和修复 flags。
6. **`dropped_clusters.tsv.gz`：空簇 audit**
   - 过滤后没有任何成员的原簇。
7. **`cleaned_clusters.tsv.gz`：清洗后的 MMseqs 两列 TSV**
   - 第一列统一为最终存活代表，第二列为存活成员。
8. **`sampling/`：训练期分层随机的紧凑输入**
   - 对齐的 `cluster_ids.u32`、`first_row_ids.u64`、`cluster_sizes.u32`；
   - 每个数组均在 manifest 中记录 dtype、端序、元素数和 SHA-256；
   - 这三个数组可直接完成先 cluster-uniform、再 member-uniform 的行号选择。
9. **`afdb_mgnify_distillation_runtime_index_20260725/`：训练直接读取的 mmap 目录**
   - 由本节 catalog 再流式转换一次；
   - 不复制 CIF、A3M 或 PKL；
   - 每次训练只读该目录，不读全库 CSV。

全库 PKL 已经预生成在每个 CIF 所在目录的 `bioassembly/` 子目录。训练读取一条 runtime row 后按
规则直接推导 CIF、A3M 和 PKL 路径，不扫描目录、不查 3350 万项 JSON 映射，也不在训练期转换
CIF。

## 10. 发布前验证不变量

### 10.1 输入闭合

- 聚类 TSV 恰好两列且无 header。
- 每个连续代表块恰好有一条合法 self 行。
- 当前快照成员总数为 33,498,651。
- 当前快照 AFQ 为 23,176,298，MGN 为 10,322,353。
- 三个权威根 catalog 的唯一模型数之和与聚类成员数一致。
- 373,027 个特殊 AF 模型必须映射到 root 1，不能被当作缺失过滤。

### 10.2 catalog 正确性

- CSV 列顺序严格等于第 5 节的 25 列。
- `row_id` 唯一、稳定并覆盖约定的连续整数区间。
- `source_protein_id` 在输出中唯一。
- `source_db` 与 namespace、`source_root_id` 一致。
- `cif_relpath` 是安全相对路径，且构建时由对应根目录中的实际 CIF 枚举得到。
- AFDB v4/v6 完整保留，文件 stem 与去 namespace 后的 ID 一致。
- 输出成员数加缺失成员数等于输入成员数。
- 每个 `cluster_id` 对应的代表必须是同簇的一条存活输出行。
- 每行 `cluster_size` 等于其 `cluster_id` 的实际输出行数。
- 只有原代表缺失的簇可以带有 `representative_replaced` 标记。
- 只有零存活成员的簇可以出现在空簇 audit 中。
- 不重新做已经完成的 CIF/A3M 序列一致性比较。

### 10.3 runtime index 与训练采样正确性

- runtime manifest 引用的 catalog 大小、mtime 与 SHA-256 与实际源表完全一致。
- `row_offsets` 严格递增，末偏移等于 `cif_relpaths.bin` 大小。
- 每个 runtime `row_id` 都能解析到唯一的 `source_root_id + cif_relpath`。
- 三个 cluster 数组长度一致，区间首尾连续且恰好覆盖全部 row。
- 抽样统计符合 outer mass、cluster-uniform、member-uniform 的定义。
- 多 rank stride 不遗漏或重复 global position。
- 保存并恢复 sampler cursor 后，剩余 target 流逐项一致。
- PDB-only profile 仍走原 `WeightedMultiDataset`，不受 hierarchical 路径影响。
- 任一训练 rank 都不需要完整加载全库 CSV。

## 11. 当前快照的全量构建结果

2026-07-24 已完成全量构建并原子发布到：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724
```

实际结果为：

- 33,498,651 个成员全部找到 CIF；同 stem A3M 沿用此前已完成的库外配对校验；
- 30,490,556 个簇全部保留；
- 缺失成员为 0；
- 代表替换为 0；
- 删除空簇为 0。

构建耗时 2,810.04 秒，约 46 分 50 秒；发布目录约 2.3 GB。聚类 TSV 的 SHA-256 为：

```text
d1f9282e6ebd3730f6a81faace65982bc4f0a192d72fdfb39f4bc6221a4cadde
```

发布后独立验证结果：

- 所有 5 个 gzip 文件均通过完整性检查；
- `distillation_index.csv.gz` 解压后为 33,498,652 行，等于 header 加 33,498,651 个成员；
- `clusters.csv.gz` 解压后为 30,490,557 行，等于 header 加 30,490,556 个簇；
- `removed_members.tsv.gz` 与 `dropped_clusters.tsv.gz` 均只有 header；
- `cleaned_clusters.tsv.gz` 解压内容的 SHA-256 与原聚类 TSV 完全相同；
- 三个 sampling 数组的文件大小和 SHA-256 与 manifest 一致；
- `cluster_ids` 严格覆盖 `[0, 30,490,556)`；
- 所有 `[first_row_id, first_row_id + cluster_size)` 区间首尾连续且恰好覆盖 `[0, 33,498,651)`；
- CSV schema、前三条 `source_root_id + cif_relpath` 路径解析和实际 CIF 存在性抽样均通过。

这些已验证值仍然不是绕过容错逻辑的理由。如果输入 manifest、目录内容或聚类 TSV 任一发生变化，
必须形成新的 provenance 版本并重新生成 catalog 与 runtime index；旧 checkpoint 的严格数据合同
不会自动接受新版本。

## 12. PKL 与 runtime index 的最终发布结果

全库 CIF→PKL 已完成：

| 项目 | 结果 |
| --- | ---: |
| CIF 总数 | 33,498,651 |
| 成功 PKL | 33,498,651 |
| 失败 | 0 |
| 输出位置 | 每个 CIF 同目录下的 `bioassembly/<stem>.pkl.gz` |

2026-07-25 runtime index 已原子发布到：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_runtime_index_20260725
```

构建耗时 313.89 秒，目录约 2.1 GB。源 `distillation_index.csv.gz` 大小为
1,006,442,484 bytes，SHA-256 为：

```text
951d22720abcf63f1460163f56d8e052bb0f03c35475f583d56bce007ed2a479
```

关键文件大小：

| 文件 | bytes |
| --- | ---: |
| `row_offsets.u64` | 267,989,216 |
| `source_root_ids.u8` | 33,498,651 |
| `cif_relpaths.bin` | 1,380,949,122 |
| `sampling/cluster_ids.u32` | 121,962,224 |
| `sampling/first_row_ids.u64` | 243,924,448 |
| `sampling/cluster_sizes.u32` | 121,962,224 |

以下完整校验已通过：

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/distillation/build_runtime_distillation_index.py verify \
  --runtime-dir \
  /mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_runtime_index_20260725 \
  --verify-source
```

另外随机抽查 300 行，CIF、A3M、PKL 缺失数均为 0。三个数据根各跑过一条真实
`BaseSingleDataset.process_one()`：

- AFDB 主库：crop 384、MSA 深度 1,619；
- MGnify：crop 384、MSA 深度 9,151；
- AFDB unique：crop 384，query-only 路径 MSA 深度 1；
- 三条均设置 `is_distillation=true` 并成功生成训练 feature/label。

另用 `persistent_workers=true` 的双 worker DataLoader 随机采样 4 条，覆盖 AFDB、MGnify、
query-only 和直接 A3M 分支；mmap 在 fork 后重开、分层 sampler、PKL/A3M 读取与 feature
生成全部通过。

## 13. 训练与续训

当前正式代码库：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler
```

在 16 个训练节点上执行相同命令；平台提供的 `MLP_WORKER_*` 与 `MLP_ROLE_INDEX` 决定各节点
rank。每台机器有 1 TB 以上内存并不会改变加载策略：仍使用 mmap，避免 8 个 GPU rank 各自
展开全表。

launcher 的 `data.num_dl_workers=8` 是每个 GPU rank 的值，因此每个 8-GPU 节点实际有 64 个
DataLoader worker，而不是全节点只有 8 个。worker 按随机 target 独立读取不同叶目录，并启用
`persistent_workers=true`、`prefetch_factor=2`；当前不需要恢复到每 rank 16 workers。

新训练入口：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/\
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=protenix_v2_crop384_db48_distill_v1_50k_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_distill_50k_v1.sh \
  > "distill_v1.node${LOG_NODE_ID}.log" 2>&1
```

默认 outer mass 是 PDB `0.5`、蒸馏库 `0.5`。如需改变，只在新训练开始前给所有节点设置相同
正数；sampler 会自动归一化：

```bash
PROTENIX_PDB_SAMPLE_WEIGHT=0.3 \
PROTENIX_DISTILLATION_SAMPLE_WEIGHT=0.7 \
RUN_NAME=protenix_v2_crop384_db48_distill_v1_50k_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_distill_50k_v1.sh
```

PDB 数据保持原有 20% query-only、30% unpaired-shuffle、50% species-paired 以及独立 20%
template 请求。预测结构蒸馏数据没有 template：20% occurrence 使用 query-only 并跳过 A3M
I/O，其余 80% 直接读取 CIF 同目录的 `<stem>.a3m` 作为 unpaired MSA；不会访问全库
sequence→MSA JSON。

严格续训使用同一数据 profile，且权重必须与源 checkpoint 一致：

```bash
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<source_run> \
RESUME_STEP=999 \
RUN_NAME=protenix_v2_crop384_db48_distill_v1_resume_1000_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash resume_crop384_db48_ddp_distill_50k_v1.sh
```

runtime index 不需要在每次训练前重建。launcher 会 fail closed 检查 manifest、roots、路径数组和
cluster 数组；缺任何文件都会在 `torchrun` 前退出。详细 strict-resume 规则见
[crop384_db48_ddp_strict_resume_cn.md](crop384_db48_ddp_strict_resume_cn.md)。
