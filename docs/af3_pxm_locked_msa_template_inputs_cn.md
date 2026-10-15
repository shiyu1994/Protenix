# PXM-Legacy 评测的冻结 MSA 与 Template 输入

## 结论

PXM-Legacy checkpoint 对比必须复用同一份评测输入，不能在每次评测时重新搜索
MSA 或在线解析 template。当前合同将两类输入分别冻结：

- Template：独立 evaluation cache，固定
  `max_template_date=2021-09-30`、最多 4 个模板、禁止随机打乱，运行时
  `required`；
- MSA：不复制公共 A3M，而是逐文件锁定绝对路径、字节数、SHA-256、MSA
  depth、query 序列和对齐宽度；正式评测前重新解析并计算全部哈希；
- 任何缺失、损坏、策略不匹配或 target 覆盖不完整都必须在推理前失败，不能回退到
  在线 template，也不能静默换一份 MSA；
- 唯一不在公共 protein sequence mapping 中的 `7tdw` entity 1 使用评测专属
  overlay。公共 Protenix 数据库保持只读。

这份锁定保证的是“不同 checkpoint 使用完全相同的底层输入文件与 template
特征”。MSA 行仍由当前 Protenix test-stage 逻辑按固定 seed 采样；因此在相同
manifest、seed 和单卡执行顺序下可复现，但 MSA lock 本身不是一份预先物化的
sampled MSA tensor。

`2021-09-30` 是结构 template 的硬 cutoff，不是当前 MSA 序列库的发布日期
cutoff。MSA lock 冻结的是 `protenix.20260101` 已有 A3M 的当前字节内容；它不会
把 homolog 按发布日期重新过滤，也不能证明底层序列数据库只包含
`2021-09-30` 以前的数据。因此这套输入适合公平、可重复的 checkpoint 回归比较，
但若要声称严格的历史时间切分，还需要使用有明确日期快照的序列数据库重新搜索
全部 MSA，并发布新的 evaluation root/policy。

主评测流程与命令见
[AlphaFold 3 测试集定义与当前 checkpoint 评测方案](af3_testset_evaluation_cn.md)。

## 固定目录与产物

```text
PXM manifest:
/mnt/project_abd/klp/proteinx_eval_data/manifests/pxm_legacy_af3_like/

冻结输入根目录:
/mnt/project_abd/klp/proteinx_eval_data/cache/pxm_legacy_af3_like_cutoff20210930_v1/

├── manifests/
│   ├── protein_sequence_sources.json
│   ├── protein_sequence_inventory.json
│   └── _INPUT_SUCCESS.json
├── msa_template/
│   ├── seq_to_msa_template_dir.json
│   └── 7tdw/entity_1/
│       ├── pairing.a3m
│       ├── non_pairing.a3m
│       └── hmmsearch.a3m
├── template_features/
│   ├── <sequence_sha256前2位>/<sequence_sha256>/template_features.pkl.gz
│   ├── _manifests/build_manifest.jsonl
│   ├── _manifests/build_stats.json
│   └── _SUCCESS.json
└── msa_lock/
    ├── msa_manifest.json
    └── _MSA_SUCCESS.json
```

`_INPUT_SUCCESS.json`、template `_SUCCESS.json` 和
`_MSA_SUCCESS.json` 是三份不同的完成凭据；不能用其中一份替代另一份。

## 已落盘的全量数据统计

以下数字来自当前最终 full manifest、`_INPUT_SUCCESS.json` 和 template
`_SUCCESS.json`，不是估算：

| 项目 | 数量 | 状态 |
| --- | ---: | --- |
| combined full targets | 1,093 | 已落盘 |
| RecentPDB 子集 | 1,045 | 已落盘 |
| RNA-Protein 子集 | 23 | 已落盘 |
| dsDNA-Protein 子集 | 37 | 已落盘 |
| smoke targets | 12 | 已落盘 |
| unique protein sequences | 1,119 | 已落盘 |
| protein entity occurrences | 1,467 | 已落盘 |
| 有至少一个有效 template 的 unique proteins | 1,096 | 已落盘 |
| 无有效 cutoff 内 template、固定 empty/dummy 的 unique proteins | 23 | 已落盘 |
| template 构建失败或缺失 | 0 | 已落盘 |
| protein MSA entity occurrences / unique sequences | 1,467 / 1,119 | 已落盘 |
| protein A3M files | 2,238 | 已落盘 |
| RNA MSA entity occurrences / unique sequences | 42 / 33 | 已落盘 |
| RNA A3M files | 33 | 已落盘 |
| 全部锁定 A3M files | 2,271 | 已落盘 |

RecentPDB、RNA-Protein 和 dsDNA-Protein 是来源子集，彼此可能重叠，所以三者
数量不能直接相加得到 combined。

旧的 1,094-target manifest 中只有 `7to1` 被最终排除。它要求当前 runtime
尚不支持的 `intra_nuc` / `nuc_prot` 指标类型；排除后 full manifest 为 1,093。
`7tdw` 没有被排除，它由下面的 evaluation overlay 补齐输入。

MSA lock 已原子发布 `_MSA_SUCCESS.json`。其 manifest SHA-256 为：

```text
a9e3e513d62c51fdef233bdc7ff7d1473ec99a9c0bcc963eb39d5a543800a8ed
```

它的 `target_count` 和 `unique_protein_sequence_count` 分别严格等于 1,093 和
1,119。旧 1,094-target 集中的 `7to1` 与其他 target 共享一条 RNA sequence；
排除它后 unique RNA 仍为 33，但 RNA entity occurrence 从 43 降为 42。

## 独立 Template cache

### 固定策略

评测策略 ID 为：

```text
pxm_eval_top4_maxdate_2021_09_30_v1
```

每个唯一 protein query sequence 只构建一次：

1. 从对应目录读取 `concat.hhr`；若没有有效 HHR，则读取
   `hmmsearch.a3m`；
2. 走当前 `TemplateHitFeaturizer.get_templates` 的标准特征处理；
3. 严格排除 release date 晚于 `2021-09-30` 的结构；
4. 保持标准命中顺序，不做训练用的 top-k shuffle；
5. 最多保存 4 个已处理模板；
6. 将 query、来源搜索文件指纹、policy、cutoff、模板特征和模板 release date
   一起写入 content-addressed cache。

cache 路径只由 query 序列决定：

```text
template_features/<sha256[:2]>/<sha256>/template_features.pkl.gz
```

23 个 `complete_empty` 不是缺失输入。它表示搜索产物存在并已成功解析，但经过
cutoff 和有效性处理后没有可用候选；运行时为这些序列生成固定 dummy template。
相反，cache 文件缺失、损坏、policy 不一致、来源哈希变化或 payload 非法都不是
`complete_empty`，必须报错退出。

当前 template `_SUCCESS.json` 的 1,096 个非空序列由 1,095 个本轮
`complete` 和 1 个经完整校验后 `skipped_existing` 组成；另有 23 个
`complete_empty`，合计正好 1,119。

### 为什么不能复用训练 cache

训练 cache 的策略是
`top60_uniform20_identity_le_0.90_v1`：取消发布日期限制，在质量排序 top 60
中均匀随机保留 20 个候选，训练时再做最多 4 个模板的随机选择。它服务于数据增强，
而不是时间切分评测。

直接把训练 cache 用于 PXM 会产生三类问题：

- 可能包含 `2021-09-30` 之后发布的结构，形成 template 时间泄漏；
- top60/uniform20 的随机候选集合与固定 top4 的评测定义不同；
- 训练 cache 的 `chain_dummy` 容错会把损坏掩盖成无模板，破坏 checkpoint
  间输入一致性。

因此评测使用独立配置字段和独立根目录，并强制
`evaluation_sequence_template_cache_mode=required`。训练 cache 环境变量即使
残留在 AIStudio shell 中，也不能改变评测 policy。

## `7tdw` evaluation overlay

公共 `seq_to_pdb_index.json` 缺少 `7tdw` entity 1 的 186-aa protein query。
当前只在评测根目录新增一条 sequence mapping 和一套搜索产物：

```text
msa_template/7tdw/entity_1/
msa_template/seq_to_msa_template_dir.json
```

这套 overlay 的 `pairing.a3m`、`non_pairing.a3m` 和 `hmmsearch.a3m` 都非空，
首条 query 与结构中的 exact sequence 一致；三份文件的 SHA-256 和行数记录在：

```text
status/7tdw_entity_1.json
```

完整搜索命令和日志分别保存在：

```text
provenance/7tdw_entity_1_commands.sh
logs/7tdw_entity_1_msa_search.log
logs/7tdw_entity_1_template_search.log
```

原始 `hmmsearch.a3m` 可以包含 query 自身及 cutoff 之后的候选；它本身不能直接
证明没有时间泄漏。安全边界是后续已处理 template cache 对每个候选再次执行
`2021-09-30` cutoff，并由 validator 检查所有缓存候选的 release date。

7tdw 的 protein MSA 来自 Protenix 公共搜索服务，但服务没有返回数据库 snapshot
ID，不能声称它与 `protenix.20260101` 预计算语料逐 bit 等价。当前 overlay
字节已经通过 MSA hash-lock 固定后，仍可保证本项目后续所有 checkpoint 对比使用
同一份输入。

## MSA hash-lock

### 为什么引用原文件而不复制

1,119 个 protein query 的公共 A3M 已存在于
`protenix.20260101/mmcif_msa_template`。复制一份既浪费空间，也会产生两个需要
同步维护的数据副本。MSA lock 因此保存 resolved absolute path 和内容指纹，运行时
仍从公共只读目录读取。

每个 protein query 固定两份文件：

```text
pairing.a3m
non_pairing.a3m
```

每份文件记录并验证：

- resolved absolute path；
- `size_bytes`；
- SHA-256；
- A3M `record_depth`；
- 首条 ungapped query 的 SHA-256 与 exact-sequence 相等；
- 去掉小写 insertion 后每行的 aligned width 等于 query length。

manifest 还锁定 full target ID、full PDB-list 哈希、protein sequence-source
mapping 哈希、RNA mapping 哈希以及 exact sequence 到所有 PDB/entity occurrence
的覆盖。绝对路径发生变化也视为不同输入，需要显式重建 lock，不能自动接受。

### paired/unpaired 语义

hash-lock 不改变正常 Protenix MSA 处理：

- protein 继续同时读取 `pairing.a3m` 和 `non_pairing.a3m`；
- 多链 target 的 paired MSA 仍由原 species/pairing 语义在 featurizer 中组装；
- query 行必须存在且精确匹配，不会因 pairing 结果为空而丢掉 query；
- 底层 A3M 固定后，test-stage 仍以固定 seed 从最多 16,384 行的输入中按当前
  配置采样。相同顺序可复现，但改变 manifest 顺序或 seed 是新的评测合同。

### RNA、DNA 和 hybrid

只有 bioassembly 中 exact polymer type 为 `polyribonucleotide` 的 entity
读取 RNA MSA。其 exact sequence 必须存在于
`rna_sequence_to_pdb_chains.json`，并按当前 loader 语义使用 mapping 的第一个
EID：

```text
rna_msa/msas/<eid>/<eid>_all.a3m
```

RNA A3M 同样锁定绝对路径、size、SHA、depth、首条 exact query 和对齐宽度。
只有 query 的 depth-1 A3M 是合法输入，不需要伪造额外 homolog。

DNA 以及 DNA/RNA-hybrid polymer type 不会拿 RNA MSA；它们不进入 RNA lock
记录。复合物中的 protein chain 仍按 protein 规则读取自己的 paired/unpaired
MSA。这一点避免把 RNA mapping 错配给 DNA 或 hybrid chain。

## CPU-only data-pipeline smoke

在与正式 launcher 等价的 eval data config 下执行了不使用 GPU 的全链路 smoke，
`BaseSingleDataset` 正确得到 1,093 个 target。

`7tdw` 实测：

- 总计 214 token，其中包括 186 个 protein token 和 28 个 DNA token；
- overlay 的 pairing/non-pairing raw depth 为 6,226 / 4,954，query exact；
- featurizer 最终 MSA shape 为 `[11031, 214]`；
- required evaluation cache 实际加载 1 个 sequence cache、4 个有效候选；
- 候选为 `6xat_A`、`2a07_F`、`2kiu_A`、`3qrf_F`，release date 全部不晚于
  `2021-09-30`；
- 单条数据处理耗时 1.316 秒。

RNA target `8aw3` 实测：

- 总计 667 token，其中包括 590 个 protein token 和 75 个 RNA token；
- RNA raw key 为 `8aw3_1`，depth 144、query exact；
- featurizer 最终 MSA shape 为 `[9693, 667]`，其中 `rna_unpair=143`；
- 两个 protein sequence cache 共提供 8 个候选，合并进入 4 个 template slot；
- 单条数据处理耗时 2.869 秒。

整个 smoke 设置 `CUDA_VISIBLE_DEVICES=''`，未启动 GPU。它同时证明 overlay、
protein paired/unpaired MSA、RNA MSA 和 required template cache 已进入实际
runtime 数据链，而不只是离线 manifest 校验通过。

## 构建命令

以下命令用于完整重建；反复比较 checkpoint 时不应重复执行。先确保 7tdw overlay
已存在，并检查当前 Protenix 环境中的 Kalign 可执行。

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

PYTHON=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
DB=/mnt/project_abd/public_databases/protenix.20260101
MANIFEST_ROOT=/mnt/project_abd/klp/proteinx_eval_data/manifests/pxm_legacy_af3_like
EVAL_ROOT=/mnt/project_abd/klp/proteinx_eval_data/cache/pxm_legacy_af3_like_cutoff20210930_v1
KALIGN_BINARY_PATH=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/kalign

test -x "${PYTHON}"
test -x "${KALIGN_BINARY_PATH}"
test -s "${MANIFEST_ROOT}/combined_le1024.txt"
test -s "${EVAL_ROOT}/msa_template/seq_to_msa_template_dir.json"

"${PYTHON}" scripts/evaluation/prepare_af3_pxm_eval_inputs.py \
  --pdb-list "${MANIFEST_ROOT}/combined_le1024.txt" \
  --bioassembly-dir "${DB}/recentPDB_bioassembly" \
  --official-sequence-mapping "${DB}/common/seq_to_pdb_index.json" \
  --official-raw-root "${DB}/mmcif_msa_template" \
  --overlay-sequence-mapping "${EVAL_ROOT}/msa_template/seq_to_msa_template_dir.json" \
  --overlay-raw-root "${EVAL_ROOT}/msa_template" \
  --output-root "${EVAL_ROOT}"

"${PYTHON}" scripts/evaluation/build_af3_pxm_template_cache.py \
  --sequence-sources "${EVAL_ROOT}/manifests/protein_sequence_sources.json" \
  --cache-root "${EVAL_ROOT}/template_features" \
  --mmcif-dir "${DB}/mmcif" \
  --kalign-binary-path "${KALIGN_BINARY_PATH}" \
  --release-dates-path "${DB}/common/release_date_cache.json" \
  --obsolete-pdbs-path "${DB}/common/obsolete_to_successor.json" \
  --workers 24 \
  --resume

"${PYTHON}" scripts/evaluation/build_af3_pxm_msa_lock.py \
  --pdb-list "${MANIFEST_ROOT}/combined_le1024.txt" \
  --bioassembly-dir "${DB}/recentPDB_bioassembly" \
  --protein-sequence-sources "${EVAL_ROOT}/manifests/protein_sequence_sources.json" \
  --rna-sequence-mapping "${DB}/rna_msa/rna_sequence_to_pdb_chains.json" \
  --rna-msa-root "${DB}/rna_msa/msas" \
  --output-root "${EVAL_ROOT}/msa_lock"
```

Builder 会先删除旧 success marker，只有全量完成后才原子发布新 success marker。
构建中断时保留部分产物用于排查，但该目录不能用于正式评测。

## 独立验证命令

Template validator 会检查 sequence-source 文件哈希、1119 条完整覆盖、manifest
和 stats 哈希、每个 cache 文件哈希、payload policy、cutoff、候选数量以及每个
模板的 release date：

```bash
"${PYTHON}" scripts/evaluation/validate_af3_pxm_template_cache.py \
  --sequence-sources "${EVAL_ROOT}/manifests/protein_sequence_sources.json" \
  --cache-root "${EVAL_ROOT}/template_features"
```

MSA validator 不是只读取存储的统计。它会从 full 1,093-target manifest 和
bioassembly 重新计算 exact polymer coverage，重新解析并计算每一份 A3M 的哈希。
`--requested-pdb-list` 可以是 full 或 smoke，但只能是 full lock 的子集：

```bash
"${PYTHON}" scripts/evaluation/validate_af3_pxm_msa_lock.py \
  --pdb-list "${MANIFEST_ROOT}/combined_le1024.txt" \
  --bioassembly-dir "${DB}/recentPDB_bioassembly" \
  --protein-sequence-sources "${EVAL_ROOT}/manifests/protein_sequence_sources.json" \
  --rna-sequence-mapping "${DB}/rna_msa/rna_sequence_to_pdb_chains.json" \
  --rna-msa-root "${DB}/rna_msa/msas" \
  --output-root "${EVAL_ROOT}/msa_lock" \
  --requested-pdb-list "${MANIFEST_ROOT}/combined_le1024_smoke2pertype.txt"
```

full formal 时把最后一个参数改为：

```text
--requested-pdb-list "${MANIFEST_ROOT}/combined_le1024.txt"
```

## Launcher 自动预检

正式入口 `eval_af3_pxm_legacy_ema_checkpoint.sh` 必须在启动 checkpoint 推理前：

1. 检查 `_INPUT_SUCCESS.json`、template `_SUCCESS.json`、MSA
   `_MSA_SUCCESS.json` 和 7tdw overlay mapping 非空；
2. 运行 template 全量 validator；
3. 以 `combined_le1024.txt` 重算 full MSA lock，并用当前实际 smoke/full
   manifest 作为 `--requested-pdb-list`；
4. 任一 validator 非零退出时立即停止，不能进入 GPU 推理；
5. 强制 evaluation template cache 为 `required`，policy 必须等于
   `pxm_eval_top4_maxdate_2021_09_30_v1`；
6. 将公共 protein mapping 与 evaluation overlay mapping 同时交给 MSA
   featurizer，但不允许在线 template fallback。

launcher 已按上述合同接入，相关覆盖变量为：

```text
PXM_MSA_LOCK_ROOT
PXM_MSA_LOCK_MANIFEST
PXM_MSA_LOCK_SUCCESS
PXM_FULL_PDB_LIST
```

默认分别指向冻结根目录下的 `msa_lock/`、`msa_manifest.json`、
`_MSA_SUCCESS.json` 和 full `combined_le1024.txt`。validator 总是用
`PXM_FULL_PDB_LIST` 重算全部 2,271 个 A3M 的锁，再用本次实际 `pdb_list`
检查 requested smoke/full subset 没有越出 full lock。

若 `_MSA_SUCCESS.json` 缺失，dry-run 最多只能检查命令拼装，任何 smoke/full
推理结果都不能视为冻结输入下的正式结果。当前 marker 已落盘；正式运行日志仍应
同时出现 template validator 与 MSA validator 的 `status=valid` 摘要。

## 何时必须生成新版本

下列任一变化都应创建新的 evaluation root/policy 版本，而不是覆盖当前 lock：

- target manifest、bioassembly 或 query sequence 改变；
- 任一 protein/RNA A3M 内容或绝对路径改变；
- 7tdw overlay 重新搜索；
- template cutoff、候选排序或最大模板数改变；
- Protenix MSA pairing、采样 seed 或输入数据库语义改变。

checkpoint、模型权重和 diffusion step 数不属于输入 cache 本身；只要上述合同不变，
可以安全地在同一冻结输入上反复比较任意数量的 checkpoint。
