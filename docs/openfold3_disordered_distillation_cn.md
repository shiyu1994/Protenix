# OpenFold3 disordered 蒸馏数据：构建、校验与训练接入

本文记录当前仓库对 OpenFold3 PDB disordered cross-distillation 数据的正式接入合同。适用代码库为：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler
```

默认数据根为：

```text
/mnt/project_abd_hs/klp/databases/pdb_disordered
```

训练前必须完成 alignment representative 补齐、结构转换和完整性校验，并生成
`protenix/_SUCCESS.json`。正式训练入口会在启动 `torchrun` 前检查这个成功标记；缺失时直接退出。

本文第 1–7 节记录 disordered 数据本身，仍然适用于当前训练。第 8–10 节以
`weighted_pdb_afdb_mgnify_short_long_disordered_v3` 为当前合同；此前的三源
`weighted_pdb_afdb_mgnify_disordered_v2` 没有被原地修改，其权重、增广和 checkpoint
只作为历史复现实验保留，不能 strict-resume 到 v3。

## 1. 数据范围：训练集是 28,567 条，不是 61,227 条

本地结构目录中的 61,227 个 PDB 条目是候选全集。当前训练不能直接遍历候选目录，而以 OpenFold3 官方
`disordered_cache_with_templates.json` 的 `structure_data` 为唯一清单：

- 候选结构：61,227 条；
- 官方过滤后的 disordered 训练结构：28,567 条；
- 这 28,567 条需要的唯一 alignment representative：23,126 个；
- 构建脚本把 28,567 写成正式期望值；cache 条目数不一致会 fail closed。

当前官方 cache 的本地位置和校验值为：

```text
/mnt/project_abd_hs/klp/databases/pdb_disordered/metadata/
disordered_cache_with_templates.json

SHA256:
3d319854f16d5c6260338a42ecc7a1d2832e5063b8c716267d40f09aba7024f8
```

官方源：

```text
https://openfold3-data.s3.amazonaws.com/pdb_training_set/dataset_caches/disordered_cache_with_templates.json
```

如果需要重新取得 cache，先下载到临时文件，校验后再原子替换：

```bash
export DISORDERED_ROOT=/mnt/project_abd_hs/klp/databases/pdb_disordered
mkdir -p "${DISORDERED_ROOT}/metadata"

curl -fL --retry 5 -C - \
  -o "${DISORDERED_ROOT}/metadata/disordered_cache_with_templates.json.partial" \
  https://openfold3-data.s3.amazonaws.com/pdb_training_set/dataset_caches/disordered_cache_with_templates.json

echo \
  "3d319854f16d5c6260338a42ecc7a1d2832e5063b8c716267d40f09aba7024f8  ${DISORDERED_ROOT}/metadata/disordered_cache_with_templates.json.partial" \
  | sha256sum -c -

mv \
  "${DISORDERED_ROOT}/metadata/disordered_cache_with_templates.json.partial" \
  "${DISORDERED_ROOT}/metadata/disordered_cache_with_templates.json"
```

不要用候选目录扫描结果重新生成训练清单，也不要通过隐藏参数修改正式的 28,567 条期望值。

## 2. 数据语义

disordered 文件是一份完整的 hybrid structure，而不是给 standard PDB 打补丁：

- 蛋白链及实验未解析蛋白区域的坐标来自 AlphaFold-Multimer v2.3；
- DNA、RNA、配体、离子和糖来自对齐后的实验结构；
- 蛋白—蛋白界面属于 AF-Multimer 预测；
- 蛋白—非蛋白界面并不是完整的联合预测结果。

因此不要把 disordered 与 standard 结构逐原子拼接。`occupancy` 也不是 pLDDT，不能把它解释成模型
置信度。

当前转换器使用：

```text
<structures-root>/<pdb_id>/<pdb_id>.npz
```

而不是同目录 CIF。原因是结构 NPZ 保留了当前转换需要的 `chain_id`、`label_asym_id`、
`auth_asym_id`、`entity_id`、`molecule_type_id`、occupancy 和 bonds；当前 CIF 主要保留
`atom_site`，不能可靠重建同一份训练元数据。

结构 NPZ 的分子类型映射固定为：

| `molecule_type_id` | OpenFold3 | Protenix |
|---:|---|---|
| 0 | protein | `protein` |
| 1 | RNA | `rna` |
| 2 | DNA | `dna` |
| 3 | ligand/other | `ligand` |

原子坐标监督有效性使用：

```python
np.isfinite(coord).all(axis=1) & np.isfinite(occupancy) & (occupancy > 0)
```

无效坐标会归零并由 `is_resolved` mask 排除。不能只检查 finite，也不能只检查 occupancy。

转换器还对官方数据中已审计出的结构边缘情况 fail closed：

- 同一 `(chain_id, res_id, ins_code)` 出现互斥 polymer/糖组分时，按最高 finite atom occupancy
  选一个组分，平局使用源文件顺序以保留 canonical first monomer；删除替代组分后，只重定向源
  NPZ 已声明且跨 source site 的化学键，不按空间距离猜键；
- OpenFold3 的一个 source `entity_id` 若跨 protein/ligand 分子类型，保留 source ID 作为审计
  注释，并为 Protenix 分配互不冲突的内部 numeric entity ID；
- cache interface 来自实验 parent，预测坐标中可能由同 entity 的另一条 chain copy 接触。因此先
  验证 entity pair 确实存在 5 Å 接触，再把 index 的 entity/chain 顺序规范成 Protenix runtime key；
- `3nzi` 的三个遗漏 peptide chain 只在全部已审计元数据、序列和接触图同时匹配时做定点修复；
  其他 NPZ-only chain 一律报错；
- 7 个已审计的 chain-copy permutation 条目只允许在同 entry、同 molecule type、alignment query
  完全相等时重指 representative，不做相似序列猜测。

## 3. alignment representative 是什么

官方 cache 的每条 polymer chain 都给出 `alignment_representative_id`。相同 query sequence
可以共享一个 alignment，因此当前结构的一条链不一定读取当前 PDB 名下的 NPZ。例如 cache 可能指定：

```text
current structure: 1b70, chain_id=1, label_asym_id=A
alignment_representative_id: 3hfz_A
```

训练时应读取 `3hfz_A.npz`，不是猜测 `1b70_A.npz`。bioassembly 中保存
`chain_id -> alignment_representative_id` 映射，DataLoader 按该映射直接读取 alignment NPZ。

需要同时遵守：

- representative 文件后缀是 `label_asym_id`，不是 `auth_asym_id`；
- chain/label asym ID 大小写敏感，绝对不能统一 `.lower()`；
- PDB ID 查找只尝试原值、lower 和 upper，chain ID 始终保持原大小写；
- alignment NPZ 是可信 OpenFold3 数据的 NumPy object array，读取需要 `allow_pickle=True`；
- 蛋白 unpaired 使用 `uniref90_hits`、`cfdb_hits`、`mgnify_hits`；
- 蛋白 paired 使用 `uniprot_hits`；
- RNA 使用 `nucleotide_collection_hits`、`rfam_hits`、`rnacentral_hits`；
- DNA 没有对应 alignment source 时保留 query；
- 当前 loader 直接保留官方 `deletion_matrix`，不经过 A3M 文本往返转换。

正式构建必须解析全部 23,126 个 representative。即使调试模式允许缺失文件返回零退出码，只要仍有
缺失，构建器也不会发布 `_SUCCESS.json`，所以正式训练不会静默退化为缺 MSA 的数据集。

## 4. 目标目录布局

完成后目录应为：

```text
pdb_disordered/
├── metadata/
│   ├── disordered_cache_with_templates.json
│   ├── alignment_download_report.json
│   ├── alignment_download_report_final_20260725.json
│   └── alignment_validation_report_final_20260725.json
├── structures/
│   └── structures/
│       └── <pdb_id>/
│           ├── <pdb_id>.cif
│           ├── <pdb_id>.npz
│           └── bioassembly/
│               ├── <pdb_id>.pkl.gz
│               └── <pdb_id>.build.json
├── alignment_arrays/
│   ├── <supplemental representative>.npz
│   └── alignment_arrays/
│       └── <original representative>.npz
└── protenix/
    ├── indices.csv.gz
    ├── alignment_representatives.tsv.gz
    ├── failures.tsv
    ├── manifest.json
    └── _SUCCESS.json
```

`indices.csv.gz` 不重复保存完整绝对 PKL 路径。`bioassembly_relpath` 列保存：

```text
<pdb_id>/bioassembly/<pdb_id>.pkl.gz
```

训练时以 `structures/structures` 为 `bioassembly_dict_dir` 安全解析相对路径。这样可以明显减少索引字符串
占用，也允许通过统一的数据根环境变量移动整个数据集。

每个 PDB 写入自己的 `bioassembly` 子目录，不存在多个 worker 写同一个 PKL 的情况。`.build.json`
记录 source/cache/output hash 和该结构的原始 index rows，供断点恢复。dataset 级 index、manifest
和成功标记只由主进程汇总写入。

## 5. 补齐 alignment：盘点、下载和断点续传

先设置公共变量：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/\
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

export PROTENIX_PYTHON=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
export PROTENIX_ROOT_DIR=/mnt/project_abd/public_databases/protenix.20260101
export DISORDERED_ROOT=/mnt/project_abd_hs/klp/databases/pdb_disordered
export DISORDERED_CACHE="${DISORDERED_ROOT}/metadata/disordered_cache_with_templates.json"
export ALIGNMENT_MAIN="${DISORDERED_ROOT}/alignment_arrays/alignment_arrays"
export ALIGNMENT_SUPPLEMENTAL="${DISORDERED_ROOT}/alignment_arrays"

mkdir -p "${ALIGNMENT_SUPPLEMENTAL}"
```

### 5.1 只盘点并导出缺失列表

以下命令不发起下载：

```bash
"${PROTENIX_PYTHON}" scripts/download_openfold3_disordered_alignments.py \
  --cache-json "${DISORDERED_CACHE}" \
  --alignment-dir "${ALIGNMENT_MAIN}" \
  --alignment-dir "${ALIGNMENT_SUPPLEMENTAL}" \
  --dry-run \
  --export-missing-prefix \
  "${DISORDERED_ROOT}/metadata/missing_alignment_representatives"
```

它会生成 `.ids.txt`、`.urls.txt`、`.tsv` 和 `.json`。如在另一台机器下载，可以把 cache 和
`.ids.txt` 一起复制过去，并通过
`--representative-list <missing.ids.txt>` 限制任务集合。

### 5.2 正式下载

```bash
"${PROTENIX_PYTHON}" scripts/download_openfold3_disordered_alignments.py \
  --cache-json "${DISORDERED_CACHE}" \
  --alignment-dir "${ALIGNMENT_MAIN}" \
  --alignment-dir "${ALIGNMENT_SUPPLEMENTAL}" \
  --workers 32 \
  --retries 5 \
  --timeout-seconds 180 \
  --report-json \
  "${DISORDERED_ROOT}/metadata/alignment_download_report_final_20260725.json"
```

`--alignment-dir` 可以重复：

- 所有目录都参与已有文件查找；
- 第一个匹配优先；
- **最后一个目录是下载目标**。

因此上面的新文件会写到较浅的 supplemental 目录，原始 241 GiB 数据不需要搬动。

下载器天然支持断点续传：中断后原样重跑命令即可。启动时会重新盘点两个目录，跳过已经存在的
representative，只下载仍缺少的文件。单个下载先写同目录临时文件，完成 NPZ schema/query
一致性校验和 `fsync` 后再 `os.replace()`；中断不会把半文件发布成正式 `.npz`。

### 5.3 全量完整性校验

下载全部完成后，对选中的 23,126 个现有 NPZ 逐个加载校验：

```bash
"${PROTENIX_PYTHON}" scripts/download_openfold3_disordered_alignments.py \
  --cache-json "${DISORDERED_CACHE}" \
  --alignment-dir "${ALIGNMENT_MAIN}" \
  --alignment-dir "${ALIGNMENT_SUPPLEMENTAL}" \
  --dry-run \
  --validate-existing \
  --workers 100 \
  --report-json \
  "${DISORDERED_ROOT}/metadata/alignment_validation_report_final_20260725.json"
```

这里会加载和反序列化所有 23,126 个 NPZ，校验阶段使用多进程；下载阶段仍使用更适合网络 I/O
的线程池。

`--dry-run` 下的未解析条目需要通过 report 显式检查：

```bash
"${PROTENIX_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DISORDERED_ROOT"])
report = json.loads(
    (
        root
        / "metadata/alignment_validation_report_final_20260725.json"
    ).read_text()
)
assert report["required_representatives"] == 23_126, report
assert report["validate_existing"] is True, report
assert report["validated_existing"] == 23_126, report
assert report["cache_sha256"] == (
    "3d319854f16d5c6260338a42ecc7a1d2832e5063b8c716267d40f09aba7024f8"
), report
assert report["unresolved_representatives"] == [], report
assert report["existing_validation_failures"] == [], report
print("alignment validation complete:", report["required_representatives"])
PY
```

## 6. 把结构 NPZ 转为 Protenix bioassembly

一台 128 核机器的正式命令：

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
PYTHONHASHSEED=0 \
"${PROTENIX_PYTHON}" scripts/build_openfold3_disordered_dataset.py \
  --cache-json "${DISORDERED_CACHE}" \
  --structures-root "${DISORDERED_ROOT}/structures/structures" \
  --alignment-dir "${ALIGNMENT_SUPPLEMENTAL}" \
  --alignment-dir "${ALIGNMENT_MAIN}" \
  --output-dir "${DISORDERED_ROOT}/protenix" \
  --workers 100 \
  --resume \
  --compresslevel 3
```

这里的 alignment 目录顺序与下载命令不同没有问题：构建器只查找文件，按第一个匹配优先；它没有
“最后一个目录为写入目标”的语义。

建议始终保留 `--resume`，包括第一次执行。第一次没有状态可复用时会正常转换；中断后重跑时会：

1. 重新计算 source NPZ 和官方 cache entry 的 hash；
2. 检查 `<pdb_id>.build.json`；
3. 解压并验证现有 PKL；
4. 对比 output hash；
5. 只跳过完全一致的条目，其余条目安全重建。

不要在多台机器上同时对同一个 `structures-root` 执行多个完整 builder。当前脚本是单主进程加本机
`ProcessPoolExecutor`，不是分布式 sharding 调度器；多次完整任务会竞争同一批 PKL 和 dataset
级 aggregate。128 核机器默认留出一部分 CPU 给共享存储、主进程和系统服务，所以使用 100 worker；
出现内存或共享盘压力时降到 64 或 96，结果语义不变。

调试时可以使用 `--limit N`，但 limit run 永远不发布 `_SUCCESS.json`。同样，
`--allow-missing-alignments` 只用于排查结构转换；只要 representative 不完整，也永远不会发布正式
成功标记。

## 7. 构建完成的正式验收

本次正式构建已于 2026-07-25 完成，最终结果为：

| 项目 | 结果 |
|---|---:|
| 官方 cache entry | 28,567 |
| 已校验 alignment representative | 23,126 / 23,126 |
| Protenix bioassembly pickle | 28,567 / 28,567 |
| index rows | 302,240 |
| chain / interface rows | 137,960 / 164,280 |
| conversion failures | 0 |
| index weighting failures | 0 |
| missing alignment representatives | 0 |

全量 alignment 内容校验使用 100 个进程，耗时 472.751 秒，报告为：

```text
/mnt/project_abd_hs/klp/databases/pdb_disordered/metadata/
alignment_validation_report_final_20260725.json

SHA256:
57e2e943fa051509a842455276c8ab33a578fa79ed0b28a6a2cf12ec8db95e60
```

本次实际补齐 2,030 个 representative，下载 11,557,081,419 bytes，最终下载报告中的
`failures` 和 `unresolved_representatives` 均为空。较早的
`alignment_download_report.json` 是补齐前的 dry-run 盘点，保留作审计，不能用它判断当前完成状态；
完成状态以 `alignment_download_report_final_20260725.json` 和上面的全量内容校验报告为准。

最终成功标记引用的 manifest SHA256 为：

```text
0c8e105e5f26a6076f1cc48bdafe8643f6e9092414e89a29d92d22882acbe810
```

先确认成功标记存在：

```bash
test -s "${DISORDERED_ROOT}/protenix/_SUCCESS.json"
```

再校验 manifest hash 和关键计数：

```bash
"${PROTENIX_PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["DISORDERED_ROOT"]) / "protenix"
success = json.loads((root / "_SUCCESS.json").read_text())
manifest_path = root / success["manifest"]
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)

assert hashlib.sha256(manifest_bytes).hexdigest() == success["manifest_sha256"]
assert success["entry_count"] == 28_567
assert success["pickle_count"] == 28_567
assert success["alignment_representative_count"] == 23_126
assert manifest["status"] == "complete"
assert manifest["success_eligible"] is True
assert manifest["counts"]["conversion_failures"] == 0
assert manifest["counts"]["index_weighting_failures"] == 0
assert manifest["counts"]["missing_alignment_representatives"] == 0
assert manifest["counts"]["validated_pickles"] == 28_567
print("OpenFold3 disordered dataset is training-ready")
PY
```

`failures.tsv` 在成功构建中只应剩表头。`manifest.json` 还记录 cache、source、bioassembly、index
和 alignment inventory 的计数、字节数与 hash，可用于后续审计。

本次全库构建还验证并固定了三类官方数据边缘值：

- cache 中 124 个 JSON `NaN` resolution 统一使用 Protenix 已有的 unknown sentinel `-1.0`；
- 147 个含 `UNX/element=X` 的条目按标准 Protenix mmCIF filter 删除未知组分，不能把未知元素
  猜成碳；其他未受支持的 `element=X` 继续 fail closed；
- 72 个 cache interface 引用了整条没有 resolved token centre 的真实链。该链可参与 cache
  entity-pair 一致性检查，但不能生成 chain/interface 训练行。

这些都是构建期数据规范化或采样过滤，不改变 Protenix 模型、loss 或训练流程。

最终产物还经过真实 Protenix 训练数据路径的只读 smoke test：

- `BaseSingleDataset` 通过 `bioassembly_relpath` 加载正式 `4bjs.pkl.gz`，完成 chain crop、官方
  alignment representative 解析、MSA 组装及 Protenix feature/label 生成；
- 输出为 240 token、2,021 atom、64 行 MSA，label 仍只有标准的 `coordinate` 和
  `coordinate_mask`；
- `4bjs` chain 1 的结构 label asym ID 是 `A`，但官方 representative 是 `4bjs_D`；loader
  使用 `4bjs_D.npz`，其 query 与目标序列完全一致，而 `4bjs_A.npz` 第 5 位不一致；
- interface entity key `('1', '1')` 能从持久化的 `entity_pair_to_chain_pairs` 解析到真实接触链；
- 另一个真实样本 `4clc_C.npz` 生成 `(2048, 179)` MSA，query row 精确匹配。

无 crop 的整个 `2j5g` full-assembly MSA 是一个约 6.7 万 raw row 的开发机峰值内存压力样本；
正式训练会先按选中的 crop/asym 取 MSA，并且生产节点内存超过 1 TiB。它不影响上述训练路径正确性，
但后续如做 DataLoader RSS/吞吐基准，应把 `2j5g` 纳入最坏样本集合。

当前数据顺序是“先确定结构 crop，再读 MSA”，不是先生成整份 assembly 的 MSA：

1. 从 bioassembly 结构计算最终 `selected_token_indices`，包括 `max_nuc_tokens` 和下述最小
   监督 token 修复；
2. MSA featurizer 只读取 crop 中出现的 chain/asym；
3. 对每条入选链仍使用完整 canonical query 读取并校验 A3M/NPZ，完成全长 row dedup、
   species pairing 和 deletion 语义；
4. 最后按 crop 的原始 residue 映射 gather MSA 列，再生成 feature/label。

因此，最小监督检查发生在任何 homolog A3M/NPZ I/O 之前。不应把 canonical query 直接改成 crop
子序列：spatial crop 可以不连续，A3M 小写 insertion、OpenFold3 query-row 校验、paired MSA
去重以及大小写敏感的 `label_asym_id` 都依赖完整序列。后续若优化长链 MSA，应实现保持完整 query
语义的 projected parser，而不是先裁短 A3M 字符串。

## 8. 当前 v3 训练采样和监督语义

专用数据 profile 是：

```text
weighted_pdb_afdb_mgnify_short_long_disordered_v3
```

其四个有序 outer sampling masses 默认为：

| 数据集 | dataset 名 | 默认 outer mass |
|---|---|---:|
| 实验 PDB | `weightedPDB_before210930_v20260101` | 0.50 |
| Short protein distill | `afdb_mgnify_short_distillation_v20260725` | 0.005 |
| Long protein distill | `afdb_mgnify_long_distillation_v20260725` | 0.495 |
| OpenFold3 disordered | `openfold3_disordered_v20260725` | 0.02 |

四个值是 outer sampling masses，不要求加和为 1。总 mass 为 1.02，sampler 归一化后，按上表顺序
的 dataset 选择概率约为
`49.0196% / 0.4902% / 48.5294% / 1.9608%`。Short + Long 的 raw mass 仍合计为 0.50，
其中 Short 是 AFDB/MGnify 蒸馏 mass 的 1%。disordered 实际约占 2%，内部继续使用官方
cache 的 cluster size 生成的 index `weights`；训练配置
`force_recompute_weight=false`，避免只按本地 28,567 条重新估算 cluster mass。

Short/Long 不是逐成员按自身长度划分，而是以当前 AFQ/MGN sequence cluster 的最终
`representative_id` canonical length 决定整簇归属：代表长度 `<=200` 的整簇进入 Short，
`>200` 的整簇进入 Long。生产分区为：

| 分区 | cluster | row |
|---|---:|---:|
| Short | 14,327,216 | 15,425,587 |
| Long | 16,163,340 | 18,073,064 |
| 合计 | 30,490,556 | 33,498,651 |

两个 runtime 必须具有相同的
pair ID `9c87d52dd446f622be689b7a09611f302d3791ac2448ecd86f8aee2488fb8734`
和 pair receipt SHA-256
`1b47dd0a7f15840b4a3f27ad67e1d821c954b263ee58c4c110c6c45a880a5476`。
Short/Long manifest SHA-256 分别为
`cab1953754d9c45302d8ab359006b21352d332628d767584471778433431ff1c` 和
`b935c0d623d1fc2982a1449fab7ffed50b259653e11fb4c41c7bda131f535d9a`。
builder 内部全量验证和独立 post-publish `verify --verify-recorded-sources` 均已通过。
Foldseek 目录这里只提供已审计的 canonical-length path 集合，不替代上述 sequence cluster。

如果新建训练确实要覆盖比例，四个变量必须在所有节点上保持一致：

```text
PROTENIX_PDB_SAMPLE_WEIGHT
PROTENIX_SHORT_DISTILLATION_SAMPLE_WEIGHT
PROTENIX_LONG_DISTILLATION_SAMPLE_WEIGHT
PROTENIX_DISORDERED_SAMPLE_WEIGHT
```

它们必须为正数，不要求手工加和为 1。正式 scratch 开始后不要改变；strict resume 会把数据合同与
checkpoint 对照。

历史 v2 的三源 raw mass `0.50 / 0.50 / 0.02`、dataset
`afdb_mgnify_distillation_v20260725` 和环境变量
`PROTENIX_DISTILLATION_SAMPLE_WEIGHT` 继续由 v2 profile/launcher 原样保留，仅用于恢复
同合同的历史 v2 run。

OpenFold3 在这里仅提供结构和 alignment 数据。数据进入训练后仍使用 Protenix 模型、特征、
loss 公式和训练循环；接入代码没有引入 OpenFold3 模型或另一套 loss。loss 层唯一新增的
dataset-specific 规则是 AF3 initial training 明确使用的 disordered protein-only diffusion
mask；另外增加了一个 Protenix 本地数据质量约束，保证每个 crop 至少包含 4 个有效监督 token。

该数据集配置为 `is_distillation=true`，因此沿用 Protenix 本来就有的蒸馏数据语义：训练加载时
resolution 设为 unknown，原有 `ProtenixLoss` 会按其 resolution 规则屏蔽 confidence 类监督。
MSE、bond、smooth-lDDT 和 distogram 仍走同一个 Protenix loss 公式。

### 8.1 与 AlphaFold 3 / OpenFold3 的 loss 语义差异

AlphaFold 3 没有定义独立的 `disorder_loss` 公式，但对 disordered 数据使用了 dataset-specific
loss masking：

- AF3 Supplementary Information §4.3 规定 pLDDT、PAE、PDE 和 experimentally-resolved
  confidence loss 只用于实验 PDB，不用于任何 distillation 数据；
- Table 6 规定 initial training 时，disordered hybrid structure 的非蛋白部分不参与 diffusion
  loss；后续 fine-tuning 阶段取消这个 mask；
- initial / fine-tuning 1 / fine-tuning 2 的 disordered sampling weight 分别是
  `0.02 / 0.01 / 0.02`；本地 OpenFold3 fine-tuning 3 是不含 disordered 数据的
  confidence-only 阶段；
- initial training 的 polymer–ligand bond loss weight 是 0，fine-tuning 才打开。

本地固定版本的 OpenFold3 参考实现与论文一致：`DisorderedPDBDataset` 没有新建一套 loss，而是把
四个 confidence loss weight 设为 0，并在 initial training 通过
`disable_non_protein_diffusion_weights=true` 让 MSE 和 smooth-LDDT 只监督 protein token。

当前 Protenix 接入已实现等价的 initial-training 语义：

- `is_distillation=true` 把 resolution 设为 `-1`，所以四个 confidence loss 被屏蔽；这一点与
  AF3/OpenFold3 一致；
- `disable_non_protein_diffusion_weights=true` 时，MSE 的 residual/denominator 和
  smooth-LDDT 的两个 pair endpoint 只保留 resolved protein atom；
- MSE 的 weighted rigid alignment 仍使用全部 resolved atom，与 OpenFold3 一致；
- 原始 `coordinate_mask` 不变，因此 bond、distogram 和 confidence 的既有语义不变；initial
  launcher 的 bond weight 本身为 0；
- 当前 disordered-v3 专用 profile 对四个训练 dataset 均固定
  `min_diffusion_supervised_tokens=4`，不原地改变历史 PDB-only、两库或 disordered-v2 profile。
  普通 PDB 与 Short/Long AFDB/MGnify 的
  eligible token 定义为“token 内至少一个 resolved atom”；disordered 在 protein-only loss
  开启时定义为“token 内至少一个 resolved protein atom”。检查 token 的全部 atom，不依赖中心
  原子是否已解析；
- disordered crop 层在 MSA I/O 前接收上述 mask，并补入距原采样中心最近的 eligible token，直到
  达到 4 个。crop 已满时优先按完整 `ref_space_uid` 移除不含已选 eligible token 的最远分组；
  Short/Long AFDB/MGnify 在当前 profile 同样启用 resolved-protein repair；普通 PDB 若极端
  complete-lig spatial crop 不足 4 个则走该数据集既有的 fail-and-resample；
- crop 后会从实际 cropped TokenArray/AtomArray 重新计数；feature/label 全部生成后，还会依据最终
  `coordinate_mask`、`atom_to_token_idx` 和 protein-only flag 再做一次硬校验，不能只相信 crop
  plan。最终实际数量写入 `basic_info.N_diffusion_supervised_token`，用于运行期审计；
- mini-rollout chain/atom permutation 仍可能改写 dataset 生成的 label，因此 minimum 会作为
  per-sample feature 继续传入模型。低于 4 的 permutation 候选会被拒绝，保留此前合法 label；
  permutation 结束后按实际 loss mask 再做硬断言，避免对称链映射到低解析度 copy 后把 4 降为 1；
- 常规 runtime window 同时记录 crop/feature 阶段的
  `N_diffusion_supervised_token`、post-permutation 最小值及 chain/atom 候选拒绝次数；
- 该下界保证 diffusion MSE 至少有 4 个带 label 的 token，但不等价于保证 distogram
  representative、bond edge 或 smooth-LDDT 距离阈值内 pair 的数量；这些是不同的 mask。

2026-07-25 使用正式环境和 32 个 fork worker 对全部 28,567 个 disordered bioassembly 做了只读
全量审计：28,567/28,567 成功，失败 0，resolved-protein token 最小值为 53，`<4` 的结构为 0；
唯一达到最小值的结构是 `1rwd`（409 个 resolved protein atom、53 个对应 token）。因此源结构
层面有充分余量，运行时的 4-token 检查主要防止 crop 选择退化。

默认开关是：

```bash
PROTENIX_DISORDERED_DISABLE_NON_PROTEIN_DIFFUSION_WEIGHTS=true
```

initial/random-init 必须保持 `true`，当前 scratch 与 strict-resume wrapper 会拒绝 `false`。
未来若需要 all-molecule diffusion，必须创建单独命名的 fine-tuning profile/launcher；不要在
同一 scratch run 的 strict resume 中途改变。开关值进入 training contract，`true` 与 `false`
的 checkpoint 合同不同。

`require_resolved_protein_crop=true` 是与上述 loss 开关独立的固定数据约束。即使未来新建
fine-tuning run 并把 diffusion mask 设为 `false`，蛋白 crop 约束仍保持开启。
当前实现会让该蛋白约束至少等于 `min_diffusion_supervised_tokens`，所以不是仅保留 1 个蛋白
token，而是仍保留至少 4 个 resolved-protein token。
`min_diffusion_supervised_tokens=4`、有效 token 定义、loss flag、crop flag 和插入策略版本都进入
canonical training data contract；通过 wrapper 或直接调用 `runner/train.py` 启动都会
被 strict resume 校验。正式 profile contract tag 包含 `minloss4_v2`，旧 minimum 策略的
checkpoint 不能严格恢复到当前配置。

当前 disordered-v3 profile 因启用下述 crop-aware partial query-only 与多链
segment/whole-physical-chain MSA 增广使用 schema 7；MSA mode policy 为
`categorical_crop_aware_partial_query_only_all_protein_multichain_segment_or_whole_shuffle_species_paired_v4`。
schema 7 锁定每个数据集的完整配置、全局配置与 dataset override 合并后实际生效的
MSA/template 配置、`train_ref_pos_augment`、`limits`，以及负值 sampler seed 继承全局
seed 后的有效值。
它还锁定 Short/Long 两个 runtime-index manifest、各自的 partition role、阈值 200、
相同 pair ID/pair receipt，以及 disordered 的 `indices.csv.gz`、`_SUCCESS.json`
和 manifest SHA256。manifest 必须为 `status=complete`、
`success_eligible=true`，并且 success 声明与 index hash 必须和实际文件一致。旧
`weighted_pdb_only_v1` 继续生成原 schema 3 payload；使用
2026-07-24 的真实 resolved config 和 run manifest 复算，旧 hash
`835a4089f4b20b138c310be537b95dedcdd0ca65cb7a8fd39eddc460947c3a08`
保持不变。

历史 `weighted_pdb_afdb_mgnify_disordered_v2` 的 schema-7 data-contract SHA-256
`c22ccd37db686dcad12d51f35bd51de23db9d9702c449f16dd74923fda124d59`
也只标识旧三源、`0.2/0.3/0.5`、segment-only 合同，不能用于 v3。v3 的
data-contract SHA-256 由正式 resolved config 和上述实际 artifacts 计算并写入
`manifest/run_manifest.json`。当前正式 v3 schema-7 training-data contract SHA-256 为
`b76b782fee58b6353992d2ebf8c4138173bf013f5dd7d05344abeaf739a8f55b`；
不要把 pair ID 误当作 data-contract hash。

本地 OpenFold3 的 fine-tune 1/2 配置把该开关设为 `false`；fine-tune 3 是不含 disordered 数据的
confidence-only 阶段。不能把 fine-tune 3 描述为继续训练 disordered diffusion。

上游依据：

- [AlphaFold 3 正文](https://www.nature.com/articles/s41586-024-07487-w)
- [AlphaFold 3 Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf)
- OpenFold3 `examples/training_yamls/initial_training.yml`
- OpenFold3 `openfold3/projects/of3_all_atom/config/dataset_configs.py`
- OpenFold3 `openfold3/core/loss/diffusion.py`

以下是本项目的 Protenix 数据增强配置，不是 AF3 论文给出的 disordered 专用策略。当前 profile
对 PDB、Short、Long 和 disordered 四个 dataset 统一使用
query-only / unpaired-shuffle / species-paired=`0.1 / 0.3 / 0.6`。这里 10% 是 sampled
target occurrence 的外层 mode gate，不是每条链各自独立抽 10%。protein template 关闭，
RNA MSA 开启。

### 8.2 crop-aware partial query-only

当前 profile 固定：

```text
enable_crop_aware_partial_query_only=true
partial_query_only_max_chains=2
```

10% query-only outer gate 命中后，处理顺序和边界如下：

1. 先确定结构 crop，再从 crop 中实际出现的 physical protein `asym_id` 构造候选集合。
   homomer 的不同 physical copy 具有不同 `asym_id`，所以可以独立被选。RNA、DNA 和
   ligand 不进入候选集合。
2. crop-visible protein chain 数 `N=1` 时沿用 legacy global query-only：该 occurrence
   跳过 protein/RNA homolog I/O，把 crop 后所有 authoritative target sequence 合成唯一
   global query row，并据此重算 row feature、`profile`、`deletion_mean` 和 alignment
   counts。contract 对罕见 `N=0` crop 也声明使用同一 legacy global fallback。
3. `N>=2` 时均匀采样
   `k ~ UniformInteger[1, min(2, N-1)]`，然后从按 `asym_id` 排序的候选中无放回选择
   `k` 条 protein chain。因而每个 multi-protein crop 至多两条 chain 被置为 query-only，
   并至少保留一条 protein chain 不受该增广强制删除 homolog。
4. 被选 protein chain 在 homolog I/O 前就被识别，既不读取 A3M，也不读取对应的
   OpenFold3 alignment NPZ；它只由结构中 authoritative canonical sequence 构造 query。
   assembled MSA 完成后还会做一次最终约束：被选 chain 的 row 0 保持 query，row `>0`
   全部为 gap，`deletion_matrix/has_deletion/deletion_value` 的对应非 query 值为 0，
   `profile` 重算为 query one-hot，`deletion_mean` 取 query deletion。
5. 未选 protein chain 继续走正常 Protenix MSA 读取、dedup 和 species pairing；
   multi-protein partial 分支中的 RNA 也继续正常读取 MSA，DNA/ligand feature 不变。
   如果最终只有一条 protein chain 保留 MSA，则把它的 paired-source hits 并入 unpaired
   pool，避免 species pairing 因没有第二条可配 protein chain 而丢弃这些来源。
6. “至少保留一条正常 protein chain”只表示代码不会把它强制 query-only；如果源 A3M/NPZ
   本身只有 query，增广不会凭空制造 homolog。
7. chain 选择 seed 来自 sampler occurrence 的独立
   `partial_query_only_chain_selection` hash stream，不消费 Python、NumPy 或 torch
   全局 RNG。由于候选集合取决于 crop，确定性范围是
   `occurrence_seeded_given_crop`：给定同一 occurrence 和同一组 crop-visible protein
   `asym_id`，`k` 与选择结果可复现；它不额外承诺 crop RNG 在任意 worker topology 下
   逐位相同。

外层 mode 仍记录为 `basic.msa_mode=query_only` 和
`basic.query_only_msa=true`；实际作用范围通过以下 telemetry 单独记录：

- `basic.N_msa_crop_protein_chain`；
- `basic.N_msa_query_only_protein_chain`；
- `basic.N_msa_retained_protein_chain`；
- `basic.msa_partial_query_only_applied`；
- `basic.msa_global_query_only_applied`。

所以 multi-protein partial occurrence 的 `N_msa` 可以大于 1，不能再把
`N_msa==1` 当作 10% query-only gate 的定义。strict training-data contract 继续使用
schema 7，但 MSA mode contract 是上述 v4 policy，并绑定 post-crop physical-asym 候选轴、
uniform `k`、最大 2 条、至少保留 1 条、selected/retained chain 处理、query 固定语义和
seed stream。正式 profile contract tag 新增 `partialquery_max2_keep1_v1`。

2026-07-25 使用当前正式文件做了两层只读 smoke：

- `1a0c` 的 4-chain、384-column、MSA depth 16,384 直连 featurizer 检查通过；
  seed 456 选择 1 条 physical chain，保留 3 条，selected columns 的非 query row
  全为 gap 且 deletion feature 为 0，retained columns 仍有 homolog。开发机单进程
  冷启动耗时约 25.5 秒、峰值 RSS 约 1.37 GiB；
- `BaseSingleDataset` 从正式 `indices.csv.gz` 和 `1a0c.pkl.gz` 做 interface spatial
  crop，得到 384 token、3,113 atom、3 条 crop-visible protein chain、1 条
  query-only、2 条 retained、2,048 行 MSA 和 382 个 diffusion-supervised token。
  这条完整数据路径耗时约 14.4 秒、峰值 RSS 约 1.32 GiB。

以上是开发机单次冷调用，不是 8-worker prefetch 下的稳态吞吐承诺。上述
`c22ccd...` hash 属于历史 v2；当前 v3 的正式 schema-7 hash 是
`b76b782fee58b6353992d2ebf8c4138173bf013f5dd7d05344abeaf739a8f55b`。

### 8.3 crop-visible protein segment/whole-chain MSA shuffle

当前 profile 只在 30% `unpaired_shuffle` 分支启用
`data.msa.enable_virtual_segment_msa_shuffle=true` 和多链 variant sampling，参数固定为：

```text
virtual_segment_residue_gap=24
virtual_segment_min_length=9
enable_multichain_msa_shuffle_variant_sampling=true
multichain_msa_shuffle_segment_probability=0.5
```

处理顺序和边界如下：

1. 先确定 crop，再统计其中不同的 physical protein `asym_id` 数量 `N`。homomer 的不同
   physical copy 保持不同 group 候选；RNA、DNA 和 ligand 不计入 `N`，也不参与该 shuffle。
2. `N=0` 时 variant 为 `none`；`N=1` 时总是请求 segment variant；`N>=2` 时使用独立的
   `multichain_msa_shuffle_variant` occurrence hash stream，以 50%/50% 请求
   segment 或 whole-physical-chain。该 stream 不消费 Python、NumPy 或 torch 全局 RNG。
3. segment variant 按每条真实 protein `asym_id` 的 canonical `res_id` 建组。同链相邻已选
   residue 的 `res_id` 差值 `>=24` 时产生 raw segment；长度按 unique canonical residue 计数，
   不按 token 数计数。长度 `<9` 的 raw segment 合并到同一真实链最近的合格 segment，等距向左；
   若该链没有任何合格 segment，则整条已选链保持一个 group。
4. whole-physical-chain variant 为每个 crop-visible physical protein `asym_id` 建一个 group，
   即使多个 homomer copy 序列相同也不合并。它不是历史 paired-only shuffle。
5. 两个 variant 都先把所有 protein paired rows、unpaired rows、AFDB/MGnify sibling A3M rows
   和 OpenFold3 NPZ rows 合入 protein unpaired pool，再按 group 对全部非 query homolog rows
   独立排列。同一 group 内所有列及 `msa/deletion_matrix` 使用相同排列。
6. query row 始终固定；既有 paired-query placeholder 也保持固定。排列发生在 canonical
   residue 轴，随后由原有 `std_idxs` 广播到 token，因此一个非标准氨基酸拆出的所有 atom token
   始终得到相同的 MSA/deletion/profile 特征。
7. “segment 50%”是 requested variant 比例，不等于实际得到 disconnected segment 的比例。
   crop 没有 gap 时，segment grouping 会退化成每条 physical chain 一个 group；全 crop 最终少于
   两个 group，或没有足够 homolog row 时可以合法 no-op。requested、eligible 和 applied 必须
   分开审计。

每个样本和 batch summary 记录：

- `basic.N_msa_shuffle_group`；
- `basic.msa_shuffle_requested_variant_code`；
- `basic.msa_shuffle_variant_eligible`；
- `basic.msa_shuffle_variant_applied`；
- `basic.msa_shuffle_whole_physical_chain_applied`。

兼容审计还保留 segment-specific 的 `basic.N_msa_virtual_segment` 和
`basic.msa_virtual_segment_shuffle_applied`。schema-7 training-data contract 绑定
50/50 variant policy、独立 seed stream、gap/min-length、physical-asym/homomer/nonprotein
语义、全部 protein homolog row 范围和 query 固定语义。正式 contract tag 包含
`allmsashuffle_multichain_seg50_whole50_gap24_min9_v2`。

真实数据 smoke 已分别强制两个 variant：实验 PDB `6coi` 和 disordered `1a52` 都成功执行
segment 与 whole-physical-chain 路径，query row 保持固定，variant/group telemetry 与请求一致。
性能实现仍只复制 crop 实际覆盖的 group 列，不复制完整
`depth × full_sequence_length` MSA。

两个新 runtime 也通过真实 DataLoader 只读 smoke：Short 最后一行生成 27-token crop、
`N_msa=3`；Long 第 0 行生成 384-token crop、`N_msa=64`。对四源 sampler 做 200,000 次
确定性抽样得到 PDB/Short/Long/disordered =
`97,914 / 948 / 97,136 / 4,002`，与归一化后的期望比例一致；这是抽样 smoke，不是把每轮
dataset 数量硬编码成固定配额。

### 8.4 历史 disordered-v2 记录

为保证旧 checkpoint 可审计，v2 的冻结语义仍是：三源 PDB/AFDB+MGnify/disordered raw mass
`0.50/0.50/0.02`，MSA query-only/unpaired-shuffle/species-paired =
`0.2/0.3/0.5`；30% 分支对全部三源固定使用 segment grouping，gap 24、最短 9，不抽
whole-physical-chain variant。它的 profile tag 含 `allmsasegshuffle_gap24_min9_v1`，
schema-7 data-contract hash 是
`c22ccd37db686dcad12d51f35bd51de23db9d9702c449f16dd74923fda124d59`。
历史 `1a0c` segment-only smoke 得到 9 个 group；这些记录不改变当前 v3。

## 9. 16 节点正式启动

训练 topology 仍锁定为 16 节点 × 8 GPU，即 world size 128；每 rank 的
`data.num_dl_workers=8`。专用 wrapper 会锁定 disordered-v3 profile，不能通过命令行覆写科学配置。

AIStudio 的 16 个 worker 执行完全相同的命令，平台提供 `MLP_WORKER_NUM`、
`MLP_ROLE_INDEX`、`MLP_WORKER_0_HOST` 和端口：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/\
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v3_crop384_db48_disordered_v3_50k_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_disordered_v3.sh \
  > "disordered_v3.node${LOG_NODE_ID}.log" 2>&1
```

`RUN_NAME` 必须同时包含 `disordered_v3` 和 `v9_straggler`。wrapper 不接受命令行参数。
默认 raw masses 就是 `0.5 / 0.005 / 0.495 / 0.02`；正式 run 不建议覆盖。

默认路径不需要额外环境变量。移动数据根时，在所有节点设置相同值：

```bash
PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR=/new/shared/path/pdb_disordered \
PROTENIX_SHORT_DISTILLATION_RUNTIME_INDEX_DIR=/new/shared/path/short_runtime \
PROTENIX_LONG_DISTILLATION_RUNTIME_INDEX_DIR=/new/shared/path/long_runtime \
RUN_NAME=protenix_v3_crop384_db48_disordered_v3_50k_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_disordered_v3.sh
```

这个入口会检查 Short/Long 两个 AFDB + MGnify runtime index 的 manifest、pair receipt、
partition role、roots、row-offset 和 sampling arrays，
以及 disordered 的 index、`_SUCCESS.json`、结构根和两个 alignment 目录。任何必需路径缺失都会在
`torchrun` 前退出。

## 10. strict full-state resume

resume 必须继续使用同一个 disordered-v3 数据合同、16×8 topology 和源 checkpoint 的四项采样权重。
源 run 的 raw 与 EMA checkpoint 必须成对存在。下面以 `999.pt` 和
`999_ema_0.999.pt` 为例：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/\
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<source_run_dir> \
RESUME_STEP=999 \
RUN_NAME=protenix_v3_crop384_db48_disordered_v3_resume1000_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash resume_crop384_db48_ddp_disordered_v3.sh \
  > "disordered_v3_resume.node${LOG_NODE_ID}.log" 2>&1
```

`RUN_NAME` 要命名一个新的输出 run，不能直接覆盖源 run。正式 resume 的绝对目标仍是 50,000 次
optimizer update。不要用通用 scratch wrapper 恢复 disordered checkpoint，也不要在 resume 时临时
改变数据根、runtime index 或 dataset 权重。

v1/v2 checkpoint 不能 strict-resume 到 v3。它们的数据集顺序、outer weights、runtime artifact
身份和 MSA policy 均不同，runner 必须在加载权重前拒绝。历史
`run_crop384_db48_ddp_disordered_v2.sh` /
`resume_crop384_db48_ddp_disordered_v2.sh` 继续只服务同合同 v2；若未来需要参数初始化而非完整续训，
必须建立另一个明确命名的非 strict 工作流，不能把它称作 v3 strict resume。

## 11. 故障排查

### cache 报 61,227 或 entry count 不一致

使用了候选目录扫描结果或错误 cache。正式 cache 必须恰好有 28,567 个 `structure_data` entry，
并通过本文 SHA256。不要把候选全集强行传给 builder。

### alignment 仍有 missing

重跑第 5.1 节盘点，查看导出的 ID/URL 表或
`alignment_download_report_final_20260725.json`。确认：

- 两个 `--alignment-dir` 都传入；
- 目录真实存在；
- label asym ID 没有被转小写；
- 远端下载时同时使用同一份官方 cache；
- 外部下载结果复制到了当前配置查找的两个目录之一。

然后重跑正式 downloader；它只补仍缺的文件。

### `existing_validation_failures` 非空

NPZ 可能截断、object payload schema 错误，或不同 source 的 query row 不一致。先把问题文件移动到
隔离目录，再重跑 downloader 取回；不要直接关闭 validation，也不要继续发布 `_SUCCESS.json`。

### structure conversion failure

查看：

```text
${DISORDERED_ROOT}/protenix/failures.tsv
${DISORDERED_ROOT}/protenix/manifest.json
```

常见原因是源 NPZ 缺字段、原子字段长度不一致、bond index 越界、chain metadata 不一致，或官方
cache 与结构链不匹配。修复源文件后用同一命令加 `--resume` 重跑；不必删除其他已完成 PKL。

### 有 PKL 但没有 `_SUCCESS.json`

PKL 存在不代表全库通过。以下任一情况都会阻止成功标记：

- 使用了 `--limit`；
- conversion 或 index weighting 有失败；
- 不是完整 28,567 条；
- 任一 representative 缺失；
- 仅用了 `--allow-missing-alignments` 调试。

以 `manifest.json` 的 `counts` 和 `success_eligible` 定位原因，不要手工伪造 `_SUCCESS.json`。

### launcher 报 required path missing

先执行第 7 节验收。若 disordered 完整，再检查
`/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_short_le200_runtime_index_20260725`
和
`/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_long_gt200_runtime_index_20260725`
下的 runtime artifacts，并核对两边 pair ID/receipt。若数据根已移动，确保 16 个节点设置完全相同的
`PROTENIX_OPENFOLD3_DISORDERED_ROOT_DIR`、`PROTENIX_SHORT_DISTILLATION_RUNTIME_INDEX_DIR`
和 `PROTENIX_LONG_DISTILLATION_RUNTIME_INDEX_DIR`。

### MSA query 与结构序列不一致

不要退回按当前 `<pdb_id>_<label_asym_id>` 猜文件。先检查 bioassembly 保存的
`alignment_representative_id_by_chain_id` 是否与官方 cache 一致，再确认 representative NPZ
没有大小写错误或损坏。loader 会验证每个 source 的 query 与结构 canonical sequence。

### 100 worker 内存或共享盘压力过大

把 builder 的 `--workers` 降到 96、64 或更低并保留 `--resume`。不要启动多个指向同一输出根的
builder 来替代降并发；那会造成 aggregate 竞争，且不会提供正确的分布式调度。

## 12. 实现位置与上游依据

当前仓库实现：

- `scripts/download_openfold3_disordered_alignments.py`：representative 盘点、下载和 NPZ 校验；
- `scripts/build_openfold3_disordered_dataset.py`：28,567 条结构转换、resume、index 和 manifest；
- `protenix/data/core/openfold3_disordered.py`：结构 NPZ 到 Protenix `AtomArray`/bioassembly；
- `protenix/data/msa/msa_featurizer.py`：直接读取 alignment arrays、选择 crop-visible
  partial-query protein chain、约束 selected-chain feature，并在 canonical residue
  轴执行 segment/whole-physical-chain all-protein-row shuffle；
- `protenix/data/pipeline/sample_context.py`：occurrence-stable MSA mode、partial-query
  chain-selection seed、shuffle seed 与独立 multichain variant seed；
- `protenix/data/pipeline/dataset.py`：相对 PKL 路径、标准 Protenix 数据读取、seed 传递和
  partial-query/shuffle-variant 运行指标；
- `scripts/distillation/build_length_partitioned_runtime_indices.py`：按最终代表长度整簇构建和
  成对验证 Short/Long runtime；
- `configs/configs_data.py`：`openfold3_disordered_v20260725`、
  `afdb_mgnify_short_distillation_v20260725` 和
  `afdb_mgnify_long_distillation_v20260725`；
- `configs/crop384_db48_data_profiles.sh`：v3 四源
  `0.50 / 0.005 / 0.495 / 0.02` raw-mass profile，同时保留历史 v2；
- `run_crop384_db48_ddp_disordered_v3.sh`：v3 scratch 入口；
- `resume_crop384_db48_ddp_disordered_v3.sh`：v3 strict resume 入口。

OpenFold3 仓库只作为数据格式和预处理语义参考，固定在
`/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/openfold-3`
的 commit `cc7a1c5e3f3988381854bb373cc753b21d293f25`。训练运行时不 import OpenFold3，
也不使用它的模型、loss、optimizer 或训练循环。

上游资料：

- [AlphaFold 3 论文](https://www.nature.com/articles/s41586-024-07487-w)
- [AlphaFold 3 Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf)
- [OpenFold3 training data/cache 文档](https://github.com/aqlaboratory/openfold-3/blob/cc7a1c5e3f3988381854bb373cc753b21d293f25/docs/source/training.md)
- [OpenFold3 structure NPZ I/O](https://github.com/aqlaboratory/openfold-3/blob/cc7a1c5e3f3988381854bb373cc753b21d293f25/openfold3/core/data/io/structure/atom_array.py)
- [OpenFold3 exact-sequence representative mapping](https://github.com/aqlaboratory/openfold-3/blob/cc7a1c5e3f3988381854bb373cc753b21d293f25/openfold3/core/data/primitives/caches/filtering.py)
- [OpenFold3 MSA pairing](https://github.com/aqlaboratory/openfold-3/blob/cc7a1c5e3f3988381854bb373cc753b21d293f25/openfold3/core/data/pipelines/sample_processing/msa.py)
- [OpenFold3 initial training config](https://github.com/aqlaboratory/openfold-3/blob/cc7a1c5e3f3988381854bb373cc753b21d293f25/examples/training_yamls/initial_training.yml)

## 13. 2026-07-26 representative 运行时恢复与全库审计

### 13.1 问题与作用范围

两次 128 卡训练在同一个 disorder 样本 `2uy1` 上失败。现有 bioassembly 把 chain 1/2
分别映射到 `2uy1_A`/`2uy1_B`，但 alignment query 的唯一精确对应关系恰好相反：

```text
chain 1: 2uy1_A -> 2uy1_B
chain 2: 2uy1_B -> 2uy1_A
```

恢复逻辑显式限制为：

```text
dataset_name == openfold3_disordered_v20260725
并且配置了 openfold3_alignment_npz_dirs
```

weightedPDB、AFDB/MGnify、普通 A3M、其他 RNA 数据路径以及 inference 不进入该逻辑。

### 13.2 训练运行时策略

正常 representative 仍走原加载路径，不增加候选扫描。只有
`alignment query row does not match the canonical sequence` 这一精确错误会触发恢复：

1. 只检查同一个 bioassembly 已声明的 representative；
2. 使用目标链分子类型对应的 alignment sources 比较 canonical query；
3. 恰好一个候选精确匹配时，改用该候选；
4. 没有匹配或有多个匹配时，只把该物理链降为 canonical query-only；
5. 候选缺失或损坏只记入 recovery warning；正常加载路径中的其他 schema/I/O 错误仍然抛出；
6. 候选以 `(representative, chain_type)` 缓存，同一错误 representative 被不同 query
   使用时不会复用错误结果。

训练日志新增三个链计数：

```text
N_openfold3_representative_query_mismatch
N_openfold3_representative_remap
N_openfold3_representative_query_only_fallback
```

它们会进入 batch summary、rank-local runtime JSONL 和 runtime outlier。templatecache 派生库中的
相同实现继续遵守 `max_input_msa_depth=16384`，包括 mismatch recovery 候选。

现有正在运行的 Python 训练进程不会热加载这些修改；必须重启训练任务后才生效。

### 13.3 Builder 根因修复

生产 builder 只要收到 `alignment_dirs`，就会在第一次 core conversion 前加载该 entry
所有 declared representative 的 canonical query，并传入
`canonical_sequences_by_representative`。因此，即使不同 entity 的序列互换不会触发历史
`inconsistent sequences across chain copies` 异常，也会在写 PKL 前被纠正。

该修改只影响未来 rebuild；本次没有重建或改写当前 disorder 数据集。

### 13.4 只读全库审计

审计入口：

```text
scripts/audit_openfold3_alignment_representatives.py
```

工具只接受 `--dataset-root`，并校验 disorder manifest、`_SUCCESS.json`、index SHA/行数和
所有输入路径边界。它支持多进程、稳定分片、断点续跑 state JSONL 以及最终结果原子发布。
最终 v2 规则只用相同 polymer type 的候选，避免混合蛋白/RNA assembly 的假 `corrupt`。

2026-07-26 使用 h1003/h1004 各 128 workers 完成了 28,567 个 PKL、66,575 条 polymer chain
的全量审计，最长 shard 用时约 735.6 秒：

| 分类 | chain 数 |
|---|---:|
| `exact_current` | 66,550 |
| `remappable_unique` | 20 |
| `no_exact` | 2 |
| `corrupt`（当前 representative 与目标 polymer type 不兼容） | 3 |
| `missing` | 0 |
| `ambiguous` | 0 |
| structure-level corrupt | 0 |

20 条唯一可恢复链来自：

```text
2iuc, 2jgd, 2uy1, 2w2k, 5nsc, 6cbz, 6ifj, 6k78, 6obh
```

三个 type-incompatible current representative 是：

```text
2p6a chain 5  protein -> 1a34_B
5fn3 chain 5  protein -> 4oo1_K
7dol chain 2  RNA     -> 3drh_B
```

对应 canonical chain 长度分别只有 10、24、6，且同类型候选中没有精确匹配；现有 loader
会安全地产生该链 query-only，不会触发 2uy1 的 fatal query mismatch。两个 `no_exact`
是 `7l1f chain 4` 和 `7ozv chain 5` 的 RNA 链，新的运行时策略同样会降为 per-chain
query-only。

最终 v2 结果位于：

```text
/mnt/project_abd/klp/proteinx_output/
  openfold3_disordered_representative_audit_v2_20260726/
    audit.shard0.jsonl
    audit.shard0.summary.json
    audit.shard1.jsonl
    audit.shard1.summary.json
```

首轮 schema v1 结果只保留作审计过程证据；正式解读应使用上述
`exact_canonical_query_string_same_polymer_candidates_v2` 结果。

### 13.5 离线 representative 修复与修复后复审

2026-07-26 使用 `scripts/repair_openfold3_alignment_representatives.py`，把 v2 审计确认的
20 条 `remappable_unique` 映射离线写入以下 9 个 bioassembly：

```text
2iuc, 2jgd, 2uy1, 2w2k, 5nsc, 6cbz, 6ifj, 6k78, 6obh
```

修复只改变 `alignment_representative_id_by_chain_id[physical_chain_id]`，不改变
structure/token、canonical sequence、index、alignment NPZ 或采样权重。完整事务备份和
receipt 位于：

```text
/mnt/project_abd/klp/proteinx_output/
  openfold3_disordered_representative_repair_20260726/
```

修复后 h1003/h1004 的全库 v3 复审覆盖 28,567 个 PKL、66,575 条 polymer chain：

| 分类 | 修复后 |
|---|---:|
| `exact_current` | 66,570 |
| `remappable_unique` | 0 |
| `no_exact` | 2 |
| type-incompatible | 3 |
| `missing` / `ambiguous` / structure corrupt | 0 |

结果位于
`/mnt/project_abd/klp/proteinx_output/openfold3_disordered_representative_audit_v3_20260726/`。

### 13.6 两条 `no_exact` 的显式 query-only 固化

v3 全库复审确认只有：

```text
7l1f chain 4  RNA  17 nt
7ozv chain 5  RNA  33 nt
```

二者在同 structure 的全部同 polymer representative 中都没有 exact match，因此不做近似
猜测，直接为这两个 physical chain 固化 query-only。2026-07-26 使用
`scripts/mark_openfold3_no_exact_query_only.py`，仅在对应 bioassembly 新增：

```text
7l1f: openfold3_query_only_chain_ids = ["4"]
7ozv: openfold3_query_only_chain_ids = ["5"]
```

工具在提交前重新证明 exact match 数仍为 0，并以数据集独占锁、外部完整备份、
success-last 原子发布和异常回滚完成事务。structure/token、canonical sequence、
representative mapping、alignment NPZ、index 和采样权重均未改变。备份及 receipt 位于：

```text
/mnt/project_abd/klp/proteinx_output/
  openfold3_disordered_no_exact_query_only_marker_20260726/
```

发布后身份：

```text
manifest_sha256 =
  deb4efe44d30c4c0edf09088dc83609f56f8b2e9c44deb9fc3a08fe6b820f4c8
_SUCCESS.json sha256 =
  2a1ad11c618338653bbfd64b8c567492ce7d40576b63935e024c8c296cbf3add
bioassembly_content_manifest_sha256 =
  5490f9f1f6fa5d86dcc125841c312b8b435d4d9e8cbe1e624c7736141f172b9d
```

独立复核重新聚合全部 28,567 个 build state，并确认目标分类为
`explicit_query_only`；两条所在 structure 的其他 8 条 chain 仍全部为
`exact_current`。新 loader 对 marker chain 保留 canonical query，但不读取错误
representative NPZ、不扫描候选，直接构造 per-chain query-only MSA。该语义只在严格的
disorder 训练路径启用，inference 与其他数据集不变；派生库原有的 top60/id90 template
cache 和 `max_input_msa_depth=16384` 行为也不受影响。
