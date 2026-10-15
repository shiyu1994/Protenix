# AFDB + MGnify 短蛋白 Foldseek 聚类

本文记录现有蒸馏库中 `canonical_length <= 200` 蛋白的 Foldseek 聚类数据合同、
固定软件版本、运行参数和可恢复执行方式。

当前状态：CIF → bioassembly PKL 转换及转换审计已经完成；本文描述的 Foldseek
全库聚类不能据此视为已经完成。只有 full runner 正常退出，且持久化目录中同时
存在非空结果、阶段完成标记和 `provenance/completion.env` 时，才能宣布聚类完成。

## 1. 权威输入与精确规模

长度筛选的唯一权威输入是 CIF → PKL 转换产生的合并审计：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_bioassembly_plan_v3_18x100_20260725/distillation_bioassembly_indices.csv.gz
SHA256 c2ab75330588b70fcabccd3fa66cc07beb3c6ef6e4ecb0a015452657ac0c5439
```

对应的 root 定义是：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724/roots.json
SHA256 467f3119d78cbcaae34a82689fe4114e978ee3eea340b9297bce12acacefdbe2
```

转换完成标记 `_FINALIZE_SUCCESS.json` 记录：

- 输入及成功转换 CIF：33,498,651；
- 转换失败：0；
- 合并审计行数：33,498,651；
- 合并审计 SHA256 与上面的固定值一致。

直接流式读取审计中的 `canonical_length`，使用精确条件
`canonical_length <= 200`，得到：

| 指标 | 精确值 |
|---|---:|
| 蛋白数 | 15,404,287 |
| residue 总数 | 1,777,014,013 |
| 平均长度 | 115.358407 |
| 最大允许长度 | 200 |

这里的“200 aa 以下”包括长度恰好为 200 的蛋白。筛选不重新读取 CIF，也不重新
检查 A3M；它只使用已经完成的转换审计，因此避免再次扫描 3,350 万个结构文件。

三个 source root 的含义保持不变：

| root_id | 数据源 | 结构根目录 |
|---|---|---|
| `0` | AFDB | `/mnt/project_abd_hs/klp/databases/afdb/msas` |
| `1` | AFDB unique | `/mnt/project_abd_hs/klp/databases/afdb/models_unique_20260724` |
| `2` | MGnify | `/mnt/project_abd_hs/biran/data/mgnify/msas` |

## 2. 相对路径 manifest 合同

生产 manifest 由
`scripts/distillation/build_foldseek_short_protein_manifest.py` 生成。核心合同为：

- `short_le200.paths.tsv` 是 UTF-8、无表头、每行一个
  `root_id/cif_relpath`；
- 每行只保存相对路径，不重复保存三条长绝对根路径；
- 相对路径的基准目录是 manifest 旁边的 `roots/`；
- `roots/0`、`roots/1`、`roots/2` 是经过解析验证的 source-root 符号链接；
- `short_le200.mapping.tsv.gz` 是无表头 gzip TSV，三列依次为
  `selected_ordinal`、`root_id/cif_relpath`、`canonical_length`；
- `short_le200.stats.json` 记录输入、筛选规模、内容摘要和各 root 计数；
- `short_le200.pilot.paths.tsv` 是固定 seed 的 hash 子集，可用于稳定 pilot。

Foldseek `createdb` 必须以 `roots/` 为当前工作目录读取 manifest。新增 runner 会
自动切换到该目录；不要把 1,540 万个 CIF 复制或软链接到一个扁平目录。

manifest 的一次性生成命令如下：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

PY=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
BASE=/mnt/project_abd_hs/klp/databases/afdb_mgnify_foldseek_le200_tm0.8_cov0.90_20260725
INPUT_INDEX=/mnt/project_abd_hs/klp/databases/afdb_mgnify_bioassembly_plan_v3_18x100_20260725/distillation_bioassembly_indices.csv.gz
ROOTS_JSON=/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724/roots.json

"$PY" scripts/distillation/build_foldseek_short_protein_manifest.py \
  --input-index "$INPUT_INDEX" \
  --roots-json "$ROOTS_JSON" \
  --output-dir "$BASE/manifest" \
  --max-length 200 \
  --pilot-hash-modulus 154 \
  --pilot-hash-buckets 1 \
  --pilot-hash-seed protenix-afdb-mgnify-foldseek-20260725
```

该命令实际生成 99,888 条稳定 pilot 记录。完整 manifest 的内容 SHA256 为
`72c5655950119771f79f9ebdaf75b2917b22c0d2597791df1297ab60158afe38`，
pilot manifest 的内容 SHA256 为
`4604e445f6e9c9175da6c8595aa4b9c66c44b9b6e27ae74563d6b7c22b955311`。

发布后的最低检查为：

```bash
test "$(wc -l < "$BASE/manifest/short_le200.paths.tsv")" -eq 15404287

"$PY" - "$BASE/manifest/short_le200.stats.json" <<'PY'
import json
import sys

stats = json.load(open(sys.argv[1], encoding="utf-8"))
counts = stats["counts"]
assert counts["selected_rows"] == 15_404_287
assert counts["selected_length_sum"] == 1_777_014_013
assert counts["selected_length_max"] == 200
print(counts)
PY
```

若输出目录已经存在，不要随意加 `--overwrite`。先比较 stats 和内容摘要；只有
明确要发布一个新 manifest 时才使用新目录，避免正在运行的 Foldseek 输入发生
变化。

## 3. 固定 Foldseek release 10 与 createdb CIF fast path

本任务以 Foldseek release `10-941cd33` 为固定上游基线。下载的官方 Linux AVX2
包保留用于对照：

```text
官方 URL
https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz

本地归档名
foldseek-linux-avx2-10-941cd33.tar.gz

归档 SHA256
e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7

foldseek version
941cd33ff0771cd2e3f144e3293e22a2b87e9fda

当前解压后二进制 SHA256
4db94a7825d409e416643772860ddf7cfe328d582679b47e6dffda7df4d41ea0
```

官方资料：

- [Foldseek release 10-941cd33](https://github.com/steineggerlab/foldseek/releases/tag/10-941cd33)
- [Foldseek 官方 README](https://github.com/steineggerlab/foldseek)
- [官方 README 的 clustering 说明](https://github.com/steineggerlab/foldseek#cluster)

生产任务使用基于同一 release 10 源码构建的 AVX2 二进制，只增加一个
`createdb` CIF 路径识别 fast path：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_foldseek_le200_tm0.8_cov0.90_20260725/software/foldseek-10-941cd33-cif-fastpath/bin/foldseek

foldseek version
8150259ad62aadafd0a49bf4dc56bc20712a4a8d

binary SHA256
bffc9d6e38971b34f9f7372bc62c745f6672eae1e6a396bdfbbde6996cad360c

patch SHA256
5dc65d740320af5d155bb45d336c53d279672a17da3c03deaffecb208d1db8b6
```

补丁只让扩展名明确为 `.pdb`、`.cif`、`.mmcif` 或相应 `.gz` 的输入直接进入
loose structure file 分支，跳过对每条路径串行探测 `<path>.dbtype` 的共享文件系统
访问；结构解析和 DB 写入代码没有改变。10,000 个真实且不与 pilot 重叠的 CIF
A/B 实验中，patched/official 本地构建分别用时 3.284/7.797 秒，patched 快
2.37 倍，14 个 createdb DB 文件逐字节相同。另以 64 个 CIF 完成聚类端到端检查，
最终 TSV 逐字节相同。

补丁、构建参数、工具链版本和 A/B 结果分别持久化在：

```text
scripts/distillation/patches/foldseek-10-941cd33-createdb-cif-fastpath.patch
$BASE/software/foldseek-10-941cd33-cif-fastpath/PROVENANCE.md
```

每次新建 run 前先验证，不允许 PATH 中的其他 Foldseek 静默替换：

```bash
FS=/mnt/project_abd_hs/klp/databases/afdb_mgnify_foldseek_le200_tm0.8_cov0.90_20260725/software/foldseek-10-941cd33-cif-fastpath/bin/foldseek

test "$("$FS" version)" = 8150259ad62aadafd0a49bf4dc56bc20712a4a8d
test "$(sha256sum "$FS" | awk '{print $1}')" = \
  bffc9d6e38971b34f9f7372bc62c745f6672eae1e6a396bdfbbde6996cad360c
```

runner 还会把实际 `version` 输出、二进制 SHA256、manifest SHA256 和固定参数写入
`provenance/`。在同一个持久化输出目录中更换二进制、manifest 或参数会被拒绝。

## 4. easy-cluster 的用途与生产流程选择

对小目录做一次性测试时，官方最简接口是：

```bash
foldseek easy-cluster INPUT RESULT_PREFIX TMP [参数]
```

它通常产生代表—成员 TSV、代表序列 FASTA 和全成员 FASTA。按照本任务参数，
小规模 smoke 命令可以写成：

```bash
"$FS" easy-cluster INPUT result tmp \
  --threads 128 \
  --alignment-type 2 \
  --tmscore-threshold 0.8 \
  --tmscore-threshold-mode 2 \
  --exact-tmscore 1 \
  -c 0.9 \
  --cov-mode 0 \
  --cluster-mode 0 \
  --cluster-reassign 1
```

但 15,404,287 个结构的生产任务不直接使用 `easy-cluster`。生产使用：

```text
createdb → cluster → createtsv
```

原因不是更换聚类算法，而是把大任务拆成可检查、可持久化、可恢复的阶段：

1. `createdb` 将一次性读取大量分层 CIF 的成本固化为 Foldseek DB；
2. 输入 DB 和 cluster DB 可分别发布到共享持久化存储；
3. 中断后可跳过已有完成标记的阶段，必要时从持久化 DB 重新 staging；
4. 每阶段有独立日志和 provenance，便于定位 createdb、prefilter 或 cluster 问题；
5. 本任务只需要代表—成员 TSV，不需要 `easy-cluster` 额外生成的两个 FASTA。

新增的生产入口是：

```text
scripts/distillation/run_foldseek_short_cluster.sh
```

runner 内部固定执行 `createdb + cluster + createtsv`，并固定下面一节的聚类语义。

## 5. 最终聚类语义

生产参数固定为：

```text
--alignment-type 2
--tmscore-threshold 0.8
--tmscore-threshold-mode 2
--exact-tmscore 1
-c 0.9
--cov-mode 0
--cluster-mode 0
--cluster-reassign 1
--threads 128
```

逐项解释：

- `--alignment-type 2` 使用快速的局部 3Di + amino-acid alignment；
- 不使用 `--alignment-type 1`，因此不执行慢的全局 TMalign；
- `--exact-tmscore 1` 对候选 alignment 启用精确 TM-score 计算，但不把整个
  alignment 流程改成全局 TMalign；
- `--tmscore-threshold 0.8 --tmscore-threshold-mode 2` 使用成员长度归一化的
  TM-score 阈值；
- `-c 0.9 --cov-mode 0` 要求代表和成员两个方向的 coverage 都达到 0.9；
- `--cluster-mode 0` 使用 greedy set-cover；
- `--cluster-reassign 1` 在 cascaded clustering 后重新分配成员，修正不满足
  最终聚类条件的初始分配；
- `--threads 128` 在当前 168 logical CPU 的机器上留出系统和 I/O 余量。

项目把该合同简称为“member-normalized exact TM-score >= 0.8，双向 coverage
>= 0.9”。Foldseek release 10 的 type-2 TM-score 检查在源码中以
`tmscore < threshold` 拒绝候选，因此恰好等于 `0.8` 的记录可通过；CLI help 中
简写的 `>` 不应被解释为排除浮点值恰好等于阈值的候选。

这个阈值只保证“成员—其最终代表”满足条件，不保证同一个簇内任意两个成员之间
也满足 TM-score 0.8 或双向 coverage 0.9。set-cover 聚类的两个成员可能都与代表
足够相似，但彼此低于阈值；下游不能把该 TSV 当成全簇两两相似性证明。

此外，release 10 的 `cluster` workflow 隐式使用 `-e 0.01`、
`--max-seqs 1000`、自动 sensitivity 8 和三级 cascaded clustering。它不是对
全部蛋白对做穷举 alignment，因此也不能反向理解为“所有满足 TM-score/coverage
条件的蛋白必然处于同一个簇”；没有被候选阶段召回的相似蛋白可能保守地留在不同
簇中。

## 6. 当前单机 128 核与内存盘策略

2026-07-25 审计当前机器时：

- `nproc` 为 168；
- 物理可见内存约 1.8 TiB；
- `/dev/shm` 总计约 920 GiB，可用约 904 GiB。

因此可以在当前机器使用 128 threads。官方 README 给出的结构 DB 核心估算是每个
residue 约 `6 bytes Cα + 1 byte 3Di + 1 byte AA`。按本次
1,777,014,013 residues 计算，核心字段下限约 14.22 GB（13.24 GiB）；这不是
全流程峰值，实际还包括坐标 DB、索引、prefilter、cluster 中间结果、文件系统
缓存和线程工作区。

推荐布局：

```text
/dev/shm/
└── protenix-foldseek-le200-{pilot|full}/  # node-local work DB 和 tmp

/mnt/project_abd_hs/klp/databases/afdb_mgnify_foldseek_le200_tm0.8_cov0.90_20260725/
├── manifest/                              # 权威相对路径 manifest
├── software/                              # 固定 Foldseek
└── runs/
    ├── pilot_100k/                        # 官方 binary 的 99,888-entry pilot
    ├── pilot_100k_fastpath/               # patched binary 的等价 pilot
    └── full/                              # full 持久化 DB、结果、日志、provenance
```

不要把原始 CIF 再复制到 `/dev/shm`。让 `createdb` 从共享源读取一次，并把紧凑的
Foldseek work DB 和 cluster tmp 放到 `/dev/shm`，可以避免共享文件系统承受后续
大量随机临时 I/O，同时不浪费内存重复保存原始结构。

full run 默认要求 work filesystem 至少有 256 GiB 空闲。启动前必须重新检查：

```bash
df -h /dev/shm
free -h
```

99,888-entry pilot 的 input DB 与保留 tmp 合计约 554 MB；按条目数线性外推，
full run 约 80 GiB。Foldseek release 10 的 prefilter 内存估算公式代入本次
15,404,287 entries、1,777,014,013 residues 和 128 threads 后约为 106 GiB。
这两个值都远低于当前机器容量，但不能当作硬上界：候选极端密集时，prefilter
结果可能显著大于 pilot 外推。运行中以 `/dev/shm` 剩余 256 GiB 或
`MemAvailable` 低于 300 GiB 为预警线，低于约 128–150 GiB 时必须立即判断是否
有其他任务抢占资源或候选异常增长。`--min-free-gb` 只在 runner 启动时检查，
不能替代运行期监控。

`/dev/shm` 是易失的 node-local 内存盘：

- runner 在 `createdb` 和 `cluster` 各阶段成功后，把对应 DB prefix 复制到
  `/mnt/project_abd_hs` 的 persistent output；
- 每个完成阶段发布 marker，重跑同一命令时跳过已完成阶段；
- 如果 `/dev/shm` 仍在，保留的 tmp 可供相同命令诊断或继续；
- 如果机器重启导致 `/dev/shm` 丢失，runner 会从 persistent output 重新 staging
  已完成的 input DB 或 cluster DB，再从下一个未完成阶段继续；
- `--remove-tmp-files 0` 固定保留临时文件，不要在结果验收前手工删除；
- pilot 和 full 必须使用不同的 persistent output 与 work directory；
- 同一 output directory 有独占 lock，不能并行启动两个 runner。

中断发生在某个尚未发布 marker 的阶段时，应以完全相同的命令重启。不要修改
manifest、参数或 Foldseek binary，也不要伪造完成标记。

runner 的 marker 是恢复提示，不是完整性证明。极小概率下，如果进程恰好在 DB
复制完成后、marker 原子发布前退出，直接重跑可能因 work DB 已存在而失败；此时
应先验收并显式 adopt 已完成 DB，而不是伪造 marker。最终结果还必须使用第 8 节
的独立集合审计和 SHA256 固化。

## 7. pilot 命令与实测结果

稳定 hash pilot 共 99,888 条，覆盖三个 source root。它用于验证 createdb
吞吐、tmp 增长速度、cluster 输出以及 patched binary 与官方 binary 的逻辑等价：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

BASE=/mnt/project_abd_hs/klp/databases/afdb_mgnify_foldseek_le200_tm0.8_cov0.90_20260725
FS="$BASE/software/foldseek-10-941cd33-cif-fastpath/bin/foldseek"

bash scripts/distillation/run_foldseek_short_cluster.sh \
  --paths-tsv "$BASE/manifest/short_le200.pilot.paths.tsv" \
  --roots-dir "$BASE/manifest/roots" \
  --output-dir "$BASE/runs/pilot_100k_fastpath" \
  --foldseek-bin "$FS" \
  --mode pilot \
  --pilot-size 99888 \
  --threads 128 \
  --work-dir /dev/shm/protenix-foldseek-le200-pilot-100k-fastpath \
  --min-free-gb 16 \
  --name short_le200
```

patched pilot 总 wall time 约 58 秒，输出 99,888 个成员、96,233 个簇，最大簇
84 个成员；没有重复或缺失成员。官方 binary 的相同 pilot 输出相同的
representative-member edge 集合、相同代表分配和相同簇数，排序后的 TSV SHA256
均为
`e3509b4f1fb04deacf2470364f88e2424061ae96049f4c6a03656743eaf5ee33`。
两个未经排序文件的 SHA256 不同仅源于并行输出行序。

pilot 验收至少检查：

```bash
test -s "$BASE/runs/pilot_100k_fastpath/result/short_le200_clusters.tsv"
cat "$BASE/runs/pilot_100k_fastpath/provenance/foldseek.version.txt"
cat "$BASE/runs/pilot_100k_fastpath/provenance/run-config.env"
cat "$BASE/runs/pilot_100k_fastpath/provenance/completion.env"
du -sh /dev/shm/protenix-foldseek-le200-pilot-100k-fastpath
```

在 full run 前记录 pilot 的 wall time、最大 RSS、createdb/cluster 各阶段时间以及
tmp 峰值；没有 pilot 实测时不要给全库完成时间作确定承诺。

## 8. full 命令与恢复

pilot 验收通过后，full run 使用完整的 15,404,287 行 manifest：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

BASE=/mnt/project_abd_hs/klp/databases/afdb_mgnify_foldseek_le200_tm0.8_cov0.90_20260725
FS="$BASE/software/foldseek-10-941cd33-cif-fastpath/bin/foldseek"

bash scripts/distillation/run_foldseek_short_cluster.sh \
  --paths-tsv "$BASE/manifest/short_le200.paths.tsv" \
  --roots-dir "$BASE/manifest/roots" \
  --output-dir "$BASE/runs/full" \
  --foldseek-bin "$FS" \
  --mode full \
  --threads 128 \
  --work-dir /dev/shm/protenix-foldseek-le200-full \
  --min-free-gb 256 \
  --name short_le200
```

进程中断后使用完全相同的命令恢复。runner 会核对既有 run config，并按以下顺序
跳过已完成阶段：

```text
createdb.done → cluster.done → createtsv.done
```

full 的最终代表—成员文件为：

```text
$BASE/runs/full/result/short_le200_clusters.tsv
```

最终验收至少包括：

```bash
test -s "$BASE/runs/full/result/short_le200_clusters.tsv"
test -s "$BASE/runs/full/provenance/completion.env"
test -f "$BASE/runs/full/provenance/stages/createdb.done"
test -f "$BASE/runs/full/provenance/stages/cluster.done"
test -f "$BASE/runs/full/provenance/stages/createtsv.done"
cat "$BASE/runs/full/provenance/completion.env"
```

完整的流式集合验收命令为：

```bash
python scripts/distillation/audit_foldseek_clusters.py \
  --manifest-paths-tsv "$BASE/manifest/short_le200.paths.tsv" \
  --clusters-tsv "$BASE/runs/full/result/short_le200_clusters.tsv" \
  --report-json "$BASE/runs/full/provenance/cluster-audit.json" \
  --tmp-dir /dev/shm \
  --sort-parallel 8 \
  --sort-memory 8G \
  --min-free-gb 16
```

该脚本以外部 GNU `sort` 保持 Python 常量级内存，并验证：

- manifest basename/stem 全局唯一；
- TSV 行数和 manifest 行数一致；
- 每个输入成员恰好出现一次，没有缺失或额外成员；
- 每个代表属于 manifest 且也作为成员出现；
- 每个代表恰好有一条 `representative == member` 自环；
- 簇数与最大簇大小。

全量运行前的独立 stem 预检已确认 15,404,287 个 ID、重名为 0；排序 ID 文件
SHA256 为
`6e8feeac37826a3a139d64e55d747ff155b5cde6fc8c501bcb1912aa3b6d2867`。
最终还应抽样复算成员—代表的 TM-score 与双向 coverage。完成这些检查之前，不要
把 full TSV 接入蒸馏采样表，也不要删除 `/dev/shm` 中保留的诊断数据。
