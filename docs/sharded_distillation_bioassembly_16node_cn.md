# AFDB/MGnify 蒸馏 bioassembly：16 节点生产方案

本文说明如何在 16 台可用 128 个 CPU 核的机器上，把 AFDB 和 MGnify 的预测结构
从 CIF 预处理为 Protenix 训练所需的 `pkl.gz`。本流程只处理 CIF，不再检查 A3M
query 与 canonical sequence 是否一致；已有的全库序列校验结果继续作为输入数据
不变的前置条件。

本文命令对应当前已经实现的 plan v3 CLI。

## 1. 为什么不能继续使用旧的目录调度方式

旧脚本虽然在每台机器上创建了一个多进程池，但 rank 一次只进入一个叶目录，
池内所有进程共同处理这个目录。结果是：

- 128 个进程同时读取一个输入目录并向同一个 `bioassembly/` 写文件；
- 其他叶目录在当前目录完成前不能开始；
- 节点之间使用静态 rank 分片，快节点不能帮助慢节点；
- plan 和 resume 会对每个 CIF 执行 `stat(size, mtime)`，给共享存储增加数千万次
  元数据操作；
- MGnify 的少数超大目录形成长尾，其他 CPU 已经空闲时仍可能只剩少量目录在运行。

plan v3 改为“叶目录优先、超大目录均匀切 chunk”：

1. 先把完整叶目录按 LPT（largest processing time first）分给 16 个 rank，同一
   叶目录不跨 rank；
2. 再在 rank 内把叶目录按最多 10,000 个 CIF 均匀切成 chunk；
3. rank supervisor 动态记录每个 leaf 的 active worker，只要还有未占用的 leaf，
   就不会给已经 active 的 leaf 提交第二个 chunk；
4. 因而正常阶段 128 个 worker 会尽量处理 128 个不同目录；只有每个剩余 leaf
   都已有 active worker 且仍有空闲核心时，才为 MGnify 超大目录开放额外 lane，
   消除尾部空转。

这里的“均匀切分”很重要。50,109 个 CIF 不应切成
`10000 + 10000 + 10000 + 10000 + 10000 + 109`，而应切成 6 个约
8,351/8,352 文件的 chunk。

## 2. 权威输入根和当前精确规模

生产 plan 必须同时包含以下三个根：

| root_id | source | 绝对路径 | 含 CIF 叶目录 | CIF 数 |
|---|---|---|---:|---:|
| `0` | AFDB | `/mnt/project_abd_hs/klp/databases/afdb/msas` | 4,883 | 22,803,271 |
| `1` | AFDB unique | `/mnt/project_abd_hs/klp/databases/afdb/models_unique_20260724` | 256 | 373,027 |
| `2` | MGnify | `/mnt/project_abd_hs/biran/data/mgnify/msas` | 296 | 10,322,353 |
| 合计 |  |  | 5,435 | 33,498,651 |

其中 AFDB 主库每目录最多 8,756 个 CIF，AFDB unique 每目录最多 1,574 个；
MGnify 有 99 个约 4,555 文件的目录，以及 197 个约 50,108 文件的超大目录。

在 `--max-files-per-task 10000` 下，预计得到：

- AFDB 主库：4,883 tasks；
- AFDB unique：256 tasks；
- MGnify 小目录：99 tasks；
- MGnify 超大目录：`197 × 6 = 1,182` tasks；
- 总计：6,420 tasks，平均每个 rank 约 401 tasks、2,093,666 个 CIF。

权威 root 配置使用已经生成的：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724/roots.json
```

相对路径必须始终与 `root_id` 配套解析。表格、plan、journal 和 index 中不要为
每个样本重复保存绝对根路径。

## 3. 输出与状态布局

每个含 CIF 的叶目录下面创建自己的 `bioassembly/`：

```text
leaf/
├── AF-...-model_v4.cif
├── AF-...-model_v4.a3m
└── bioassembly/
    ├── AF-...-model_v4.pkl.gz
    ├── indices.csv.gz
    ├── failures.tsv
    ├── _SUCCESS
    └── .state/
        ├── chunk-0000.lock
        ├── chunk-0000.progress.jsonl
        ├── chunk-0000.indices.csv.gz
        ├── chunk-0000.failures.tsv
        ├── chunk-0000._SUCCESS.json
        ├── chunk-0001.lock
        └── ...
```

同一 leaf 只归属一个 rank；同一 leaf 内不同 chunk 的 CIF 集合互不重叠。
chunk 使用独立 lock、journal、failure 和 success marker，所以尾段可以安全地
并行处理同一 MGnify 目录。

每个 pickle 使用 gzip level 3 和最高 pickle protocol。写入规则为：

1. 在最终输出所在目录创建唯一临时文件；
2. 完整写完并关闭 gzip stream；
3. 新文件权限遵循运行进程的 umask；16 节点 launcher 固定执行 `umask 0022`，
   因而新 PKL 为 `0644`，覆盖已有文件时保留其 mode；
4. 使用 `os.replace()` 原子发布最终 `pkl.gz`。

为了避免 3,350 万次同步写放大，不对每个 pickle 执行 `fsync()`。journal 按批
flush，chunk 完成时再同步必要状态并原子发布 `_SUCCESS.json`。进程或节点在
pickle rename 后、journal flush 前退出时，该文件最多会在 resume 时安全重做，
不会把半写文件当成成功结果。

叶目录的 `indices.csv.gz` 和 `_SUCCESS` 只能在该 leaf 的全部 chunk 成功后生成。
chunk/leaf marker 同时记录 index 的字节数和 SHA256；resume 或 finalize 遇到
index 损坏时不会误跳过。leaf 首次完成时用一次 `scandir()` 核对预期 PKL basename
全部存在，finalize 再做一次相同集合校验；这不对 3,350 万个文件逐一 `stat()`。
合并 index 时应流式读取 chunk shards，不能把整库结果装入 pandas。

## 4. 本流程不做 A3M 校验

转换 worker 只读取 `.cif`，不打开同名 `.a3m`，也不执行单独的 `audit-msa`
阶段。原因是 AFDB 已完成全库结构/A3M 序列一致性检查，MGnify 输入也保证结构
和 A3M 一一对应。

因此：

- plan 只枚举 `.cif`，跳过已有的 `bioassembly/`；
- A3M 文件保持原位且不被修改；
- pickle 中的 canonical sequence 仍由 CIF 解析结果产生；
- 训练接入时通过 `root_id + cif_relpath` 推导同目录 A3M 路径；
- 若 plan 之后增删或替换 CIF，应生成一个新 plan，不能复用旧 `_SUCCESS`。

## 5. 环境与 hosts 文件

下面命令在当前训练仓库根执行：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k

PY=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
PROTENIX_DATA=/mnt/project_abd/public_databases/protenix.20260101
ROOTS_JSON=/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724/roots.json
PLAN_DIR=/mnt/project_abd_hs/klp/databases/afdb_mgnify_bioassembly_plan_v3_20260724
HOSTS=/mnt/project_abd_hs/klp/databases/distillation_hosts_16.txt
BUILDER=scripts/build_sharded_distillation_bioassembly.py
LAUNCHER=scripts/launch_sharded_distillation_bioassembly.py
```

`$PROTENIX_DATA` 必须至少包含：

```text
common/components.cif
common/components.cif.rdkit_mol.pkl
```

`$HOSTS` 每个非空、非注释行放一个 SSH destination，行号从 0 开始对应 rank。
例如：

```text
worker-00
worker-01
worker-02
worker-03
worker-04
worker-05
worker-06
worker-07
worker-08
worker-09
worker-10
worker-11
worker-12
worker-13
worker-14
worker-15
```

所有节点必须满足：

- 可以从启动节点无交互 SSH 登录；
- 以相同绝对路径看到代码、三个输入根、CCD 根、plan 和输出；
- 使用同一 Python 环境和代码版本；
- 支持 `flock`，且共享文件系统的跨节点 advisory lock 行为可靠；
- `ulimit -n` 足够，但 worker 不应同时长期持有大量 CIF/PKL 文件描述符。

远程命令应固定：

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
MALLOC_ARENA_MAX=2
```

防止 128 个进程内部再次套用 BLAS/OpenMP 多线程。

## 6. 生成 plan v3

plan 只运行一次：

```bash
"$PY" "$BUILDER" plan \
  --roots-json "$ROOTS_JSON" \
  --world-size 16 \
  --workers-per-rank 128 \
  --max-files-per-task 10000 \
  --scan-workers 64 \
  --plan-dir "$PLAN_DIR"
```

生成后必须核对：

```text
roots = 3
leaf_count = 5,435
cif_count = 33,498,651
task_count = 6,420
world_size = 16
```

还要检查 16 个 rank 的 CIF 总数和 task 数是否接近。plan 阶段用固定上限的
`--scan-workers` 线程和动态 `scandir()` 队列并行枚举目录；默认 64，只读取目录项
及 CIF basename，不打开 CIF 或显式执行逐 CIF `stat()`。`bioassembly` 目录会被
跳过，目录 symlink 不会被跟随。worker 完成后再按 `(root_id, leaf_relpath)` 排序，
所以并发完成顺序不会改变 leaf ID 或文件名摘要。

rank 运行前只扫描自己拥有的 leaf 一次并验证同样的摘要。生产 plan 还会把
`roots.json`、同目录 `manifest.json` 的 SHA256 以及每个 rank JSONL 的 SHA256
固化进 `plan.json`，并逐 root 核对权威 `cif_files` 计数；任一目录项分类或扫描
错误、或者 rank plan 被删改时必须 fail closed。

不要在非空 `$PLAN_DIR` 上使用 `--force` 覆盖生产状态。输入快照变化后使用新的
plan 目录和新的 `plan_id`。

## 7. SSH dry-run 与正式启动

先执行 dry-run。默认不得启动远端任务，只打印 16 条完整、经过 shell quoting 的
SSH 命令：

```bash
"$PY" "$LAUNCHER" launch \
  --hosts "$HOSTS" \
  --plan-dir "$PLAN_DIR" \
  --python "$PY" \
  --protenix-root-dir "$PROTENIX_DATA" \
  --workers 128 \
  --compresslevel 3 \
  --tail-max-lanes-per-leaf 0 \
  --require-fast-path
```

核对 host/rank、Python、代码目录、plan、CCD 根、日志路径和 worker 数后，追加
`--execute`：

```bash
"$PY" "$LAUNCHER" launch \
  --hosts "$HOSTS" \
  --plan-dir "$PLAN_DIR" \
  --python "$PY" \
  --protenix-root-dir "$PROTENIX_DATA" \
  --workers 128 \
  --compresslevel 3 \
  --tail-max-lanes-per-leaf 0 \
  --require-fast-path \
  --execute
```

每个节点只启动一个 rank supervisor。supervisor 先在父进程预热 parser、CCD、
RDKit 和预测单体 fast path，再使用 POSIX `fork` 创建 worker，避免 128 个进程
分别冷启动和复制大缓存。supervisor 只维持最多 `workers` 个 in-flight future，
并根据 active leaf 状态逐个提交下一个 chunk。

`--tail-max-lanes-per-leaf 0` 表示只有进入尾段后，才允许一个超大 leaf 使用其
plan 中的多个 chunk lane；它不是从开局就让多个进程写同一目录。若希望任何时候
都绝不并行写同一 leaf，改为：

```text
--tail-max-lanes-per-leaf 1
```

严格模式会显著拉长 197 个超大 MGnify 目录的尾部时间，建议只用于共享存储对
同目录并发极其敏感时。

`--require-fast-path` 要求预测单体 CIF reader 和向量化补全路径都命中；不满足
准入条件的 CIF 明确记入 failure，而不是静默退回明显更慢的通用流程。

## 8. 查看状态

```bash
"$PY" "$LAUNCHER" status \
  --hosts "$HOSTS" \
  --plan-dir "$PLAN_DIR"
```

状态显示每个 rank 的：

- host、PID、state；
- 已完成 task 数、成功/失败 CIF 数；
- CIF/s、运行时间和最后 heartbeat 时间；
- 最近的 supervisor message。

若 30 秒内没有 chunk 完成，supervisor 仍会刷新一次 heartbeat 并记录 active/queued
task 数；worker pool 硬退出或调度器异常时 rank 状态会写成 `crashed`，不会长期
停留在旧的 `running`。

flock lease 和 PID 位于 `$PLAN_DIR/logs/`；细粒度 fast-path 命中数位于 leaf
及 chunk marker。共享 plan 目录只保存 16 个 rank heartbeat 和少量日志，不为
每个 CIF 创建共享队列锁。

所有 rank 对 `$PLAN_DIR/.activity.lock` 持共享锁，因此 16 个 rank 可以同时运行；
finalize 必须拿独占锁，若仍有任一 rank 活跃会立即失败。反过来，finalize 运行时
也不能启动新 rank，避免完成 marker 与仍在变化的 leaf 状态发生竞态。

## 9. 故障恢复

节点或进程异常退出后：

1. 先确认旧进程已经不存在，或其 rank lease 已经释放；
2. 检查对应 chunk 的 failure 和最后 heartbeat；
3. 使用原 plan、原 rank 映射和相同参数恢复。

先 dry-run：

```bash
"$PY" "$LAUNCHER" launch \
  --hosts "$HOSTS" \
  --plan-dir "$PLAN_DIR" \
  --python "$PY" \
  --protenix-root-dir "$PROTENIX_DATA" \
  --workers 128 \
  --compresslevel 3 \
  --tail-max-lanes-per-leaf 0 \
  --require-fast-path \
  --resume
```

确认后追加：

```text
--execute
```

若原节点不可用，可把 `$HOSTS` 对应 rank 的那一行换成替代节点，再以
`--resume --execute` 启动。因为输入、输出和 plan 都在共享路径，rank 不绑定物理
机器。禁止在旧 rank 仍持有 lease 时从第二台机器重复启动。

任何 rank 再次启动都会删除旧 `$PLAN_DIR/_FINALIZE_SUCCESS.json`；恢复完成后
必须重新执行 finalize，不能沿用修复前的全局完成标记。

恢复判定顺序为：

1. 对已有 completion marker 的 leaf 各做一次目录枚举；若缺 PKL，先失效它所属
   chunk marker，使同一次 resume 就能只重建缺失文件；
2. 匹配 `plan_id`、leaf 文件名摘要和 chunk 范围的 `_SUCCESS.json`：整块跳过；
3. 同时核对 chunk index 的 schema、size 和 SHA256；损坏时从 journal 重建
   index，不重做 CIF；
4. 未完成 chunk：读取该 chunk 最多 10,000 条 journal；
5. journal 尾部若是断电留下的未完成 JSON，持有 chunk lock 时先截断坏尾再续写；
6. 只验证 journal 中记录为成功的输出是否存在；
7. 缺失、失败或 journal 未记录的 CIF 重新处理。

不要在正常 resume 前递归扫描全部 3,350 万个 pickle，也不要删除完整
`bioassembly/` 后重跑。

## 10. finalize

全部 rank 和全部 chunk 成功后执行：

```bash
"$PY" "$BUILDER" finalize \
  --plan-dir "$PLAN_DIR" \
  --catalog "$PLAN_DIR/distillation_bioassembly_catalog.csv" \
  --merged-indices "$PLAN_DIR/distillation_bioassembly_indices.csv.gz" \
  --require-fast-path
```

本流程没有 A3M audit，因此 finalize 不应要求 `_MSA_SUCCESS`。finalize 必须：

- 验证全部 6,420 个 chunk marker；
- 验证 5,435 个 leaf `_SUCCESS` 和 index；
- 每 leaf 用一次目录枚举确认计划中的所有 PKL basename 存在；
- 流式合并 index，保留 `root_id + 相对路径`；
- 输出总 leaf、chunk、CIF、成功和失败计数；
- 不重新解析 CIF，不把 3,350 万行载入 pandas。

两个 audit 输出都被强制限制在 `$PLAN_DIR` 内，且代码明确拒绝
`distillation_index.csv.gz` 这个训练 catalog 文件名，避免参数写错时覆盖权威
聚类表。leaf audit 与 merged conversion audit 分别带
`artifact_role=distillation_bioassembly_leaf_audit` 和
`artifact_role=distillation_bioassembly_conversion_audit`。

只有 finalize 计数严格等于 plan、failure 为 0 且成功生成：

```text
$PLAN_DIR/_FINALIZE_SUCCESS.json
```

才算整体完成。生产命令请求了两个 audit，因此这个 marker 会绑定两者的
size/SHA256，并记录 `merged_audit_present=true`；若 merged audit 在 leaf audit
发布后失败，旧完成 marker 会被删除，不能仅以 CSV 文件存在作为成功依据。

这里的 `distillation_bioassembly_catalog.csv` 是转换完成情况的审计目录，
`distillation_bioassembly_indices.csv.gz` 也是转换侧审计 index；二者都不能替换
已有聚类采样权威表：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_distillation_catalog_20260724/distillation_index.csv.gz
```

训练侧继续从该聚类表采样，并用 `(source_root_id, cif_relpath)` 唯一定位样本。
对应 pickle 无需对 3,350 万行做 pandas 全量 join，可直接确定性派生：

```text
<dirname(cif_relpath)>/bioassembly/<cif_stem>.pkl.gz
```

训练默认的 `read_indices_csv()` 会在 pandas 读取前检查第一行
`artifact_role`，发现 leaf/conversion audit 时直接报错；只有显式
`allow_non_training_audit=True` 才允许把它作为人工审计数据读取。因此配置误指向
conversion audit 不会再静默使用 `NotInClusterTxt` 作为聚类训练输入。

如果训练代码仍通过现有 `read_indices_csv()` 全量读取 3,350 万行 CSV，relative
path 只能减少字符串字节数，不能消除 pandas DataFrame、全局 cache 和深拷贝的
峰值内存；正式接入应继续使用聚类 catalog 已生成的 mmap 采样数组，或实现
shard/lazy row store，而不是读取转换侧 merged index。

## 11. 已完成的 64/96/128 worker pilot

不要仅因为节点可用 128 核就直接认定 128 worker 最快。生产瓶颈可能从 CIF 解析
转移到共享存储的 open/create/rename 和 gzip 写入。

已在目标 `/mnt/project_abd_hs` 共享盘构造固定的 AFDB/MGnify 混合 pilot：

- 128 个不同 leaf；
- 每 leaf 50 个 CIF；
- AFDB 3,200 个、MGnify 3,200 个；
- 合计 6,400 个 CIF；
- 每个 leaf 只有一个 task，因此三组均不存在并行写同一目录。

gzip level 3、严格 fast path、冷输出目录的单节点结果：

| worker | 总时间 | prewarm | warm 转换时间 | warm CIF/s | 失败 |
|---:|---:|---:|---:|---:|---:|
| 64 | 44.02 s | 15.32 s | 28.69 s | 223.0 | 0 |
| 96 | 42.12 s | 15.46 s | 26.67 s | 240.0 | 0 |
| 128 | 38.37 s | 13.48 s | 24.89 s | 257.1 | 0 |

三组均为 fast reader `6,400/6,400`、fast completion `6,400/6,400`。每组输出
约 597.6 MB；该固定样本平均 `pkl.gz` 93.4 KB、p50 115.6 KB、p95 130.9 KB。
128 worker 相对 96 的 warm 吞吐只提高约 7.1%，说明单节点已经接近共享盘或其他
固定开销限制。

生产可以先使用 128 worker，但正式全库前应在 16 节点同时运行一个固定小 plan；
若全局吞吐相对 `16 × 96` 没有提升，改用 96 会给共享存储留出更稳定的余量。
生产 plan 不依赖 worker 数，同一 plan 可以用不同 `--workers` 恢复。

按单节点 warm 吞吐线性外推，16 台 128 核的理想下限约 2.3 小时；考虑共享存储
聚合元数据竞争、节点波动和冷启动，生产排期应按约 3–6 小时估算，并以 16 节点
小 plan 的实测总吞吐修正。

## 12. 容量与运维注意事项

全库会产生 33,498,651 个 pickle，是一个千万级小文件工作负载。开始前必须检查：

```bash
df -h /mnt/project_abd_hs
df -i /mnt/project_abd_hs
```

不能只看字节容量，还要确认 inode/文件数配额。按 pilot 中 AFDB 与 MGnify
分别约 121.7 KB 和 65.1 KB 的均值、再按整库数量加权，纯 pickle 粗估约
3.49 TB（约 3.18 TiB）。仍应至少预留 20%–30% 给长度分布偏差、临时文件、
index、journal 和文件系统元数据，即建议按约 4.5 TB 可用空间准备。最终容量以
更大随机样本为准。

其他注意事项：

- `bioassembly/` 写在 CIF 所在叶目录下，不建立一个含 3,350 万文件的全局平面
  输出目录；
- 临时文件必须与最终文件同目录，确保 `os.replace()` 不跨文件系统；
- 不进行每文件 `fsync()`，但 chunk marker 只能在全部输出关闭后发布；
- 日志避免记录每个成功 CIF 的完整绝对路径，失败记录才保留必要诊断；
- plan 和状态必须放共享持久存储，不能放节点 `/tmp`；
- 节点替换、SSH 断开或 supervisor 重启不要求重做已完成 chunk；
- 输入根一旦生成 plan 就视为只读；若内容变化，创建新 plan，不混用旧 marker；
- full run 前先验证共享 `flock`、原子 rename 和 gzip pickle 反序列化行为。
