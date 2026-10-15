# 按序列离线构建 Template Cache：Top60 均匀抽取、最多20个、Identity≤90%

## 目标

本实现把训练时昂贵且具有长尾的 mmCIF 读取、结构解析、Kalign
和模板坐标特征构建提前到离线阶段。每个唯一蛋白 query 序列生成一个
`template_features.pkl.gz`；训练只加载这个文件并从其中随机选择最多4个
模板，不再在线解析 mmCIF。

该策略的固定标识为：

```text
schema_version = 1
policy_version = top60_uniform20_identity_le_0.90_v1
```

## 已确定的候选策略

对 `seq_to_pdb_index.json` 中每个唯一蛋白序列执行：

1. 读取对应目录中的 `hmmsearch.a3m`。文件不存在或为空是合法的
   `N=0`，仍写入 `complete_empty` cache。
2. 执行原有的非日期预过滤：hit 名称合法、对齐覆盖率大于10%、
   模板至少10个残基，并排除长度超过 query 95% 的完整子串重复。
3. 按去 gap 后的 hit 序列去重。
4. 计算 query/template 在搜索对齐中的序列等同度：

   ```text
   identity = 相同残基的双非 gap 列数 / 双非 gap 列总数
   ```

   `identity > 0.90` 被排除，恰好 `0.90` 可以保留。A3M 小写插入不占
   query 对齐列。
5. 按 `sum_probs` 降序排序；分数相同以原搜索结果顺序稳定打破平局。
   `hmmsearch.a3m` 没有 `sum_probs` 时相当于保留源顺序。
6. 只保留质量排名 Top60。
7. 使用
   `SHA256(query_sequence + policy_version)` 作为确定性随机种子，将
   Top60 做无放回均匀随机排列。相同序列重复构建得到相同候选次序。
8. 按该次序执行 mmCIF 解析、Kalign 和模板特征构建。处理后再次按
   Kalign 对齐结果检查 `identity≤0.90`，并去除处理后模板序列重复。
9. 候选失败、等同度超限或重复时，继续使用 Top60 中后续候选补齐；
   收集20个有效模板后停止。不会越过 Top60。
10. Top60 全部处理完仍不足20个时，有几个就保存几个；最终为0时写
    `complete_empty`，训练使用 dummy template。

本策略**取消发布日期 cutoff**。同一个序列全局只有一个固定 cache，
不再使用 `target_release_date - 60 days` 或 `2021-09-30` 过滤。该选择
是有意的训练策略变化，防泄漏约束只保留 `identity≤90%`。

## Cache 路径与内容

输出路径沿用源映射目录：

```text
<CACHE_ROOT>/<seq_to_pdb_index相对目录>/template_features.pkl.gz
```

默认根目录：

```text
/mnt/project_abd/klp/databases/protenix_sequence_template_cache_top60_uniform20_id90_v1
```

每个 gzip pickle 保存：

- schema、policy、状态、原始 sequence 和 SHA256；
- 原始相对目录及源 hit 文件；
- 各阶段候选数、失败数、最终有效数；
- 最多20个候选的 PDB/chain、质量排名、抽样顺序、搜索对齐与
  Kalign 后 identity；
- 已对齐的完整模板特征；
- 有限数量的过滤详情及 Top60 内的处理失败详情；
- 构建耗时、策略参数和日期过滤关闭标记。

写入采用目标目录内临时文件、`fsync` 和原子重命名；`--resume` 只跳过
能够通过当前 schema/policy/sequence 校验的完整文件，旧版本或损坏文件
会原地重建。

## 双机并行划分

两台机器使用相同映射和输出根目录，但按稳定序列哈希分成两个互斥
shard：

```text
shard = uint64(SHA256(sequence)[:8]) % 2
```

- 当前 h1003 shell：`shard-id=0`
- `ssh h1004`：`shard-id=1`
- 每台默认128个 worker；BLAS/OpenMP 等原生库在 import 前固定为
  单线程，所以并发来自 worker 进程，不会产生 128×多线程过度订阅。
- worker 默认常驻（`MAX_TASKS_PER_CHILD=0`），避免重建进程池影响吞吐；
  如后续发现长期内存增长，可以显式设置非零值定期重启 worker。
- 两个 shard 写不同序列目录，并分别写 manifest、stats、log 和 PID。

入口脚本：

```text
scripts/template_cache/launch_two_host_sequence_template_cache.sh
```

默认数据：

```text
sequence mapping  /mnt/project_abd/public_databases/protenix.20260101/common/seq_to_pdb_index.json
search root       /mnt/project_abd/public_databases/protenix.20260101/mmcif_msa_template
mmCIF root        /mnt/project_abd/public_databases/protenix.20260101/mmcif
obsolete mapping  /mnt/project_abd/public_databases/protenix.20260101/common/obsolete_to_successor.json
Kalign            /mnt/project_abd/klp/miniconda3/envs/protenix/bin/kalign
Python            /mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
```

## 执行顺序

进入代码库：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90
```

先做只读预检：

```bash
bash scripts/template_cache/launch_two_host_sequence_template_cache.sh preflight
```

再执行两机 smoke。每台使用4个 worker，只处理其 shard 中前4条序列；
输出使用独立的 `_smoke` 根目录，不会污染正式 cache：

```bash
bash scripts/template_cache/launch_two_host_sequence_template_cache.sh smoke
```

smoke 成功后启动正式批量。该命令在本机和 h1004 后台各启动一个
128-worker builder，并默认携带 `--resume`：

```bash
bash scripts/template_cache/launch_two_host_sequence_template_cache.sh start
```

这三个命令也可以显式写出关键参数：

```bash
WORKERS=128 \
SMOKE_WORKERS=4 \
CACHE_ROOT=/mnt/project_abd/klp/databases/protenix_sequence_template_cache_top60_uniform20_id90_v1 \
REMOTE_HOST=h1004 \
bash scripts/template_cache/launch_two_host_sequence_template_cache.sh start
```

查看运行状态和最近日志：

```bash
bash scripts/template_cache/launch_two_host_sequence_template_cache.sh status
```

正常停止两个正式 shard：

```bash
bash scripts/template_cache/launch_two_host_sequence_template_cache.sh stop
```

重新执行 `start` 会通过 `--resume` 校验并跳过已经完成的 cache。

## 日志、进度与成功判据

正式构建文件：

```text
<CACHE_ROOT>/_logs/shard-00000-of-00002.log
<CACHE_ROOT>/_logs/shard-00001-of-00002.log
<CACHE_ROOT>/_manifests/shard-00000-of-00002.pid
<CACHE_ROOT>/_manifests/shard-00001-of-00002.pid
<CACHE_ROOT>/_manifests/manifest.shard-00000-of-00002.jsonl
<CACHE_ROOT>/_manifests/manifest.shard-00001-of-00002.jsonl
<CACHE_ROOT>/_manifests/stats.shard-00000-of-00002.json
<CACHE_ROOT>/_manifests/stats.shard-00001-of-00002.json
```

stats 每25条更新一次，重点检查：

- `completed_tasks` 持续增加；
- `remaining_tasks` 逐步降至0；
- `sequences_per_second` 非0；
- `status_counts.failed` 不存在或为0；
- 最终两个进程正常退出，两个 shard 都满足
  `completed_tasks == total_tasks`。

`complete_empty` 是“已成功处理但没有合格模板”，不是错误；
`skipped_existing` 是 resume 成功命中。真正需要排查的是 `failed`，
其完整异常和 traceback 保存在 shard manifest 对应行中。

## 训练读取方式

全量构建并审核完成后，正式 stage-init 与 strict-resume launcher 自动锁定：

```text
PROTENIX_SEQUENCE_TEMPLATE_CACHE_ROOT=/mnt/project_abd/klp/databases/protenix_sequence_template_cache_top60_uniform20_id90_v1
PROTENIX_SEQUENCE_TEMPLATE_CACHE_MODE=required
PROTENIX_SEQUENCE_TEMPLATE_CACHE_ERROR_POLICY=chain_dummy
```

用户不需要也不应在 AIStudio 命令中手工传这些变量；launcher 会拒绝继承的
同名覆盖，并在真实启动的 `torchrun` 前检查 cache 根目录和两个 shard 的完成
stats：`completed_tasks == total_tasks`、`remaining_tasks == 0`、
`failed == 0`。`DRY_RUN=1` 或显式 `SKIP_PATH_CHECK=1` 的无训练审计只锁定
变量而跳过共享存储检查，便于在未挂载生产数据库的主机验证 launcher。

`required` 模式只用于训练；当前 inference 明确保留原在线行为。训练中
20%的 template gate 仍按 target occurrence 抽取。被选中的 target 对
每条不同蛋白序列加载一次 cache，相同序列的多个 chain 复用；然后在
缓存的 `N≤20` 个有效候选中随机排列，并沿用当前规则每次选择最多4个。
`complete_empty` 正常产生 dummy template。某条 unique query sequence 的 cache
文件缺失、损坏或 schema/policy/sequence 不匹配时，不做 online fallback；
该 target 中具有同一 query sequence 的全部 chain 使用固定 dummy template，
其他 unique sequence 已成功加载的真实 template 保留。因此这里的 `chain_dummy`
实际是 sequence-local 粒度，不会让一个坏序列清空整个复合物的模板。该 fallback
会写入训练 telemetry，便于之后审计。只有全局 cache 根目录或两份完成 stats
不符合正式合同，launcher 才会在训练进程启动前 fail-fast。

## 测试

聚焦单元测试覆盖 identity 定义与90%边界、Top60 稳定排序与确定性
均匀排列、候选失败后的 Top60 内补齐、`N=0`、`N<20`、gzip schema
读取、显式 `raise` 策略以及 sequence-local/chain-dummy fallback：

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  -m pytest -q tests/test_sequence_template_cache.py
```
