# Strict resume 后的训练数据 ID 覆盖审计

对应脚本：

```text
scripts/runtime/audit_train_sampler_coverage.py
```

完整的脚本原理、输出格式和普通单阶段用法见：

```text
docs/train_sampler_id_coverage_audit_cn.md
```

本文只说明训练从 raw checkpoint 严格恢复或迁移到新代码后，如何继续推断模型已经使用过哪些
训练 target。

## 1. 结论

strict resume 后覆盖审计仍然有效，但必须遵守下面两条规则：

1. `--updates` 必须填写该训练采样流从最初开始计算的 **绝对 completed updates**；
2. 不要把 strict-resume 源 run 的 coverage 再通过 `--initial-coverage` 合并一次。

脚本固定从 sampler epoch 0 重放到 `--updates` 指定的完整前缀。它目前不接受
`start_update`、`sampler_epoch` 或 checkpoint cursor，因此不能把恢复后的新 run 当成一个
从 update 0 开始的独立采样阶段。

## 2. 当前恢复点

当前 v10 到 v11 的恢复源是 zero-based checkpoint `step=9999`：

| 项目 | 值 |
| --- | ---: |
| 已完成 optimizer updates | 10,000 |
| 恢复后的第一步 | `step=10000` |
| gradient accumulation | 4 |
| world size | 128 |
| epoch size | 10,240 |
| 每个 rank 每个 sampler epoch 的 target 数 | 80 |
| `consumed_global_batches` | 40,000 |
| `sampler_epoch` | 500 |
| `next_local_offset` | 0 |

计算关系为：

```text
completed_updates = zero_based_checkpoint_step + 1
consumed_global_batches = completed_updates * accumulation
sampler_epoch = consumed_global_batches // (epoch_size / world_size)
next_local_offset = consumed_global_batches % (epoch_size / world_size)
```

因此本次恢复恰好位于 sampler epoch 500 的起点。恢复后第一个 target 来自 epoch 500，
而不是重新从 epoch 0 抽样。

## 3. 最常用命令：审计截至当前 checkpoint 的完整覆盖

从代码库根目录执行：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90

RUN_DIR=/mnt/project_abd/klp/proteinx_output/REPLACE_WITH_RESUMED_RUN_DIR
CURRENT_STEP=10199
UPDATES=$((CURRENT_STEP + 1))

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/audit_train_sampler_coverage.py \
  --run-dir "${RUN_DIR}" \
  --updates "${UPDATES}" \
  --output-dir "${RUN_DIR}/sampler_coverage_step${CURRENT_STEP}"
```

上例表示训练已经完成到 `step=10199`，所以：

```text
--updates = 10199 + 1 = 10200
```

脚本会重放当前连续采样流的 update `0..10199`，其中自然包含恢复前的
`0..9999` 和恢复后的 `10000..10199`。

如果需要展开每个数据库的完整未见 ID 文件，增加：

```bash
--write-unseen-ids
```

3,000 多万条蒸馏数据的未见 ID 文件较大，通常先使用默认生成的 bitmap 和
`seen_ids.tsv.gz` 即可。

## 4. step 与 `--updates` 对照

| 当前日志/checkpoint | 恢复后新增 updates | 正确的 `--updates` |
| --- | ---: | ---: |
| 刚载入 `step=9999` | 0 | 10,000 |
| `step=10099` | 100 | 10,100 |
| `step=10199` | 200 | 10,200 |
| `step=10499` | 500 | 10,500 |
| `step=10999` | 1,000 | 11,000 |

无论训练中间重新启动多少次，始终使用：

```text
--updates = 当前 zero-based step + 1
```

不要使用“本次新 run 运行了多少步”作为 `--updates`。

## 5. 禁止的调用方式

### 5.1 不能只填恢复后的增量

下面的命令是错误的：

```bash
# 错误：当前实际已经到 step=10199，却只填写恢复后新增的 200 updates。
--updates 200
```

它会重放 sampler 的 update `0..199`，而不是真实的 `10000..10199`。

### 5.2 不能重复合并 strict-resume 源 coverage

下面的组合也是错误的：

```bash
# 错误：10200 的绝对前缀已经包含前 10000 updates，
# 又合并 source coverage 会把前 10000 updates 计算两遍。
--updates 10200 \
--initial-coverage "${SOURCE_V10_RUN}/sampler_coverage_step9999"
```

`--initial-coverage` 只会验证 row catalog 身份后累加 occurrence count；它不会判断两个
coverage 的训练区间是否重叠。

## 6. 哪些前序阶段可以通过 `--initial-coverage` 合并

只有 sampler 真正重新从 epoch 0 开始的独立训练阶段才应合并。例如当前 v4 训练最初从
v8 `step=999` 做 params-only warmstart，v8 和 v4 是两条独立采样流，因此可以把 v8
coverage 合并一次。

先生成 v8 覆盖：

```bash
OLD_V8_RUN=/mnt/project_abd/klp/proteinx_output/protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_msamix_q20_u30_p50_50k_v8_20260724_155026

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/audit_train_sampler_coverage.py \
  --run-dir "${OLD_V8_RUN}" \
  --updates 1000 \
  --output-dir "${OLD_V8_RUN}/sampler_coverage_step999"
```

再生成当前 strict-resume 连续流的绝对覆盖，并且只合并 v8：

```bash
CURRENT_RUN=/mnt/project_abd/klp/proteinx_output/REPLACE_WITH_RESUMED_RUN_DIR
CURRENT_STEP=10199

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/audit_train_sampler_coverage.py \
  --run-dir "${CURRENT_RUN}" \
  --updates $((CURRENT_STEP + 1)) \
  --initial-coverage "${OLD_V8_RUN}/sampler_coverage_step999" \
  --output-dir \
    "${CURRENT_RUN}/model_history_coverage_v8step999_plus_current_step${CURRENT_STEP}"
```

这里不要再合并 v10 `step=9999` coverage，因为当前绝对前缀已经包含该部分。

判断规则：

| 阶段关系 | 是否使用 `--initial-coverage` |
| --- | --- |
| params-only warmstart，sampler 从 0 重新开始 | 是 |
| full-state strict resume，sampler cursor 续接 | 否 |
| v10 到 v11 migration，保留 dataloader/sampler state | 否 |
| 不确定是否恢复 sampler state | 先检查 checkpoint/恢复 manifest，不能猜测 |

## 7. 如何检查结果

主要结果位于：

```text
${OUTPUT_DIR}/summary.json
```

可以快速查看：

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python -m json.tool \
  "${RUN_DIR}/sampler_coverage_step${CURRENT_STEP}/summary.json" \
  | less
```

至少检查：

- `updates` 等于 `CURRENT_STEP + 1`；
- `world_size=128`；
- `accumulation=4`；
- `epoch_size=10240`；
- `effective_sampler_seed=42`；
- `replay.total_occurrences = updates * 4 * 128`；
- `train_sets`、外层权重和各数据库 `source_identity` 与正式训练一致；
- `initial_coverages` 中不存在 strict-resume 源 run。

在恢复起点 `--updates 10000`，预期：

```text
replay.total_occurrences        = 5,120,000
replay.next_sampler_epoch       = 500
replay.next_global_epoch_offset = 0
replay.next_local_epoch_offset  = 0
```

这些值与 checkpoint 保存的 sampler cursor 一致。

## 8. 能回答与不能回答的问题

脚本能够确定：

- 每个数据库计划被抽中多少次；
- 哪些 target sampler row 被计划抽中过；
- 哪些数据 ID 至少被计划抽中过；
- 哪些 ID 截至当前仍未被计划抽中；
- 合法合并独立训练阶段后的模型历史覆盖。

脚本不能恢复：

- `random_sample_if_failed=True` 后 DataLoader worker 临时选择的替代 target；
- crop、template、MSA 最终生成的 tensor；
- 某个 target 内部具体使用了哪些 MSA row；
- worker prefetch queue 和 worker 内部随机数轨迹。

因此输出的严格名称是 **planned sampler target coverage**，不是最终成功完成 forward 的
样本流水账。

本次新增的 query-row 保留规则只改变已抽中 target 内部的 MSA 采样，不改变 dataset、
target ID、sampler seed、权重或 sampler RNG 消费，所以不影响本文的数据 ID 覆盖结论。

## 9. 只分析恢复后的新增 ID

当前脚本没有 `--start-update`，不能用一次命令直接审计区间 `10000..N`。

需要分别生成：

1. 恢复起点前缀：`--updates 10000`；
2. 当前累计前缀：`--updates CURRENT_STEP+1`；
3. 对每个数据库的 `*.seen_rows.npz` 中 `sample_count` 做当前值减起点值；
4. 差值大于 0 的 row 才是恢复后实际计划抽中的 row。

不能用“增量 updates + `--initial-coverage`”代替上述差分，因为增量 replay 仍会错误地从
sampler epoch 0 开始。

如果后续经常需要区间分析，应给审计脚本增加显式的
`--start-update/--end-update` 或 `--start-checkpoint` 接口，并把起止 cursor 和 source
checkpoint SHA-256 写入 `summary.json`。

## 10. 精确性前提

绝对前缀重放要求 strict resume 前后保持：

- sampler seed；
- world size；
- gradient accumulation；
- epoch size；
- replacement 设置；
- train dataset 顺序和外层权重；
- CSV row 顺序、row weights；
- distillation cluster/runtime-index fingerprint；
- 兼容的 CPU PyTorch sampler 随机数行为。

当前 strict-resume 实现会恢复并校验 sampler cursor 和 signature。本次 v10 到 v11 迁移
只允许 query-pin 数据契约变化，不改变 target sampler，因此满足上述条件。
