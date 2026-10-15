# 训练 sampler 数据 ID 覆盖审计：使用说明与效果

对应实现：

```text
scripts/runtime/audit_train_sampler_coverage.py
```

如果训练是从 full-state checkpoint 严格恢复或迁移到新代码，先阅读：

```text
docs/train_sampler_id_coverage_strict_resume_usage_cn.md
```

## 1. 目标

脚本根据正式训练保存的配置和不可变数据索引，确定性重放训练 sampler。它回答：

- 每个数据库被抽中了多少次；
- 哪些 sampler row 至少被抽中过一次；
- 哪些数据 ID 至少被抽中过一次；
- 哪些 row / 数据 ID 尚未被抽中过；
- 多阶段训练后，当前模型在整个训练历史中见过哪些 ID。

脚本只读取索引、weights 和 compact runtime index，不读取 CIF、bioassembly、
MSA 或 template，不使用 GPU，也不影响正在运行的训练。

结果严格定义为 **sampler 计划覆盖**。如果
`random_sample_if_failed=True` 的数据发生坏 row 替换，seed 能恢复原始计划 row，
但不能恢复 DataLoader worker 后来随机挑选的替代 row。当前正式训练截至
step 2099 只有 12 次这类替换，不影响总体覆盖分析。

## 2. 实现原理

脚本不是根据采样权重估算覆盖率，而是直接复用正式训练的 sampler 实现，逐 epoch
重放完整抽样序列。

当前正式任务的参数为：

| 参数 | 值 |
| --- | ---: |
| global seed | 42 |
| world size | 128 |
| gradient accumulation | 4 |
| 每个 update 的 target 数 | 512 |
| epoch size | 10,240 |
| 每个 sampler epoch 的 update 数 | 20 |

对于 zero-based optimizer update `u`、累积槽 `a` 和 rank `r`：

```text
global_microstep = 4 * u + a
sampler_epoch = global_microstep // 80
local_epoch_offset = global_microstep % 80
global_position_in_epoch = 128 * local_epoch_offset + r
```

每个 sampler epoch 使用 CPU `torch.Generator(seed=42+sampler_epoch)`：

1. 根据四个数据库的外层 mass 抽 dataset；
2. 按正式 child 顺序继续消费同一个 generator；
3. weighted child 使用正式 row weights；
4. clustered distillation child 先均匀抽 cluster，再均匀抽 cluster member；
5. 将抽中的 row 计入对应数据库 coverage。

必须重放完整 epoch，不能只根据一个 target 的位置单独调用一次随机数。dataset
抽样和 child 抽样共用同一个 generator，任何一次随机数消费顺序变化都会改变后续
ID。

## 3. 运行要求

- 从本代码库根目录运行；
- 使用正式 Protenix Python 环境；
- run 目录中存在训练保存的 `config.yaml`；
- 默认从 `manifest/run_manifest.json` 读取 world size；
- 数据索引、row 顺序、weights 和 runtime index 与训练时一致；
- 输出目录尚不存在，脚本不会覆盖已有审计结果。

推荐解释器：

```text
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
```

## 4. 参数说明

| 参数 | 是否必需 | 说明 |
| --- | --- | --- |
| `--run-dir` | 是 | 需要审计的训练 run 目录 |
| `--updates` | 是 | 已完成的 optimizer update 数量 |
| `--output-dir` | 是 | 新建的审计结果目录 |
| `--world-size` | 否 | 覆盖 manifest 中的 world size，通常不应使用 |
| `--initial-coverage` | 否 | 合并前序训练阶段；可以重复传入 |
| `--write-unseen-ids` | 否 | 展开全部未见 ID 文本文件 |
| `--log-level` | 否 | `DEBUG/INFO/WARNING/ERROR`，默认 `INFO` |

特别注意 `--updates` 不是 zero-based step：

| 训练日志/checkpoint | 脚本参数 |
| --- | ---: |
| `step=199` | `--updates 200` |
| `step=999` | `--updates 1000` |
| `step=1999` | `--updates 2000` |
| `step=2099` | `--updates 2100` |

关系为：

```text
updates = zero_based_checkpoint_step + 1
```

## 5. 单阶段使用

下面重放当前 run 从自身 step 0 到 step 2099 的 sampler：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90

RUN_DIR=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/audit_train_sampler_coverage.py \
  --run-dir "${RUN_DIR}" \
  --updates 2100 \
  --output-dir "${RUN_DIR}/sampler_coverage_step2099"
```

该结果只代表当前 run，不自动包含起始 checkpoint 以前的训练数据。

## 6. 合并模型的前序训练阶段

当前 v4 run 从 v8 的 `step999` 权重启动。要判断当前模型在整个历史中是否见过某个
ID，必须先重放 v8 的 1000 updates，再与当前阶段取并集。

```bash
OLD_RUN=/mnt/project_abd/klp/proteinx_output/protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_msamix_q20_u30_p50_50k_v8_20260724_155026
CURRENT_RUN=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/audit_train_sampler_coverage.py \
  --run-dir "${OLD_RUN}" \
  --updates 1000 \
  --output-dir "${OLD_RUN}/sampler_coverage_step999"

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/audit_train_sampler_coverage.py \
  --run-dir "${CURRENT_RUN}" \
  --updates 2100 \
  --initial-coverage "${OLD_RUN}/sampler_coverage_step999" \
  --output-dir \
    "${CURRENT_RUN}/model_history_coverage_v8step999_plus_current_step2099"
```

同一模型有更多前序阶段时，可以重复传入：

```bash
--initial-coverage /path/to/stage1_coverage \
--initial-coverage /path/to/stage2_coverage
```

当前 run 的数据配置必须包含前序 coverage 中的所有数据集。合并前会逐库校验：

- catalog row 数；
- CSV 源索引 SHA-256；
- 数据过滤配置；
- runtime-index sampling fingerprint。

只要其中一项不同，合并就会终止，防止把不同 row 顺序的 bitmap 错误取并集。

## 7. 输出文件

每个数据集都会生成：

- `*.seen_bitmap.bin`：每个 sampler row 一位，精确表示 seen/unseen；
- `*.seen_rows.npz`：抽中过的 row ID 和每个 row 的抽中次数；
- `*.seen_ids.tsv.gz`：抽中过的数据 ID；
- PDB/disorder 额外生成 `*.seen_rows.tsv.gz`，包含 chain/interface row；
- `summary.json`：每个库的 occurrence、seen、unseen 和覆盖率。

默认不展开 3,300 多万条蒸馏数据的完整 unseen 文本清单。bitmap 已经精确表示每个
row 是否见过；对后续候选 ID 集合，使用 `seen_ids.tsv.gz` 做集合查询更高效。

### 7.1 `summary.json`

主要字段：

| 字段 | 含义 |
| --- | --- |
| `updates` | 当前阶段重放的 optimizer update 数 |
| `coverage_total_occurrences` | 当前阶段加所有前序阶段的 occurrence 总数 |
| `initial_coverages` | 已合并的前序 coverage 及其 SHA-256 |
| `datasets.*.catalog_rows` | 该数据库可采样的 row 总数 |
| `datasets.*.sampled_occurrences` | 累计抽样次数，包含重复抽样 |
| `datasets.*.unique_seen_rows` | 至少被抽中过一次的 row 数 |
| `datasets.*.unseen_rows` | 从未抽中过的 row 数 |
| `datasets.*.unique_seen_data_ids` | 至少有一个 row 被抽中的数据 ID 数 |
| `datasets.*.unseen_data_ids` | 尚未抽中的数据 ID 数 |
| `datasets.*.source_identity` | 用于证明 row catalog 未变化的身份信息 |

快速查看：

```bash
jq '{
  updates,
  coverage_total_occurrences,
  datasets: (
    .datasets
    | with_entries(
        .value |= {
          sampled_occurrences,
          unique_seen_rows,
          unseen_rows,
          unique_seen_data_ids,
          unseen_data_ids
        }
      )
  )
}' "${COVERAGE_DIR}/summary.json"
```

### 7.2 `*.seen_ids.tsv.gz`

weightedPDB 和 disordered 按 PDB ID 聚合：

```text
data_id    sample_count
```

AFDB/MGnify distillation 保留稳定 row 地址：

```text
data_id    sampler_row_id    sample_count    root_id    cif_relpath
```

查看前几条：

```bash
gzip -cd \
  "${COVERAGE_DIR}/afdb_mgnify_long_distillation_v20260725.seen_ids.tsv.gz" \
  | head
```

### 7.3 `*.seen_rows.npz`

该文件只保存实际抽中过的 row，因此比完整长度的计数数组小：

```python
import numpy as np

with np.load("weightedPDB_before210930_v20260101.seen_rows.npz") as data:
    row_ids = data["row_id"]
    sample_counts = data["sample_count"]
```

`row_ids[i]` 被 sampler 抽中了 `sample_counts[i]` 次。

### 7.4 `*.seen_bitmap.bin`

bitmap 中每个 catalog row 占一位，位顺序是 `little`：

```python
import json
import numpy as np

with open("summary.json") as f:
    summary = json.load(f)

name = "afdb_mgnify_long_distillation_v20260725"
row_count = summary["datasets"][name]["catalog_rows"]
packed = np.fromfile(f"{name}.seen_bitmap.bin", dtype=np.uint8)
seen = np.unpackbits(packed, bitorder="little")[:row_count].astype(bool)

print("seen rows:", int(seen.sum()))
print("unseen rows:", int((~seen).sum()))
```

## 8. 判断一批候选 ID 是否见过

拿到候选 ID 文件后，不需要生成完整蒸馏库 unseen 清单。直接与 seen 集合比较：

```bash
gzip -cd \
  "${COVERAGE_DIR}/afdb_mgnify_long_distillation_v20260725.seen_ids.tsv.gz" \
  | awk 'NR > 1 {print $1}' \
  | sort -u > /tmp/long_seen_ids.txt

sort -u candidate_ids.txt > /tmp/candidate_ids.sorted.txt

comm -23 \
  /tmp/candidate_ids.sorted.txt \
  /tmp/long_seen_ids.txt \
  > candidate_ids.unseen.txt
```

`candidate_ids.unseen.txt` 就是候选集合中 sampler 从未抽中过的 ID。

确实需要所有未见 ID 的文本时增加：

```bash
--write-unseen-ids
```

它会生成 `*.unseen_ids.tsv.gz`。蒸馏库清单很大，地址解析和 gzip 输出需要数分钟。

## 9. 当前 step 2099 的实际效果

当前阶段独立重放：

```text
2100 updates × 4 accumulation × 128 ranks = 1,075,200 occurrences
```

单阶段结果：

| 数据集 | occurrence | seen row | unseen row |
| --- | ---: | ---: | ---: |
| weightedPDB | 516,046 | 230,531 | 810,176 |
| short distillation | 5,329 | 5,329 | 15,420,258 |
| long distillation | 532,747 | 524,282 | 17,548,782 |
| disordered | 21,078 | 15,906 | 286,334 |

合并 v8 step999 后，模型历史共覆盖 1,587,200 个 occurrence：

| 数据集 | occurrence | 已见 ID | 未见 ID |
| --- | ---: | ---: | ---: |
| weightedPDB | 1,028,046 | 125,047 | 39,784 |
| short distillation | 5,329 | 5,329 | 15,420,258 |
| long distillation | 532,747 | 524,282 | 17,548,782 |
| disordered | 21,078 | 11,277 | 17,290 |

说明：

- weightedPDB/disordered 的 ID 口径是 PDB ID；
- distillation 的 ID 口径是 runtime catalog 中的独立预测结构；
- weightedPDB/disordered 同时保留更严格的 sampler-row 覆盖；
- occurrence 包含重复抽样，不能与 unique seen ID 混用。

当前报告位置：

```text
/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/model_history_coverage_v8step999_plus_current_step2099
```

## 10. 正确性验证

对当前 step 2099 结果进行了两层验证：

1. 四库 occurrence 与 84 个训练日志窗口的全量聚合逐项完全一致；
2. 对 10,752 条已持久化的周期样本 ID 逐条核对，10,751 条完全一致。

唯一不一致项：

```text
weightedPDB planned row: 6v8o
training actual replacement: 6jml
```

`6v8o` 在数据处理阶段失败，训练代码通过 `random_sample_if_failed=True`
随机选择了替代样本。这不是 sampler 重放错误，而是脚本明确不恢复的 worker
fallback。

## 11. 性能和存储

当前 2100 updates 的实测：

| 项目 | 结果 |
| --- | ---: |
| 运行时间 | 约 40 秒 |
| GPU 使用 | 0 |
| 结构/MSA/template 读取 | 0 |
| 默认单阶段输出 | 约 18 MiB |
| 合并 v8 后输出 | 约 19 MiB |

计算量主要来自：

- 加载和过滤 weightedPDB/disordered CSV index；
- 重建正式 row weights；
- 按 epoch 调用 CPU PyTorch sampler；
- 写 seen ID 和紧凑 bitmap。

运行时间随 sampler epoch 数量近似线性增长。50,000 updates 重放预计为分钟级，
仍远小于真实训练时间。使用 `--write-unseen-ids` 时，3,300 多万蒸馏 ID 的地址
解析和 gzip 输出会成为主要开销。

## 12. 精确性边界

脚本能够精确恢复：

- sampler 计划选择的 dataset；
- sampler 计划选择的 catalog row；
- 每个 row 的计划抽样次数；
- row/data-ID 的 seen/unseen；
- sampler epoch 和 global position。

脚本不恢复：

- 坏 row 的 DataLoader worker 随机替代 ID；
- crop 后的完整 token 集合；
- worker RNG 控制的部分 MSA 抽样；
- template/MSA/CIF 最终处理得到的完整 tensor；
- 训练时预取队列内部尚未 yield 的任务。

严格重放要求：

- 使用与训练相同的 CPU PyTorch sampler 行为；
- seed、world size、accumulation 和 epoch size 不变；
- dataset 顺序、外层权重、child sampler 类型不变；
- CSV row 顺序、weights 和 cluster arrays 不变。

脚本将 PyTorch 版本和源索引身份写进 `summary.json`，便于结果追溯。

## 13. 常见问题

### 13.1 输出目录已经存在

脚本会拒绝覆盖：

```text
FileExistsError: Refusing to overwrite existing output directory
```

请使用新的目录名，或在确认不需要旧结果后人工处理旧目录。脚本不会自动删除。

### 13.2 `--updates` 应填 checkpoint step 还是下一步

填写已完成 update 数：

```text
updates = zero_based_checkpoint_step + 1
```

### 13.3 合并时报 source identity 不一致

这表示两个阶段的同名数据集 row catalog 不完全相同，不能直接按 row ID 合并。
不要绕过校验；应分别保留两个 coverage，或建立经过证明的 row-ID 映射。

### 13.4 run manifest 不存在

正式 run 通常有 `manifest/run_manifest.json`。旧 run 缺少 manifest 时，可以显式
传入训练时的 world size：

```bash
--world-size 128
```

world size 填错会改变 rank-strided sampler 的已消费位置，审计结果不可信。

## 14. 测试

聚焦测试：

```bash
PROTENIX_MP_SHARING_STRATEGY=file_descriptor \
  /mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  -m pytest -q tests/test_audit_train_sampler_coverage.py
```

当前结果：

```text
4 passed
```
