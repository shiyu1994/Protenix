# step 21999 新聚类库迁移与完整恢复合同（已正式验收）

## 状态

这是独立于 step 21399 历史入口的新迁移边界。截至 2026-08-04，checkpoint、
target config、schema-v2 schedule 和正式 validator 均已验收：

```text
正式 residual runtime catalog   已发布，可复用
正式 ge200 runtime catalog      已发布，可复用
step21999 target config         已发布并验收
21999.pt                        已完成稳定哈希与完整状态验收
21999_ema_0.999.pt              已完成稳定哈希与配对验收
step21999 schema-v2 schedule    已原子发布并验收
正式 validator                  退出 0
训练                            未启动
```

本次验收只准备恢复边界和未来采样，不启动训练。真实启动仍会由 launcher 和 runner
重复验证同一份 raw/EMA/schedule 绑定；任何文件漂移都会 fail closed。

## 固定边界

```text
source checkpoint step        21999
completed optimizer updates   22000
next global microstep         88000
world size                    128
gradient accumulation         4
epoch size                    10240
sampler seed                  42
absolute target updates       50000
schedule gamma                0.2
```

因此正式 schedule 的确定性尺寸应为：

```text
origin consumed batches/rank  22000 × 4 = 88000
target consumed batches/rank  50000 × 4 = 200000
future waves                  200000 - 88000 = 112000
global_rows.u32 rows          112000 × 128 = 14,336,000
global_rows.u32 size          14,336,000 × 4 = 57,344,000 bytes
```

任何正式 schedule 若不满足这组 cursor、行数和字节数关系，都必须拒绝。

source run 固定为：

```text
/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999_v9_straggler_20260730_104039
```

新 schedule 只能发布到：

```text
/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/step21999_residual_ge200_g02_v2
```

四个 child、顺序和 raw mass 继续固定为：

```text
weightedPDB_before210930_v20260101                         0.49
afdb_mgnify_legacy_residual_distillation_v20260804         0.005
all_sources_ge200_structural_distillation_v20260804        0.505
openfold3_disordered_v20260725                             0.02
```

正式 runtime catalogs 原样复用，不重新构建：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804
/mnt/project_abd_hs/klp/databases/all_sources_ge200_structural_runtime_index_20260804
```

本边界专用 target config 已原子发布到：

```text
/mnt/project_abd/klp/proteinx_output/catalog_replacement_configs/step21999_residual_ge200_v1
target_config.yaml SHA-256  0c66a97ef7a69672d43443b1f4b84766b3391ebc25695ba23bbf50d83167ba4f
receipt.json SHA-256       a3327f2e416cf5628662fc45446a4859b7c199804d282c71ecef3f751f61d262
_SUCCESS.json SHA-256      4dcdf418f6534ea16a0c3642a6c564d367eab0707d8e20d7a4c4edcb9bd3851c
```

其 receipt 固定记录 `checkpoint_step=21999`、`next_optimizer_update=22000`；
target YAML 与已验收的 step21399 target YAML 字节一致，但两份边界 receipt 保持独立。

## 已执行的 schedule 构建

下列命令已经成功执行。它使用上面已发布的 step21999 专用 target config；schedule
重新绑定了 step 21999 的 source config、raw、EMA、128-rank cursor 和历史 exposure。

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02/

SOURCE_RUN_DIR=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999_v9_straggler_20260730_104039
TARGET_CONFIG=/mnt/project_abd/klp/proteinx_output/catalog_replacement_configs/step21999_residual_ge200_v1/target_config.yaml
SCHEDULE_DIR=/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/step21999_residual_ge200_g02_v2

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/prepare_soft_cluster_schedule.py \
  --source-run-dir "${SOURCE_RUN_DIR}" \
  --source-checkpoint-step 21999 \
  --target-config "${TARGET_CONFIG}" \
  --target-updates 50000 \
  --gamma 0.2 \
  --output-dir "${SCHEDULE_DIR}"
```

构建于 `2026-08-04T13:31:29Z` 完成，构建器内部用时 318.62 秒、命令总用时
328 秒。构建日志为：

```text
/mnt/project_abd/klp/proteinx_output/catalog_build_logs/step21999_residual_ge200_g02_schedule_20260804.log
SHA-256  7e99385f0ead33e4e64fba17fe1d6b4ca9757721457e2875c531e534d461bd70
```

正式 schedule 收据：

```text
schema/version                    protenix.soft_cluster_schedule.v2 / 2
status                            complete
manifest.json SHA-256             3b272882b12d2d7e35e37df1a7ef2524edab7fd8a9ab04d76d77f2a9ec596dea
global_rows.u32 SHA-256           b6571b197471e38be0d1a4ac580663c666dd680bf4acfdb0a051553c538e758f
global_rows.u32 rows / bytes      14,336,000 / 57,344,000
artifact_set SHA-256              1f4e9d15c71bfa0ecee9ab247f5b2274d2149c134028b33642ce503d49d12f24
_SUCCESS.json SHA-256             8a46247ebd3785844ab35283202301a5ab64d355e6543f81e08d34ab5009a21e
catalog_migration.audit SHA-256   b8e977045ed81ecea1aaf54c40fbacbe46c7607401db4db2a977b8acad7a9175
```

构建器从真正的 raw/EMA 文件读取完整 checkpoint header，并原子发布 schema-v2
schedule；文件缺失、pair 不匹配或输出目录已存在都会失败。

## 正式哈希验收（已通过）

schedule 发布后已在 CPU 节点执行以下命令，退出码为 0：

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/validate_soft_cluster_migration_request.py \
  --schedule "${SCHEDULE_DIR}" \
  --source-run-dir "${SOURCE_RUN_DIR}" \
  --source-step 21999 \
  --source-raw-checkpoint "${SOURCE_RUN_DIR}/checkpoints/21999.pt" \
  --source-ema-checkpoint "${SOURCE_RUN_DIR}/checkpoints/21999_ema_0.999.pt" \
  --target-updates 50000 \
  --migration-mode catalog-replacement-v2
```

validator 稳定读取并重新哈希 raw/EMA，确认读取期间文件未被替换，再与 schedule
绑定值比较。正式结果为：

```text
checkpoint step / completed updates   21999 / 22000
next global microstep                  88000
raw SHA-256                            3ce863810a109dd15f5cb6669f9882d05a6d3f3b2d9e14660b4b81f3711200a0
EMA SHA-256                            fc222762f08ca3d100da21f6f813d6d5f8c6df0bb012ea14189cfc7f4b885b9a
checkpoint pair ID                    940ee066209082aee50c34464f51d955215f43e48629c9a537aa1253f73a77a8
checkpoint binding SHA-256            eacc8d1323f3a0a2f08ea52ebab0c292334e0d48bcc07505cf21df4a31de2194
source dataset binding SHA-256        762d15afc8c6d5bdbe56f69f10ddd97b4ace6d555aaabeafc23f0c411fa97f77
target dataset binding SHA-256        faee230baca09fc06cef26befdd5b46bddad4a7f67e9e8fed8c2a3a59a2275c6
runtime catalog pair ID               8f79ea1460a2f60be5609278556df34cda95c2c45c9f969e1125f62f0326c121
```

128 个 rank 的 raw checkpoint 均记录 `consumed_global_batches=88000`、
`sampler_epoch=1100`，且 model、optimizer、scheduler、GradScaler、Python/NumPy/
torch/CUDA RNG 和 sampler state 全部存在。validator 日志为：

```text
/mnt/project_abd/klp/proteinx_output/catalog_build_logs/step21999_residual_ge200_g02_validator_20260804.log
SHA-256  cef634566cfda21e7a7bbdd3ef1258f913af448a363b2b7f79c4170b6d77ca26
```

历史映射也闭合：5,632,767 次 source occurrence 全部映射，`unresolved=0`、
`intentional_missing=0`。142 个已记录歧义 ID 按 `seen_floor_1_v1` 处理，其中 48 个
alternate cluster 获得 floor，因此 target initial exposure 总和比 occurrence 总数多
2；这是显式策略结果，不是重复计数错误。

## 16×8 AIStudio 入口

只有上节 validator 退出 0 后，16 个节点才执行同一条命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash migrate_step21999_to_residual_ge200_softcluster_g02.sh \
  > "step21999_residual_ge200_softcluster_g02.node${LOG_NODE_ID}.log" 2>&1
```

无需传 `SOFT_CLUSTER_SCHEDULE_DIR`：新 launcher 将其锁定为本文的新目录，并拒绝
21399 schedule、其他 checkpoint、非正式 runtime catalogs、非 16×8 拓扑及任何
继承的内部恢复合同变量。`SKIP_PATH_CHECK=1` 只允许配合 `DRY_RUN=1`，不能用于真实
训练。真实启动时 launcher 会再次执行 validator；进入 runner 后，global rank 0
还会在反序列化前按 schedule 再次核对 raw/EMA SHA，只有全部验证成功才应用状态。
原
`migrate_step21399_to_residual_ge200_softcluster_g02.sh` 保持独立且语义不变。
