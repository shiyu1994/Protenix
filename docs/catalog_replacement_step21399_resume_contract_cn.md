# step 21399 新聚类库迁移与完整恢复合同

## 目标集合

本阶段把旧 Short/Long 两个 child 替换为互斥的两份新资产：

```text
legacy residual = 旧 AFDB/MGnify 全集 - 新 ge200 中的 AFQ/MGN 条目
target distillation = legacy residual ∪ all-source ge200 structural
```

四个 child 的顺序和 raw mass 固定为：

```text
weightedPDB_before210930_v20260101                         0.49
afdb_mgnify_legacy_residual_distillation_v20260804         0.005
all_sources_ge200_structural_distillation_v20260804        0.505
openfold3_disordered_v20260725                             0.02
```

因此 `<200`、未进入新库的 `=200` 以及其他旧库 residual 条目仍被保留；进入
ge200 的 AFQ/MGN 不会同时留在 residual 中。AFU/ESM 只存在于 ge200。

## 锁定的恢复边界

正式 migration manifest 必须绑定：

```text
source checkpoint step        21399
completed optimizer updates   21400
next global microstep         85600
world size                    128
gradient accumulation         4
epoch size                    10240
sampler seed                  42
absolute target updates       50000
```

checkpoint 身份：

```text
raw SHA-256   9250ec1f5de4846904580af8614a324932821e3eadbb58b093075aa4eddf5d4a
EMA SHA-256   74a9e24f3e377b278e0db214e3986676b43edafc5b39f4e1a5b5eb2c9c65288c
pair id       dadd29c0819c0911a43c9b7490a3c60679d073dc21b4ad4cb33117c0323d29c4
data hash     6c98c5cab75197b7e4d41b7ce5287af65ca3f750a707ec69a1bdaf29ed24b8de
model hash    cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c
```

构造 target runtime index 不依赖 checkpoint 文件是否已写完，可以提前进行。历史
exposure 投影和最终 schedule 则必须在 checkpoint 落盘后读取并绑定上述 checkpoint
身份。现在 step 21399 raw/EMA 已经落盘并核验。

## 离线生成 target config 与 schedule

不需要等待旧训练结束。迁移边界固定为 `21399` 后，旧任务在这个边界之后产生的
update 不属于新 lineage；新训练从 update `21400` 开始。先用 source run 当时实际
落盘的 `config.yaml` 生成 target config：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02/

TARGET_CONFIG_DIR=/mnt/project_abd/klp/proteinx_output/catalog_replacement_configs/step21399_residual_ge200_v1

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/prepare_catalog_replacement_target_config.py \
  --output-dir "${TARGET_CONFIG_DIR}"
```

正式命令故意不传 runtime path：工具与 launcher 使用相同的两个默认目录：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804
/mnt/project_abd_hs/klp/databases/all_sources_ge200_structural_runtime_index_20260804
```

工具先验证两边的 `manifest.json`、`_SUCCESS.json`、`roots.json` 和
`target_catalog_pair.json`，要求两份 pair receipt 字节级一致，然后在同目录文件
系统上原子发布：

```text
target_config.yaml
receipt.json
_SUCCESS.json
```

它只允许把旧 Short/Long 两个 child 原位替换为 residual/ge200，并只更新新 child
的 `runtime_index_dir`、`source_roots_fpath`、`mmcif_dir` 和
`bioassembly_dict_dir`。以下值任何一个漂移都会直接失败，不会自动修改：

```text
weights       0.49 / 0.005 / 0.505 / 0.02
epoch_size    10240
seed          42
accumulation  4
max_steps     50000
```

两个 target runtime 尚未完成原子发布时，这条命令失败是预期行为；历史回放代码可
并行准备，但不得用半成品发布 schedule。target config 完成后构建 schema-v2
schedule：

```bash
SOURCE_RUN_DIR=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999_v9_straggler_20260730_104039
SCHEDULE_DIR=/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/step21399_residual_ge200_g02_v2

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/prepare_soft_cluster_schedule.py \
  --source-run-dir "${SOURCE_RUN_DIR}" \
  --source-checkpoint-step 21399 \
  --target-config "${TARGET_CONFIG_DIR}/target_config.yaml" \
  --target-updates 50000 \
  --gamma 0.2 \
  --output-dir "${SCHEDULE_DIR}"
```

两个离线工具都不加载训练模型到 GPU，也不会启动训练。

## 发布后的分层验收

实际执行顺序固定为：catalog 验收 → target config → schema-v2 schedule → migration
validator → 正式 launcher；本节把所有验收命令集中列出。

catalog builder 退出码为 0 且两个正式目录都出现 `_SUCCESS.json` 后，先运行轻量
验收。它核对精确 counts、两边字节级一致的 pair receipt/build audit、checksum
清单完整性、runtime index 可加载性、首尾 cluster 覆盖以及每个 root 的有界
CIF/A3M/PKL 路径抽样；它不会扫描原始 2,541 万行 TSV：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02/

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/validate_catalog_replacement_runtime_indices.py
```

正式发布只需额外做一次全量生成物 checksum 验收。该选项会顺序读取两个 runtime
目录中的生成物，I/O 会更久，但仍不会读取原始 mapping/numeric TSV：

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/validate_catalog_replacement_runtime_indices.py \
  --verify-checksums
```

生产 counts 必须同时满足：

```text
legacy input rows       33,498,651
ge200 AFQ/MGN overlap   18,175,512
legacy residual rows    15,323,139
  short residual        15,313,927
  long residual              9,212
ge200 rows              25,417,250
ge200 clusters           9,050,355
target union rows       40,740,389
ambiguity alternates            142
```

step 21399 raw/EMA 已核验过；若要在正式迁移前独立重做一次大文件校验，可在任一
单节点执行（不需要在 16 个节点重复）：

```bash
SOURCE_RUN_DIR=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999_v9_straggler_20260730_104039

cd "${SOURCE_RUN_DIR}/checkpoints"
sha256sum -c - <<'EOF'
9250ec1f5de4846904580af8614a324932821e3eadbb58b093075aa4eddf5d4a  21399.pt
74a9e24f3e377b278e0db214e3986676b43edafc5b39f4e1a5b5eb2c9c65288c  21399_ema_0.999.pt
EOF
```

schema-v2 schedule 发布后、启动 torchrun 以前，显式执行恢复合同 validator：

```bash
SOURCE_RUN_DIR=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999_v9_straggler_20260730_104039
SCHEDULE_DIR=/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/step21399_residual_ge200_g02_v2

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/validate_soft_cluster_migration_request.py \
  --schedule "${SCHEDULE_DIR}" \
  --source-run-dir "${SOURCE_RUN_DIR}" \
  --source-step 21399 \
  --source-raw-checkpoint "${SOURCE_RUN_DIR}/checkpoints/21399.pt" \
  --source-ema-checkpoint "${SOURCE_RUN_DIR}/checkpoints/21399_ema_0.999.pt" \
  --target-updates 50000 \
  --migration-mode catalog-replacement-v2
```

只有以上命令全部退出 0 才把 schedule 交给正式 launcher。validator 只读取并验证
资产，不启动训练。

## 唯一允许的训练合同变化

schema-v2 schedule 的 `catalog_migration` 必须声明：

```text
format          protenix.catalog_replacement.v1
allowed_delta   legacy_short_long_to_legacy_residual_ge200_v1
global id       source_prefix_pipe_raw_id_v1
exposure        occurrence_sum_v1
ambiguous       seen_floor_1_v1
gamma           0.2
```

runner 分别验证：

- raw/EMA 文件 SHA、pair id、step 和 cursor；
- source checkpoint 中的 model、optimizer、scheduler、GradScaler 和 step；
- 128 个 rank 各自的 Python、NumPy、torch CPU、CUDA RNG 和 dataloader cursor；
- source sampler signature 对应旧 Short/Long binding；
- live target sampler signature 对应 residual/ge200 binding；
- PDB/disordered child 指纹完全不变；
- outer masses、MSA/crop/template/loss 和其他非 catalog 策略不变；
- source/target runtime manifest、crosswalk、ambiguity audit 和 schedule 全部受 SHA-256
  绑定。

任意检查失败都会在应用 optimizer/model 状态前终止。这里是完整状态恢复，不是
params-only warm start；恢复后第一步是 update 21400。

## 正式入口

runtime index 和 schema-v2 schedule 完整发布后，16 个 AIStudio 节点使用同一命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_step21399_residual_ge200_softcluster_g02_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
SOFT_CLUSTER_SCHEDULE_DIR=/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/step21399_residual_ge200_g02_v2 \
bash migrate_step21399_to_residual_ge200_softcluster_g02.sh \
  > "residual_ge200_softcluster_g02.node${LOG_NODE_ID}.log" 2>&1
```

launcher 默认使用正式 residual/ge200 runtime-index 路径；若发布目录不同，可显式设置
`PROTENIX_LEGACY_RESIDUAL_RUNTIME_INDEX_DIR` 和
`PROTENIX_GE200_STRUCTURAL_RUNTIME_INDEX_DIR`。live runtime fingerprints 仍必须与
schedule manifest 完全一致。

## 2026-08-04 正式产物与验收结果

本次已固定 step `21399` 为切换边界，不等待旧 lineage 继续训练。两份 runtime
catalog、target config 和 schema-v2 schedule 均已原子发布；尚未启动新训练。

```text
runtime pair id       8f79ea1460a2f60be5609278556df34cda95c2c45c9f969e1125f62f0326c121
pair receipt SHA      158965a33bf77d80164ba92f3b6ebcb818e78cfcb665964f1e77493f80b7c4dc
target config SHA     0c66a97ef7a69672d43443b1f4b84766b3391ebc25695ba23bbf50d83167ba4f
schedule manifest SHA 5c4177d2c557b04db50ea3ed17b162b1f08ed501e2855676b8408fac283bd30d
global_rows SHA       932a8fb70daa9194328fa2adde77557b797b8bc294015a8b5698df61dac1c369
artifact set SHA      c25bee85ec36db1b1e188b16d5e29803b4d594760435b078f7b3a69cb4901d97
```

正式目录：

```text
/mnt/project_abd_hs/klp/databases/afdb_mgnify_legacy_residual_not_ge200_runtime_index_20260804
/mnt/project_abd_hs/klp/databases/all_sources_ge200_structural_runtime_index_20260804
/mnt/project_abd/klp/proteinx_output/catalog_replacement_configs/step21399_residual_ge200_v1
/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/step21399_residual_ge200_g02_v2
```

正式验收结果：

```text
runtime checksum entries                39/39 verified
history sampled occurrences             5,478,900
history unique global IDs               4,694,806
history unresolved / intentional missing 0 / 0
ambiguity records / seen alternates       142 / 48
future schedule rows                   14,643,200
future schedule waves                     114,400
future residual / ge200 draws          71,373 / 7,252,046
migration validator                      exit 0
```

离线生成与验证日志位于：

```text
/mnt/project_abd/klp/proteinx_output/catalog_build_logs/residual_ge200_runtime_indices_20260804.log
/mnt/project_abd/klp/proteinx_output/catalog_build_logs/residual_ge200_runtime_indices_checksum_validation_20260804.log
/mnt/project_abd/klp/proteinx_output/catalog_build_logs/step21399_residual_ge200_target_config_20260804.log
/mnt/project_abd/klp/proteinx_output/catalog_build_logs/step21399_residual_ge200_g02_schedule_20260804.log
/mnt/project_abd/klp/proteinx_output/catalog_build_logs/step21399_residual_ge200_g02_validator_20260804.log
```
