# step27799 weightedPDB MSA 双源独立预算续训合同

## 目标

本仓库是一个单变量实验副本。从已完成训练的 step `27799` raw/EMA 配对
checkpoint 恢复全部训练状态，继续到绝对终点 `50000` updates；唯一有意改变的
训练语义是 weightedPDB 的 MSA raw-source 预算：

- `pairing.a3m` 独立读取前 10,000 行；
- 真正的 `non_pairing.a3m` 独立读取前 10,000 行；
- 后续按既有规则合并、unpaired pool 去重，最终组装深度不超过 16,384；
- paired block 保持不去重；
- 不足预算时不 padding，也不把一个 source 的剩余额度转给另一个 source；
- 只在训练期启用，inference 行为不变。

这不是 trajectory-exact resume，因为训练输入合同有意变化；但 model、EMA、
optimizer、scheduler、trainer step、GradScaler、128-rank RNG 和 sampler cursor
都从同一 save event 完整恢复。

## 锁定的来源

| 项目 | 锁定值 |
| --- | --- |
| source run | `protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler_20260804_140441` |
| raw | `checkpoints/27799.pt` |
| raw SHA-256 | `4cefb9af5088a4671b4287149c95efcd72cac37449136fe37b4e4d060754f94a` |
| EMA | `checkpoints/27799_ema_0.999.pt` |
| EMA SHA-256 | `b3ad325bde97f2b249a03c4004a2a00d704e92a3aa0be925b0705e1a23ee67c8` |
| checkpoint pair ID | `d40d64151cc8a8ff17458c310ca0a53f40047d3b5d4adf5577db3d653ae1c8f2` |
| completed updates | `27800` |
| next trainer step | `27800` |
| restored scheduler step | `27800` |
| restored LR | `0.00095` |

source training contract 为：

```text
crop384_db48_ddp_disordered_v5_ws128_acc4_catalogreplacement_residual_ge200_20260804_softcluster_g02_checkpointbound_step21999_wave_schedule_v2
```

新合同只追加：

```text
_weightedpdb_pairing_nonpairing_independent10k_v1
```

严格迁移模式是：

```text
weightedpdb_independent_pairing_nonpairing_full_state_v1
```

## 保持不变的项目

本实验继续使用原 schedule：

```text
/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/step21999_residual_ge200_g02_v2
```

schedule 的 origin 是 update 22000；恢复的 sampler cursor 位于 update 27800，
因此训练直接从同一 wave-major schedule 的相应位置继续，不重放 22000–27799，
也不生成新 schedule。

以下项目明确不变：

- 四库及 outer masses：PDB/residual/ge200/disordered =
  `0.49/0.005/0.505/0.02`；
- gamma `0.2`、cluster 顺序、member 选择和 sampler RNG/cursor；
- PDB member schedule；本实验不使用
  `step27799_pdb_member_horizon_to50000_v1`；
- LR scheduler；不 re-warm，不升至 `1.3e-3`；
- optimizer、EMA `0.999`、accumulation `4`、16×8 / world size 128；
- crop 384、diffusion batch 48、模型结构、loss、template policy；
- MSA `sample_cutoff.train=2048`、`msa_chunk_size=2048`、recycle 随机 1–4；
- query-only / unpaired / 原正常 MSA 三路概率与 paired-MSA shuffle 逻辑；
- AFDB/MGnify 蒸馏单 A3M 的 16,384 source 预算。

## 16×8 AIStudio 正式启动命令

16 个 worker 执行完全相同的命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02_msaindependent_only/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_step27799_msaindependent_only_softcluster_g02_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_NNODES=16 \
bash migrate_step27799_to_weightedpdb_msa_independent.sh \
  > "step27799_msaindependent_only.node${LOG_NODE_ID}.log" 2>&1
```

不要传 `RESUME_STEP`、checkpoint 路径、schedule 路径、训练合同或
`INDEPENDENT_PAIRING_NON_PAIRING_BUDGET`；locked launcher 会拒绝这些继承值，
再自行设置正确值。

## 启动前校验与日志判定

真实启动在 `torchrun` 前执行两层校验：

1. `validate_soft_cluster_migration_request.py` 校验原 schema-v2 schedule、数据集
   binding、gamma、目标 cursor 范围，并重新计算 step27799 raw/EMA SHA-256；
2. runner 严格校验 checkpoint pair、架构、旧/新 data contract 的唯一允许差异，
   然后才加载训练状态。

launcher 的前置摘要应包含：

```text
source_step=27799 next_trainer_step=27800
source_scheduler_step=27800 source_lr=0.00095 restored_unchanged=true
cursor_continuation=true unchanged=true
PDB_member_schedule=parent_exact unchanged=true
weightedPDB_MSA=pairing.a3m_top10000+genuine_non_pairing.a3m_top10000 independent=true final<=16384
only_allowed_delta=weightedpdb_independent_pairing_nonpairing_top10k_only_v1
```

若 raw/EMA、schedule、catalog、topology 或合同不匹配，任务应在第一步训练前
fail closed。生产启动前可先做不读取大 checkpoint 的命令展开检查：

```bash
DRY_RUN=1 SKIP_PATH_CHECK=1 \
MLP_WORKER_NUM=16 MLP_ROLE_INDEX=0 MLP_WORKER_0_HOST=worker-0 \
RUN_NAME=protenix_v4_step27799_msaindependent_only_softcluster_g02_v9_straggler_dryrun \
BASE_DIR=/tmp/protenix_msaindependent_dryrun \
bash migrate_step27799_to_weightedpdb_msa_independent.sh
```

`SKIP_PATH_CHECK=1` 只允许和 `DRY_RUN=1` 同时使用，真实训练无法借此绕过校验。

## 回退

本迁移不修改源 checkpoint。若实验结果不理想，切回父仓库并再次从同一
`27799.pt` 与匹配 EMA 启动即可。不要把本实验生成的新 checkpoint 当成旧数据
合同的 trajectory-exact continuation。

截至本文档写入时，仅完成代码、校验和 dry-run 测试，没有启动生产训练。
