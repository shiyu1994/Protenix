# step47199 同合同 full-state resume

## 结论

2026-08-13 中断的训练最后写到 metric step `47324`，但最后一对完整、可恢复的
checkpoint 是 `47199.pt` 和 `47199_ema_0.999.pt`。本仓库新增：

```text
resume_step47199_msaindependent_only.sh
```

它从该 checkpoint pair 恢复到下一 update `47200`，继续到绝对终点 `50000`。
中断后已经执行但未保存的 `47200–47324` 会按 checkpoint 语义回放；这比从
`27799` 重放约 19,400 个 updates 正确得多。

这次是同一训练合同的 continuation，不会再次执行
`weightedpdb_independent_pairing_nonpairing_full_state_v1` 迁移。旧入口
`migrate_step27799_to_weightedpdb_msa_independent.sh` 仍用于历史审计，不能用于本次
恢复。

## 已审计的 checkpoint

| 项目 | 值 |
| --- | --- |
| source run | `protenix_v4_step27799_msaindependent_only_softcluster_g02_v9_straggler_20260807_053504` |
| raw checkpoint | `checkpoints/47199.pt` |
| raw bytes | `6,494,003,469` |
| raw SHA-256 | `858ef830befb2e812b3eef1f7ec0f4a8e7db962e18a894975f95af87f9c8ce6f` |
| EMA checkpoint | `checkpoints/47199_ema_0.999.pt` |
| EMA bytes | `2,164,205,909` |
| EMA SHA-256 | `1a29bf7b121dab71533d783c761bf850db76431605fc8424a01dc16156ed8730` |
| pair ID | `bd6a4a8c0ca5f5f2c50384ef0b88af9edd249b525916273b14e96a9468ef70d5` |
| completed updates | `47,200` |
| next trainer step | `47,200` |
| next global microstep | `188,800` |
| scheduler | `last_epoch=47200`, LR `0.00095` |

raw checkpoint 已确认包含：

- 5,086 个 model tensor；
- 5,082 条 Adam 状态及 param group；
- scheduler 和 BF16 下正确的空 GradScaler state；
- 128 个 rank 的 Python、NumPy、Torch CPU/CUDA RNG；
- 128 个 rank 的 sampler cursor，全部为 epoch `2360`、offset `0`、
  `consumed_global_batches=188800`。

EMA companion 与 raw 的 resume contract 逐字段相同。正式启动还会再次校验文件
大小、SHA-256、raw/EMA pair、world size、模型架构和数据合同，任何一项不一致都在
应用训练状态前失败。

## 保持不变的训练合同

- 16 nodes × 8 GPUs，world size `128`；
- accumulation `4`，global target batch `512`；
- 四库及 raw weights `0.49/0.005/0.505/0.02`；
- 原 `step21999_residual_ge200_g02_v2` schedule，gamma `0.2`；
- schedule cursor 从 `188800` 继续到 `200000`，剩余恰好 `2,800` updates；
- weightedPDB paired 与 genuine-unpaired 各自 10,000 行 source 预算，最终不超过
  16,384；
- MSA sample cutoff/chunk 仍为 `2048/2048`；
- crop `384`、diffusion batch `48`、PF64、模型结构和 loss；
- Adam、LR scheduler、EMA `0.999`、checkpoint interval `200`；
- template cache 和 chain-dummy error policy。

边界：8 个 DataLoader worker 的内部 RNG 没有写入 checkpoint。sampler/target/
cluster/member 顺序以及各 rank 主进程 RNG 会恢复，但 worker 内部的 crop 与部分 MSA
随机细节不承诺 bitwise 完全一致。这是原有 checkpoint 合同的已知边界。

## 16×8 AIStudio 启动命令

先停止误从 `27799` 重启的任务。然后在原来的 16×8 AIStudio 任务规格中，让 16 个
worker 执行完全相同的命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02_msaindependent_only/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_from47199_msaindependent_only_softcluster_g02_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_NNODES=16 \
bash resume_step47199_msaindependent_only.sh \
  > "from47199_msaindependent_only.node${LOG_NODE_ID}.log" 2>&1
```

不要传 `RESUME_STEP`、checkpoint 路径、schedule、训练合同、MSA 独立预算开关或旧
migration mode；新 launcher 会拒绝继承值并设置审计后的固定值。`set -euo
pipefail` 已经写在 launcher 内，AIStudio 外层命令无需再写。

启动日志必须出现：

```text
source_step=47199 next_trainer_step=47200 next_global_step=188800
resume_mode=same_contract contract_migration=none
state=model+optimizer+scheduler+step+scaler+128_rank_rng+sampler_cursor+matched_ema
schedule=... gamma=0.2 cursor=188800 target=200000 unchanged=true
target_max_steps=50000 remaining_updates=2800 checkpoint_interval=200
```

runner 随后还应打印 `Loading optimizer state`、`Loading checkpoint step`、
`Loading scheduler state`、128-rank RNG/sampler cursor 恢复信息，并从 step `47200`
产生第一条新训练记录。

## Dry-run

不读取大 checkpoint 的命令展开检查：

```bash
DRY_RUN=1 SKIP_PATH_CHECK=1 \
MLP_WORKER_NUM=16 MLP_ROLE_INDEX=0 MLP_WORKER_0_HOST=worker-0 \
RUN_NAME=protenix_v4_from47199_msaindependent_only_softcluster_g02_v9_straggler_dryrun \
BASE_DIR=/tmp/protenix_from47199_dryrun \
RESUME_NNODES=16 \
bash resume_step47199_msaindependent_only.sh
```

`SKIP_PATH_CHECK=1` 只能用于 dry-run；真实训练无法绕过 checkpoint 校验。

## 50k 之后

该入口只负责原 gamma `0.2` 合同续到 step `49999`。计划中的 gamma `0` / hard-min
阶段必须用 `49999` raw/EMA、`completed_updates=50000`、
`origin_consumed_global_batches=200000` 建立新的 schedule 和显式训练合同；具体终点由
新 schedule 的覆盖计算决定。不能修改或复用本入口的固定终点，也不能从 `47199`
提前切换 gamma。
