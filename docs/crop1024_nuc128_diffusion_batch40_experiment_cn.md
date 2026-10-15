# crop1024 / nuc128 / diffusion batch 40 试验

## 1. 试验设置

本仓库用于验证降低核酸 crop 上限并使用中间 diffusion batch 后，正式长训练能否
稳定运行，并验证恢复原 immune-heavy 的正常跨链 MSA 配对后效果。新入口锁定：

```text
max_nuc_tokens:       256 -> 128
diffusion_batch_size:  48 -> 40
pair_msa_as_unpaired_only: true -> false
shuffle_pair_msa_when_unpaired_only: true -> false
```

总 `train_crop_size` 仍为 1024。`max_nuc_tokens=128` 只限制一个 crop 中最多保留
多少核酸 token；实际核酸不足 128 时全部使用，不会 padding 到 128。

`diffusion_batch_size=40` 表示一个 target/crop 内生成 40 份 noisy diffusion
training samples，不是一次读取 40 个 target。`diffusion_chunk_size=4`，所以每个
target 的 diffusion 计算分成 10 个 chunk。

## 2. 最终训练内容与保持不变项

- 数据集为 weightedPDB、`sabdab-train`、`distill-sabdab`，采样权重
  `0.25 / 0.50 / 0.25`；
- crop 上限 1024，三个数据集的核酸上限统一为 128；
- MSA featurizer pool 上限 16,384；每个 recycle 从
  `1..min(actual_depth, 2048)` 独立随机采样；计算高度和 chunk 均为 2,048；
- 多蛋白链恢复正常 species pairing，不再把 paired A3M 合并为各链 unpaired，
  也不再打乱 paired hits；
- `c_z=256`、`c_m=128`、BF16、`N_cycle=4`、diffusion `N_step=20`；
- 全模型训练，共 464,441,841 个可训练参数；
- FSDP2 full-shard、rank-local CPU EMA，EMA decay 为 0.999；
- Pairformer 每 2 层一个 checkpoint region，每隔 2 个 region 对 checkpoint
  输入做 CPU offload；MSA offload cadence 为 1；
- Transition chunk 128，并开启 chunk checkpoint；
- 学习率 `3e-6`、warmup 2,000、总训练 2,000 step、每 250 step 保存 raw 和 EMA。

raw 参数与 EMA 参数均从旧训练的同一个 EMA checkpoint 做 params-only 初始化：

```text
/mnt/project_abd/klp/proteinx_output/
  protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/
  checkpoints/999_ema_0.999.pt
```

optimizer、scheduler 和训练 step 不从上游恢复，新任务从 step 0 开始。

## 3. 专用入口

只使用：

```text
run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh
```

该入口会覆盖复用 shell 中残留的 profile 环境变量，强制选择 old-content、
nuc128、db40 和 `species_paired`。模型 Python 实现没有为本试验改动。

## 4. 提交前 dry-run

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40/

RUN_NAME=immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_fsdp2_dryrun \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
DRY_RUN=1 \
NPROC_PER_NODE=1 \
bash run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh
```

输出必须同时包含：

```text
content_profile=immune_heavy_old_content
crop=1024 max_nuc_tokens=128 diffusion_batch/chunk=40/4
MSA min/cutoff/pool/chunk=1/2048/16384/2048
pair_msa_profile=species_paired pair_as_unpaired=false shuffle_paired_hits=false
```

展开的 `torchrun` 命令还必须对三个数据集分别出现
`cropping_configs.max_nuc_tokens 128`，并且全局及两个 SAbDab 局部配置的
`pair_msa_as_unpaired_only` 均为 `false`。

## 5. AIStudio 正式训练

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_fsdp2 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh \
  > "immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_fsdp2.node${LOG_NODE_ID}.log" 2>&1
```

所有节点使用相同 `RUN_NAME`，但外层 stdout 按 worker 分文件保存。这样即使某个
rank 首先 OOM，也不会再被其他节点的 TCPStore 连锁报错覆盖。

## 6. 成功标准

- 日志确认实际为 crop1024、nuc128、db40/chunk4、正常 species-paired MSA；
- 所有 rank 能持续完成 forward、backward 和 optimizer update；
- 无 CUDA OOM、NaN、non-finite gradient 或 rank 间退出不一致；
- 成功越过此前 db48 两次失败的 step 250；
- step 249 和后续周期 checkpoint 的 raw/EMA 文件完整生成。

db40 和 nuc128 都会降低部分工作负载，但峰值显存取决于实际 crop 的 token/atom
组成和各模块临时张量，是否足够稳定必须以这次真实训练结果为准。
