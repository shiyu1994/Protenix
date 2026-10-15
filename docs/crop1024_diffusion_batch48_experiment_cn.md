# crop1024 / diffusion batch 48 对照试验

## 1. 试验目的与差异边界

本仓库是
`Protenix_train2_fsdp2_msa2048_offload`
的独立副本，用来比较当前稳定训练配置下 diffusion batch 32 与 48 的显存和速度。

相对当前 db32 基线，唯一的科学设置变化是：

```text
diffusion_batch_size: 32 -> 48
```

这里的 diffusion batch 是**一个 target/crop 内部生成的 diffusion/noise training
samples 数量**，不是一次 DataLoader 读取 48 个 target，也不改变每个 optimizer
step 的 target 数量。继续使用 `diffusion_chunk_size=4`，所以每个 target 的 diffusion
计算从：

```text
db32: 32 / 4 =  8 个 chunk
db48: 48 / 4 = 12 个 chunk
```

Pairformer、MSA 等 target 级前端计算没有因为这个数字重复 1.5 倍；主要增加的是
diffusion 部分的工作量。峰值显存和整体 step 时间必须通过真实 GPU benchmark
测量，不能直接按 50% 外推。

## 2. 保持不变的训练内容

除 db48 外，继续继承当前 `immune_heavy_old_content` 配置：

- `train_crop_size=1024`，为正式随机 crop 上限，不代表每条样本都严格为 1024 token；
- 三个数据集的 `max_nuc_tokens=256`；
- 数据集依次为 weightedPDB、`sabdab-train`、`distill-sabdab`，权重
  `0.25 / 0.50 / 0.25`；
- MSA featurizer pool 上限 16,384 行；每个 recycle 在
  `1..min(actual_depth, 2048)` 独立随机取真实深度；计算高度和 chunk 均为 2,048；
- paired MSA 按 unpaired 使用并 shuffle；
- `c_z=256`、`c_m=128`、BF16、`N_cycle=4`、diffusion `N_step=20`；
- 全模型训练，FSDP2 full-shard、rank-local CPU EMA；
- Pairformer 每 2 层一个 checkpoint region，并保留当前 activation checkpoint
  input CPU offload；Transition chunk 128 且开启 chunk checkpoint；
- 学习率 `3e-6`、warmup 2,000、总训练 2,000 step、每 250 step 保存；
- optimizer、loss、随机采样和 checkpoint/resume 语义均不改变。

raw model 和 EMA 都继续从同一个历史 checkpoint 做 params-only warm start：

```text
/mnt/project_abd/klp/proteinx_output/
  protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/
  checkpoints/999_ema_0.999.pt
```

该文件是上游训练完成第 1,000 次 optimizer update 后保存的 EMA 权重。新任务不会
恢复上游 optimizer、scheduler 或 step；它们仍从新的 step 0 开始。

## 3. AIStudio 正式训练

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_db48/

RUN_NAME=immune_heavy_old_content_crop1024_nuc256_db48_fsdp2 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_immune_heavy_old_content_crop1024_nuc256_db48_fsdp2_finetune.sh \
  > immune_heavy_old_content_crop1024_nuc256_db48_fsdp2.log 2>&1
```

不要复用 db32 的 `RUN_NAME` 或日志文件名，否则会混淆 checkpoint、manifest 和
运行日志。

## 4. 提交前验证顺序

### 4.1 dry-run

先验证 profile、路径和最终参数展开，不启动训练：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_db48/

RUN_NAME=immune_heavy_old_content_crop1024_nuc256_db48_fsdp2_dryrun \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
DRY_RUN=1 \
bash run_immune_heavy_old_content_crop1024_nuc256_db48_fsdp2_finetune.sh
```

输出中必须明确显示 `diffusion_batch/chunk=48/4`，并且 checkpoint、数据权重、
crop、MSA、FSDP2 和 EMA 设置与 db32 基线一致。

### 4.2 one-step smoke

一步模式覆盖 checkpoint 加载、forward、backward 和一次 optimizer update，并跳过
大 checkpoint 保存：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_db48/

RUN_NAME=immune_heavy_old_content_crop1024_nuc256_db48_fsdp2_smoke1 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
PROTENIX_SMOKE_TEST=1 \
bash run_immune_heavy_old_content_crop1024_nuc256_db48_fsdp2_finetune.sh \
  > immune_heavy_old_content_crop1024_nuc256_db48_fsdp2_smoke1.log 2>&1
```

成功标准至少包括所有 rank 完成一次参数更新、无 OOM/NaN/未捕获异常，并生成完整的
运行状态记录。随机正式数据的一步 smoke 不保证恰好抽到 1024-token 样本。

### 4.3 长 1024 benchmark

smoke 成功后，再使用准备好的 exact-crop 1024 长样本做多步 benchmark，并与 db32
使用完全相同的数据顺序、warmup/measure 步数、GPU 数量和采样周期。重点比较：

- 全局 GPU 峰值和各 rank 峰值；
- warmup 后的平均/中位 step time；
- forward、backward 和 optimizer update 是否都成功；
- 是否出现 OOM、non-finite loss/gradient 或 rank 间退出不一致。

只有 exact-crop 1024 benchmark 通过后，才提交上面的 2,000-step 正式任务。

## 5. 与当前 db32 训练的隔离

db48 使用独立代码目录、独立入口、独立 `RUN_NAME` 和独立输出目录。没有修改源仓库：

```text
Protenix_train2_fsdp2_msa2048_offload
```

因此已经由源目录启动的 db32 进程不会重新读取本仓库的 shell 或 Python 文件，
不会受到 db48 试验影响。后续若需重启 db32，也应继续从源目录使用原 db32 入口。
