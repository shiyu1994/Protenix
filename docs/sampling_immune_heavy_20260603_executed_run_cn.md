# `sampling_immune_heavy` 已执行训练逻辑与 checkpoint 来源档案

> 状态：已执行 run 的权威记录  
> 建档日期：2026-07-12  
> 描述对象：2026-06-03 完成的 `sampling_immune_heavy` 训练  
> 文档存放目录：`Protenix_train2/docs/`  
> 实际执行源码目录：`/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train`

这份文档冻结第二套训练的**实际执行逻辑**，重点保存起始 checkpoint 的完整来源链。它不是对当前 `Protenix_train2` 默认参数的描述；文档放在 `Protenix_train2` 只是为了后续开发和对照。

实际运行值的证据优先级为：

1. 输出目录中的 `manifest/run_manifest.json`；
2. 输出目录中的最终 `config.yaml`；
3. `checkpoints/checkpoints.jsonl` 与 `manifest/run_status.json`；
4. 实际执行目录 `Protenix_train` 中的源码；
5. wrapper 默认值和人工说明。

如果这些来源冲突，以前四项中的运行产物为准。

## 1. 实验身份

原始启动命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train

EXPERIMENT_NAME=sampling_immune_heavy \
TRAIN_SAMPLE_WEIGHTS=0.25,0.5,0.25 \
bash run_sab23h2_sampling_finetune.sh
```

实际输出目录：

```text
/mnt/project_abd/klp/proteinx_output/
protenix_sampling_sampling_immune_heavy_from20260531_999ema_lr0.000003_20260603_004205_20260603_004232
```

运行结果：

| 字段 | 实际值 |
|---|---|
| 状态 | `completed` |
| 退出原因 | `max_steps_reached` |
| 开始时间（manifest 记录） | `2026-06-03 00:42:37` |
| 完成时间（status 记录） | `2026-06-03 16:12:58` |
| optimizer updates | 2000 |
| world size | 32（4 节点 × 8 GPU） |
| Python | 3.11.15 |
| PyTorch | 2.7.1+cu126 |
| CUDA | 12.6 |
| Git HEAD | `bd54a05d047b8925a241056f36d619700604068a` |
| Git 状态 | dirty；见第 10 节限制说明 |

`EXPERIMENT_NAME=sampling_immune_heavy` 只参与 run name，不触发任何 Python 训练分支。真正改变数据分布的是 `TRAIN_SAMPLE_WEIGHTS=0.25,0.5,0.25`。

## 2. checkpoint 来源总链路

完整参数来源链是：

```text
官方本地根 checkpoint
/mnt/project_abd/public_databases/protenix.20260101/checkpoint/protenix-v2.pt
    │
    │ params-only 初始化；新建 optimizer/scheduler/EMA；step 从 0 开始
    ▼
上游 weightedPDB + SAbDab finetune
protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805
    │
    │ 完成前 1000 次 update 后保存 zero-based step=999 的 EMA
    ▼
999_ema_0.999.pt
    │
    │ 再次 params-only 初始化；再次新建 optimizer/scheduler/EMA；step 再从 0 开始
    ▼
sampling_immune_heavy
0.25 weightedPDB + 0.50 sabdab-train + 0.25 distill-sabdab
2000 次新 update
```

因此第二套训练不是从上游 step 1000 继续恢复 optimizer 的 resume，而是一个以 `999_ema_0.999.pt` 为参数初始化的 stage-2 warm start。

### 2.1 参数链根：官方 `protenix-v2.pt`

```text
path:
/mnt/project_abd/public_databases/protenix.20260101/checkpoint/protenix-v2.pt

size:
1,859,785,497 bytes

sha256:
8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599
```

该文件内部只有 `model`，包含 4174 个 state-dict entries，不含训练 step。在本地没有比它更早的 run manifest，因此本档案把它作为可核验的 provenance 根。

### 2.2 生成 `999_ema` 的上游训练

上游 run：

```text
/mnt/project_abd/klp/proteinx_output/
protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805
```

上游实际 manifest 记录的关键配置：

| 参数 | 上游实际值 |
|---|---:|
| 初始 normal checkpoint | 官方 `protenix-v2.pt` |
| 初始 EMA checkpoint | 官方 `protenix-v2.pt` |
| `load_params_only` | `true` |
| train sets | `weightedPDB_before210930_v20260101,sabdab-train` |
| train sample weights | `0.7,0.3` |
| distill-sabdab | 未加入该阶段 |
| LR | `1e-5` |
| scheduler | AF3 linear warmup / step decay |
| warmup | 2000 updates |
| max steps | 20000 |
| checkpoint interval | 1000 |
| EMA decay | 0.999 |
| crop size | 768 |
| dtype | bf16 |
| diffusion batch size | 48 |
| `model.N_cycle` | 4 |
| `sample_diffusion.N_step` | 20 |
| eval | 完全关闭 |
| world size | 32 |

checkpoint index 中的来源记录：

```json
{
  "created_at": "2026-05-31 15:25:46",
  "ema_suffix": "_ema_0.999",
  "event": "checkpoint",
  "is_ema": true,
  "path": "/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt",
  "step": 999
}
```

保存判断使用 `(step + 1) % checkpoint_interval == 0`，文件名使用保存前的 zero-based `step`。所以 `999_ema_0.999.pt` 表示**第 1000 次 optimizer update 完成后**的 EMA 参数，而不是只完成了 999 次 update。

该 checkpoint 内部 keys 为 `model/optimizer/scheduler/step`，其中 `step=999`、model state-dict 有 4174 个 entries。EMA 保存时临时把 shadow 参数应用到 model，所以 `model` 字段是 EMA 权重；文件附带的上游 optimizer/scheduler 不会被第二阶段使用。

```text
path:
/mnt/project_abd/klp/proteinx_output/
protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/
checkpoints/999_ema_0.999.pt

size:
5,578,984,015 bytes

sha256:
9e0df397d9deb18ea46921c678a62afc4a587241aacd01acd27887729e3a0c3e
```

上游 step 999 的训练日志记录 LR 为 `5e-6`；它来自 2000-step warmup 的中点。上游之后至少继续保存到了 `5999_ema_0.999.pt`，但第二套训练明确选择早期的 `999_ema_0.999.pt`，没有选择后续 checkpoint。

### 2.3 第二阶段如何加载 `999_ema`

wrapper 把同一个 EMA 文件同时设置为：

```bash
export LOAD_CHECKPOINT_PATH="${base_checkpoint}"
export LOAD_EMA_CHECKPOINT_PATH="${base_checkpoint}"
```

Trainer 初始化顺序是：

```text
init_env
  -> init_basics(step=0)
  -> init_log
  -> init_model + fresh Adam + fresh scheduler + initial EMA register
  -> init_loss
  -> init_data
  -> try_load_checkpoint
```

`try_load_checkpoint` 的有效行为：

```text
1. 将 999_ema checkpoint 的 model 字段加载到 active model；
2. 重新 register EMA shadow，使 shadow 等于这份 EMA 参数；
3. 再从同一路径加载 active model；
4. 因 load_params_only=true，不恢复 checkpoint 中的 optimizer、scheduler、step；
5. sampling_immune_heavy 从新 step 0 和新 warmup 开始。
```

最终 active model 与新的 EMA shadow 起点完全相同，均为上游 `999_ema_0.999.pt` 中的 EMA 权重。

## 3. 入口脚本与参数传递

实际入口脚本：

```text
Protenix_train/run_sab23h2_sampling_finetune.sh
```

当前 `Protenix_train`、`Protenix_train2` 中该脚本内容相同，SHA256 为：

```text
953b08f0e9a8cd50dce4635eb1afb91a26827a3c91e1e1068679035a19d1aa48
```

调用链：

```text
run_sab23h2_sampling_finetune.sh
  -> cd 到脚本硬编码的 Protenix_train
  -> export lowercase training variables
  -> finetune_demo.sh
  -> torchrun ./runner/train.py
  -> parse base/data/model-type configs
  -> CLI overrides 再次覆盖配置
  -> AF3Trainer
```

wrapper 的变量语义：

| 外部变量 | 必需 | 默认值 | 实际作用 |
|---|---|---|---|
| `EXPERIMENT_NAME` | 是 | 无 | 仅用于命名 |
| `TRAIN_SAMPLE_WEIGHTS` | 是 | 无 | 写入 `data.train_sampler.train_sample_weights` |
| `TRAIN_SETS` | 否 | 三个数据集 | 决定数据集及权重的对应顺序 |
| `LR` | 否 | `3e-6` | 全参数 Adam 的目标 LR |
| `MAX_STEPS` | 否 | 2000 | 新训练阶段 optimizer update 数 |
| `CHECKPOINT_INTERVAL` | 否 | 250 | normal/EMA 保存频率 |
| `RUN_TIMESTAMP` | 否 | 当前时间 | wrapper run name 时间戳 |
| `RUN_NAME` | 否 | 自动生成 | 覆盖 wrapper 默认名称 |

脚本还完成 AIStudio 到 torchrun 的映射：

```text
WORLD_SIZE -> NNODES
RANK       -> NODE_RANK
MLP_WORKER_0_HOST -> MASTER_ADDR
MLP_WORKER_0_PORT -> MASTER_PORT
NPROC_PER_NODE 默认 gpu
```

注意：这里的 `WORLD_SIZE` 在 AIStudio wrapper 语境下表示节点数，而 Trainer manifest 中的 `world_size=32` 表示总进程/GPU 数。

## 4. 第二阶段最终有效配置

实际落盘 `config.yaml` 的核心值：

| 分类 | 参数 | 实际值 |
|---|---|---:|
| 模型 | `model_name` | `protenix-v2` |
| 模型 | `model.N_cycle` | 4（训练每步实际在 1–4 间选取） |
| 模型 | `sample_diffusion.N_step` | 20 |
| 模型 | `diffusion_batch_size` | 48 |
| 精度 | `dtype` | bf16 |
| 数据 | `train_crop_size` | 768 |
| 优化 | optimizer | `torch.optim.Adam`，全参数 |
| 优化 | betas | `(0.9, 0.95)` |
| 优化 | weight decay | `1e-8` |
| 优化 | grad clip norm | 10 |
| 优化 | LR | `3e-6` |
| 优化 | scheduler | `af3` |
| 优化 | warmup | 2000 |
| 优化 | max steps | 2000 |
| 优化 | gradient accumulation | 1 |
| EMA | decay | 0.999 |
| 保存 | checkpoint interval | 250 |
| 日志 | log interval | 50 |
| 评估 | `eval_interval` | 0 |
| 评估 | `eval_at_end` | false |
| 评估 | `eval_only` | false |
| 追踪 | wandb | false |

loss 权重：

```text
alpha_diffusion  = 4.0
smooth_lddt      = 1.0
alpha_distogram  = 0.03
alpha_confidence = 0.0001
alpha_pae        = 0.0
alpha_bond       = 0.0
```

这不是 confidence-only、ranking-only 或局部层微调，而是全模型的低学习率训练。`TRAIN_SAMPLE_WEIGHTS` 不进入 loss。

### 4.1 学习率的真实时间线

AF3 scheduler 在 warmup 内使用：

```text
lr(step) = step / warmup_steps * target_lr
```

由于 `max_steps=warmup_steps=2000`，整个第二阶段都处在 warmup：

```text
第 1 次 update 使用 LR 0
第 2 次 update 使用约 1.5e-9
最后一次 update 使用约 2.9985e-6
最后一次 scheduler.step 后日志达到 3e-6，随后训练结束
```

所以该 run 没有满 `3e-6` 的平台期，也没有进入 50000-step decay 阶段；平均实际 LR 约为名义目标的一半。

## 5. 数据集和两级采样

数据集顺序和外层权重：

| 数据集 | 外层概率 | 角色 |
|---|---:|---|
| `weightedPDB_before210930_v20260101` | 0.25 | 通用 PDB 结构能力 |
| `sabdab-train` | 0.50 | 真实抗体/免疫复合物 |
| `distill-sabdab` | 0.25 | SAbDab 相关补充集合 |

这些权重是结构样本的抽样概率，不是 token 比例、atom 比例或 loss 系数。

实现分两级：

```text
第一级：每个数据集内部计算 row weight
第二级：每个数据集内部权重归一化后乘 dataset weight

final_row_weight = dataset_weight * row_weight / sum(dataset_row_weights)
```

最终 `DistributedWeightedSampler` 使用 `torch.multinomial(..., replacement=True)` 生成全局样本序列，再按 rank 切片。每卡 DataLoader batch size 为 1；32 个 rank、无梯度累积时，每次 optimizer update 对应 32 个 cropped structure slots。

完整运行的实际计数：

| 数据集 | structure slots | 实际占比 | token 占比 | atom 占比 |
|---|---:|---:|---:|---:|
| weightedPDB | 16,121 | 25.1891% | 36.8351% | 36.7049% |
| sabdab-train | 31,887 | 49.8234% | 32.6358% | 32.5237% |
| distill-sabdab | 15,992 | 24.9875% | 30.5291% | 30.7714% |
| 总计 | 64,000 | 100% | 100% | 100% |

外层结构比例符合 `0.25/0.50/0.25`；但 weightedPDB 样本通常更长，所以其 token/atom 占比高于 25%。

### 5.1 `distill-sabdab` 名称的实现注意点

执行版本中 `BaseSingleDataset` 只有显式收到 `is_distillation=True` 才设置 distillation 标记。`distill-sabdab` 配置没有传入这个字段，因此实际 `features_dict["is_distillation"]` 仍为 false。该数据集名称本身不会自动切换 distillation loss/label 行为。

## 6. 实际 MSA 逻辑

这是区分“当时真实运行”和“当前 `Protenix_train2` 代码”的关键部分。

### 6.1 当时真实执行的是标准 species pairing

执行目录 `Protenix_train` 当时没有 `pair_msa_as_unpaired_only` 和 paired-row shuffle 的训练入口。蛋白质 MSA 配置为：

```text
prot_pairing_dbs     = ["pairing"]
prot_non_pairing_dbs = ["pairing-non_pairing"]
```

不同蛋白序列组成的复合物会按 species identifier 做正常 paired MSA；paired MSA 中已经使用的序列会从 unpaired 特征中清理。三套数据都使用这套标准逻辑。

### 6.2 `max_size.train=4096` 在旧执行代码中没有生效

实际 `config.yaml` 写有 `data.msa.max_size.train=4096`，但旧版 `get_msa_featurizer` 没有把这个值传给 `MSAFeaturizer`，`make_msa_features` 直接调用默认的 `FeatureAssemblyLine()`。默认 data-side 上限仍为 16384。

所以不能把本次 run 解释为“MSA 最大 4096”。真实 data-side assembled MSA cap 是 16384。

### 6.3 模型侧抽样和固定 padding

实际配置：

```yaml
min_size:
  train: 1
sample_cutoff:
  train: 16384
```

模型每次前向先执行：

```text
N = data-side MSA 行数
K ~ Uniform{1, ..., N}
indices = randperm(N)[:K]
indices = indices[:16384]
```

因为 `N <= 16384`，cutoff 通常不再缩短 K。训练用 MSAStack 为满足 DDP `static_graph=True`，又将 MSA hidden state固定 pad 到 16384 行后计算。

因此第二套训练的有效 MSA 口径是：

```text
标准 species pairing
data-side cap 16384
模型随机深度 1..N
MSAStack 固定 padding 16384
```

### 6.4 与当前 `Protenix_train2` 的区别

当前 `Protenix_train2/finetune_demo.sh` 已新增：

```text
data.msa.max_size.train=16384
data.msa.sample_cutoff.train=2048
pair_msa_as_unpaired_only/shuffle_pair_msa_when_unpaired_only 参数
```

且当前 `Protenix_train2` 已把 `max_size` 真正接入 MSA featurizer。若今天在 `Protenix_train2` 中直接复跑同名 wrapper，默认会得到 train cutoff 2048，而不是 2026-06-03 已执行 run 的 16384。二者不能视为完全相同的实验。

旧文档 `sab23h2_sampling_finetune_full_flow_cn.md` 中的 2048 描述对应当前 `Protenix_train2` 代码路径，不对应本档案所冻结的已执行 run。

## 7. 模型前向、loss 和参数更新

每个 rank 的主要训练顺序：

```text
weighted sampler 取一个 index row
  -> 读取 mmCIF/bioassembly
  -> crop 到最多 768 tokens
  -> 在线生成 MSA/template/features/labels
  -> 搬到当前 GPU
  -> trunk recycle（训练实际 1..4 cycles，只有最后 cycle 保留主梯度）
  -> 48 个 diffusion noisy samples 的单步 denoise 训练
  -> no-grad mini rollout / permutation / confidence 辅助路径
  -> diffusion + distogram + confidence 等 loss
  -> backward
  -> grad clip 10
  -> Adam.step
  -> AF3 scheduler.step
  -> EMA.update(decay=0.999)
  -> 条件满足时保存 normal + EMA checkpoint
```

需要区分两个 batch：

1. 结构 global batch：32 rank × 每 rank 1 个 crop = 32 个结构槽位/update；
2. diffusion batch：每个 crop 内 48 个 noisy coordinate samples，共享 trunk 表示。

名义训练规模：

```text
2000 updates * 32 structure slots = 64,000 cropped structures
64,000 * 48 = 3,072,000 noisy diffusion samples
```

由于 sampler 带放回，64,000 是训练槽位数，不是唯一结构数。

## 8. checkpoint 保存和完成状态

每 250 次 update 保存一次 normal 和 EMA。zero-based 文件名为：

```text
249.pt                 249_ema_0.999.pt
499.pt                 499_ema_0.999.pt
749.pt                 749_ema_0.999.pt
999.pt                 999_ema_0.999.pt
1249.pt                1249_ema_0.999.pt
1499.pt                1499_ema_0.999.pt
1749.pt                1749_ema_0.999.pt
1999.pt                1999_ema_0.999.pt
```

最终状态文件记录：

```text
status                = completed
exit_reason           = max_steps_reached
last_step             = 2000
latest_checkpoint     = .../checkpoints/1999.pt
latest_ema_checkpoint = .../checkpoints/1999_ema_0.999.pt
```

训练内验证完全关闭；checkpoint 的 SAb23H2/DockQ 选择由独立推理和评估队列完成，不能从训练 loss 自动推断最佳 checkpoint。

## 9. 可核验产物及 SHA256

| 产物 | SHA256 |
|---|---|
| 官方 `protenix-v2.pt` | `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599` |
| sampling wrapper | `953b08f0e9a8cd50dce4635eb1afb91a26827a3c91e1e1068679035a19d1aa48` |
| 上游 `config.yaml` | `b7b741f93b738049c4d28f8686f92d442e8fb056d237de319ea9935c31eda402` |
| 上游 `run_manifest.json` | `87b257c000f6559c50594135d887c133b2eacbe77c452c2174c55828faa5b98d` |
| 上游 raw `999.pt` | `b91a12f3fddcbd7c655402ab3b4aa61f89927bb582009d0e54c4f5c3a74633cb` |
| 起始 `999_ema_0.999.pt` | `9e0df397d9deb18ea46921c678a62afc4a587241aacd01acd27887729e3a0c3e` |
| 第二阶段 `config.yaml` | `8df323f63ab961fe1753532af69dd56577453a3afae5bc95f3a0253a83d98098` |
| 第二阶段 `run_manifest.json` | `565099c6ae840acc945d7c1d2b0deb220e2ce22f5a19fccfcaa1252e064add27` |

对应路径：

```text
上游 config/manifest/checkpoints:
/mnt/project_abd/klp/proteinx_output/
protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/

第二阶段 config/manifest/status/logs/checkpoints:
/mnt/project_abd/klp/proteinx_output/
protenix_sampling_sampling_immune_heavy_from20260531_999ema_lr0.000003_20260603_004205_20260603_004232/
```

## 10. provenance 限制与复现要求

### 10.1 运行时 Git worktree 是 dirty

两个 run manifest 都记录：

```text
branch = main
commit = bd54a05d047b8925a241056f36d619700604068a
dirty  = true
```

manifest 保存了 `git status --short`，但没有保存所有 dirty diff 的内容。因此仅 checkout 该 commit 不能逐字节恢复运行时源码。这是当前来源链唯一无法完全闭合的部分。

本档案通过以下方式尽量固定行为：

- 保存实际 resolved config 和 manifest 的路径与哈希；
- 保存入口脚本哈希；
- 保存起始 checkpoint 的路径、大小和哈希；
- 以实际日志验证数据比例、步数和最终状态；
- 明确记录旧 `Protenix_train` 和当前 `Protenix_train2` 的 MSA 行为差异。

### 10.2 严格复现至少要固定

```text
起始 checkpoint SHA256
实际源码 worktree/snapshot
resolved config.yaml
三个 index 文件及 MSA/template mapping 的 checksum
world size 和 GPU/torch/cuda 环境
seed=42
数据失败重采样日志
```

### 10.3 本档案的一句话实验定义

`sampling_immune_heavy` 是：从官方 `protenix-v2.pt` 经上游 `0.7 weightedPDB + 0.3 sabdab-train` 前 1000 次 update 得到的 `999_ema_0.999.pt` 出发，丢弃上游 optimizer/scheduler/step，以新 Adam 和新 2000-step warmup，在 `0.25 weightedPDB + 0.50 sabdab-train + 0.25 distill-sabdab` 的结构采样分布下完成的 32-GPU 全参数 stage-2 finetune；其已执行版本使用标准 species pairing、data-side MSA 上限 16384 和模型侧 1..N 随机深度/16384 固定 padding。
