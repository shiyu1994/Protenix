# MSA 2048 / chunk 2048 / FSDP2 正式训练配置

本文固化当前已经完成 benchmark 验证的正式训练配置。唯一正式入口是仓库根目录下的：

```text
run_msa2048_crop1024_nuc256_db32_chunk2048_fsdp2_finetune.sh
```

该入口使用正式三数据集，不使用为性能测试准备的 800 条长蛋白数据集；默认执行
20,000 次 optimizer update，每 1,000 次 update 保存 raw 与 EMA checkpoint，并在
训练结束时保证保存最终 checkpoint。

## 1. AIStudio 启动命令

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload

RUN_NAME=protenix_v2_msa2048_chunk2048_crop1024_nuc256_db32_fsdp2 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
MAX_STEPS=20000 \
bash run_msa2048_crop1024_nuc256_db32_chunk2048_fsdp2_finetune.sh \
  > protenix_v2_msa2048_chunk2048_crop1024_nuc256_db32_fsdp2.log 2>&1
```

`RUN_NAME` 必须显式提供且只能包含字母、数字、点、下划线和短横线。
`BASE_DIR` 的默认值为 `/mnt/project_abd/klp/proteinx_output`，`MAX_STEPS` 的默认值为
`20000`。为了复现本文配置，三者应使用上面的值；更换 `RUN_NAME` 不改变训练语义。
训练产物位于 `${BASE_DIR}/${RUN_NAME}_YYYYMMDD_HHMMSS`，checkpoint 位于其
`checkpoints/` 子目录。

脚本只接受 `RUN_NAME`、`BASE_DIR` 和 `MAX_STEPS` 这三个操作参数，不接受位置参数或
额外 CLI 配置。AIStudio 注入的 `WORLD_SIZE`、`RANK` 等拓扑信息仍由
`finetune_submit.sh` 转换为 `torchrun` 参数。正式启动前可以只展开最终命令：

```bash
RUN_NAME=protenix_v2_msa2048_chunk2048_dryrun \
DRY_RUN=1 \
bash run_msa2048_crop1024_nuc256_db32_chunk2048_fsdp2_finetune.sh
```

## 2. 初始化来源

raw 模型和 EMA shadow 均从同一个官方 PX v2 checkpoint 初始化：

| 用途 | 配置项 | 路径 | SHA256 |
| --- | --- | --- | --- |
| raw 参数 | `load_checkpoint_path` | `/mnt/project_abd/public_databases/protenix.20260101/checkpoint/protenix-v2.pt` | `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599` |
| EMA 参数 | `load_ema_checkpoint_path` | `/mnt/project_abd/public_databases/protenix.20260101/checkpoint/protenix-v2.pt` | `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599` |

非 dry-run 启动会现场计算文件 SHA256，不匹配就直接退出。加载设置为
`load_strict=true`、`load_params_only=true`：模型参数和 EMA 参数来自官方文件，但
optimizer、LR scheduler 和训练 step 都从零重新建立，不继承官方 checkpoint 的训练
状态。

## 3. 正式训练数据

训练 sampler 固定为：

| 数据集 | 采样权重 |
| --- | ---: |
| `weightedPDB_before210930_v20260101` | 0.50 |
| `sabdab-train` | 0.30 |
| `distill-sabdab` | 0.20 |

相关数据根目录为：

- weighted PDB：`/mnt/project_abd/public_databases/protenix.20260101`
- SAbDab 与 distill-SAbDab：`/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix`
- weighted PDB index：`indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz`
- SAbDab index：`indices/sabdab-train.csv`
- distill-SAbDab index：`indices/distill-sabdab.csv`

sampler 设置为 `epoch_size=10000`、`replacement=true`、`seed=-1`，其中负的 sampler
seed 继承全局训练 seed 42；DataLoader worker 数为 16。权重表示名义采样概率，不表示
每 10,000 条一定精确出现 5,000/3,000/2,000 条。

三套数据各自显式设置 `cropping_configs.max_nuc_tokens=256`。这里的 256 是一次 crop
中 DNA 与 RNA token 总数的上限，不是固定核酸长度，也不是为样本补齐核酸。

### crop 1024 的准确含义

`train_crop_size=1024` 是正式随机 cropping 的目标上限，不是 benchmark 的
“每条恰好 1024 token”。正式数据继续使用 weighted data 默认 crop 分布：

```text
ContiguousCropping / SpatialCropping / SpatialInterfaceCropping
                         = 0.2 / 0.4 / 0.4
```

原始结构不足 1024 token 时不会制造 token；对 crop 后超过上限的核酸再应用
`max_nuc_tokens=256` 也可能使最终 token 数低于 1024。因此运行日志中的真实 token 数
可以逐条变化。

## 4. 模型与计算负载

| 配置 | 固定值 |
| --- | --- |
| 模型 | `protenix-v2` |
| 总参数 | 464,442,431（约 464.44 M） |
| 可训练参数 | 464,441,841 |
| 固定参数 | 590 |
| `c_z` | 256，并同步覆盖所有使用 pair channel 的子模块 |
| MSA `c_m` | 128 |
| Pairformer block 数 | 48（官方 PX v2 结构） |
| 数值精度 | BF16 autocast；FSDP 参数/归约设置另见下一节 |
| recycle | `model.N_cycle=4` |
| diffusion train batch | 32 |
| diffusion chunk | 4 |
| diffusion sampling steps | `sample_diffusion.N_step=20` |
| 梯度累积 | 1 |
| triangle kernels | attention 与 multiplicative 均为 `cuequivariance` |

这是近乎全模型训练，不是只训练 Pairformer 之后的层：除 590 个固定常量外，其余
464,441,841 个参数均参与优化。

### Activation checkpoint 与 transition

| 位置 | 设置 |
| --- | --- |
| Pairformer | `blocks_per_ckpt=2` |
| Pairformer checkpoint input CPU offload | `checkpoint_cpu_offload_every_n=2` |
| MSA module | 4 blocks，`blocks_per_ckpt=1` |
| MSA checkpoint input CPU offload | `checkpoint_cpu_offload_every_n=1` |
| diffusion module | `blocks_per_ckpt=1`，`use_fine_grained_checkpoint=true` |
| Pairformer pair transition | chunk 128，chunk checkpoint 开启 |
| diffusion pair transition | chunk 128，chunk checkpoint 开启 |

Pairformer 每 2 层组成一个 checkpoint region，共 24 个 region；每隔 2 个 region
offload 一次 region 输入，即约每 4 个 Pairformer block 一个 offload 边界。MSA 的
4 个 block 各自为一个 checkpoint region，并在每个 region offload checkpoint 输入。

这里的 CPU offload 仅指 activation-checkpoint 重算所需输入的
`torch.autograd.graph.save_on_cpu`。它不把模型参数、梯度或 optimizer state 放到 CPU；
模型参数仍由 FSDP2 管理。

## 5. MSA 设置与 2048 的准确含义

| 层级 | 配置 | 固定值 |
| --- | --- | ---: |
| 数据 featurizer | `data.msa.max_size.train` | 16,384 |
| 模型训练采样下界 | `data.msa.min_size.train` | 2,048 |
| 模型训练采样截断 | `data.msa.sample_cutoff.train` | 2,048 |
| MSA 行计算 chunk | `model.msa_module.msa_chunk_size` | 2,048 |
| protein paired-MSA 处理 | `pair_msa_as_unpaired_only` | `true` |
| paired hit 顺序 | `shuffle_pair_msa_when_unpaired_only` | `true` |

`pair_msa_as_unpaired_only=true` 对三套正式数据都生效。`sabdab-train` 和
`distill-sabdab` 原本有 dataset-level 的 `false` 默认值，launcher 已显式将这两个值
覆盖为 `true`；shuffle 没有被 dataset-level 配置覆盖，继承全局 `true`。

16,384 是数据组装阶段允许保留的最大 MSA pool。每次 MSA module forward 从当前样本
的真实 pool 无放回采样：

- 真实 pool 深度不小于 2,048 时，送入 MSA module 的真实行数为 2,048；
- 真实 pool 深度小于 2,048 时，只能使用已有的全部真实行，不能凭空采出更多序列。

训练计算为了固定图形状，会把不足 2,048 的 MSA 边界用零填充到 2,048 行，再在每个
MSA block 输出处截回真实深度。这些 padding 行只是计算占位，不是有效 MSA，不增加
`sampled_depth`，也不增加生物序列信息。因此正式三数据集不能被描述为“每条样本都
有 2,048 条真实 MSA”；只有源 pool 足够深的样本才满足这一点。

在训练路径中 runtime MSA 高度为 2,048，`msa_chunk_size=2048` 因而是一个完整行
chunk，不做行维度的 256/128 小块拆分。它不会对不足 2,048 的样本再额外 padding 到
下一个 2,048 倍数；此处只有固定到 runtime 高度 2,048 的一次 padding。

## 6. FSDP2、EMA 与优化器

| 配置 | 固定值 |
| --- | --- |
| distributed strategy | `fsdp2` |
| DeviceMesh | `auto` |
| FSDP parameter dtype | FP32 |
| FSDP reduce dtype | FP32 |
| reshard after forward | `true` |
| optimizer sharding 选项 | `none` |
| forward/backward autocast | BF16 |
| EMA decay | 0.999 |
| EMA shadow device | rank-local CPU |
| optimizer | Adam，`beta1=0.9`、`beta2=0.95` |
| AdamW | `false` |
| weight decay | `1e-8` |
| learning rate | `1e-5` |
| LR scheduler | `af3` |
| warmup | 2,000 updates |
| decay interval | 50,000 updates |
| gradient clip norm | 10 |

FSDP2 使用 bottom-up `fully_shard` 和一维 DeviceMesh；`reshard_after_forward=true`
对应 full-shard 生命周期。CPU EMA 保存每个 rank 的本地 DTensor shard shadow，不在
GPU 上复制一整套 464 M 模型参数。

## 7. 训练生命周期与 checkpoint

| 配置 | 固定值 |
| --- | --- |
| 默认训练预算 | 20,000 optimizer updates |
| checkpoint interval | 1,000 updates |
| final checkpoint | 开启，`skip_final_checkpoint=false` |
| 日志 interval | 50 updates |
| periodic eval | 关闭，`eval_interval=0` |
| final eval | 关闭，`eval_at_end=false` |
| WandB | 关闭 |
| benchmark warmup/measure | `-1/-1`，完全关闭 |

由于梯度累积为 1，一个训练 step 就是一次 optimizer update。每次保存同时产生 raw 与
EMA 文件；文件名中的 step 是从 0 开始的内部 step。例如第 1,000 次 update 后保存
`999.pt` 和 `999_ema_0.999.pt`。在默认 20,000 步配置中，第 20,000 次 update 后保存
`19999.pt` 和 `19999_ema_0.999.pt`；周期保存和 final 保存条件同时命中时只保存一次。

## 8. 明确排除的 benchmark-only 设置

正式 launcher 启动时会清空继承到当前 shell 的全部 `BENCHMARK_*` 变量和
`PROTENIX_BENCHMARK_DIR`。下列设置只属于之前的性能实验，不属于正式训练：

- `benchmark_long_protein_context_1024_800_msa2048` 的 800 条专用 index；
- 单数据集权重 1.0、`epoch_size=800`、无放回 sampler 和 seed 77；
- 把 7d1a 安排在第 31 个人工 step 的固定样本顺序；
- 10-step 的 2 warmup + 8 measured window，或其他 10+20 / 10+90 timing window；
- `BENCHMARK_EXACT_CROP=true`、固定 1024-token 数据视图以及 benchmark crop 方法；
- benchmark 的 `checkpoint_interval=-1`、跳过 final checkpoint 或仅采集 summary；
- benchmark 的 4 个 DataLoader workers、单次窗口 log interval 和 NVML profiler。

因此正式训练是可变 token 数、三数据集有放回混合采样的 20,000-step 训练，不能把
800 条 benchmark 的样本覆盖率、固定次序或吞吐直接当成正式 run 的数据语义。

## 9. 已有验证证据及其边界

最终选择 `msa_chunk_size=2048` 的 matched 10-step benchmark 已完成 10/10 步，无
CUDA OOM、NaN、非有限梯度或 skip：2 步 warmup 后 8 步测量均值为约
50.407 秒/step，观测到的最高 NVML 显存为 60,595 MiB。详细结果见
`docs/msa_chunk_2048_no_split_10step_20260716_cn.md`。

该结果证明当前 workload 在当时的 AIStudio GPU/软件环境中能完成完整
forward、loss、backward、optimizer、FSDP2 训练生命周期和 CPU EMA 更新。benchmark
按设计关闭了 checkpoint 保存，因此正式 run 的周期/final checkpoint 仍需单独监控。
它不保证正式三数据集每一步都具有相同显存或速度：正式数据的 token 组成、真实 MSA
深度、核酸/配体组成和首次编译状态都会变化，运行时仍需持续观察峰值显存、非有限值和
checkpoint 落盘状态。
