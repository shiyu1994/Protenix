# crop384 / DB48 / DDP 随机初始化 5 万次更新快速训练说明

> **版本说明：** 本文主体记录复制前的 v8 科学配置和 strict-resume 设计，保留作
> 基线审计。当前仓库已经切换为 v9-straggler contract，`RUN_NAME`、AIStudio
> 命令、DataLoader/health-sync 差异以及 100/250-update benchmark 必须以
> [v9 straggler 优化与 benchmark 说明](crop384_db48_ddp_v9_straggler_benchmark_cn.md)
> 为准。本文中的 v8 路径、contract ID 和启动命令不能直接用于当前入口。

## 1. 训练目标与入口

本仓库用于在 Protenix-v2 网络架构上做长期、通用的从头训练：

- 训练从 step 0 开始，目标是绝对 50,000 次 optimizer update；
- 不加载官方 `protenix-v2.pt`；
- 不加载任何旧训练的 raw 或 EMA checkpoint；
- 使用 crop 384、diffusion batch 48 和 16×8=128 卡 DDP；
- 每 rank 每个 micro-step 处理 1 个 target，累计 4 次后更新，全局 target
  batch 为 512；
- checkpoint 默认每 1,000 step 保存一次。

唯一正式的随机初始化入口是：

```text
run_crop384_db48_ddp_scratch_50k.sh
```

仓库目录名与内部兼容实现仍带 `200k`，不代表当前更新数。
这个入口会拒绝非空的 checkpoint/resume 环境变量和 checkpoint 命令行参数，并显式
抑制底层 fine-tune 入口的官方 checkpoint fallback。配置中的
`MODEL_NAME=protenix-v2` 只选择模型架构，不表示下载或加载官方参数。

真正的随机初始化也不意味着每个 tensor 都必须服从随机分布：网络代码中有意定义的
zero initializer、固定常量和不可训练结构参数仍按架构要求初始化；关键保证是没有任何
参数来自官方模型或历史训练 checkpoint。

> **警告：不要直接运行 `finetune_demo.sh`。** 它保留了 fine-tune 场景的官方
> checkpoint fallback。绕过锁定入口会把“从头训练”悄悄变成“从官方权重微调”。

## 2. 锁定的科学配置

| 项目 | 当前值 |
| --- | --- |
| 模型架构 | Protenix-v2 宽度，Pairformer `64` 层，`c_s=384`、`c_z=256`、`c_m=128` |
| Diffusion / conditioning | `c_token=768`、`c_s=384`；本次不改 conditioning |
| 参数量 | 总计 `529,902,655`；可训练 `529,902,065` |
| 初始化 | 真正随机初始化，无 raw/EMA checkpoint |
| 分布式策略 | 16 节点 × 8 GPU DDP，world size 128；optimizer 不分片 |
| 精度 | BF16 |
| crop | 最多 384 token；正式训练不是 exact crop |
| diffusion batch | 每个 target 48 个 diffusion sample |
| recycle / diffusion step | 每步 1–4 轮（`N_cycle=4` 为上限）/ `N_step=20` |
| 总步数 | 50,000 个累计窗口；正常时为 50,000 次参数更新 |
| 优化器 | Adam，不使用 AdamW |
| 学习率 | `1e-3` |
| warmup | 1,000 optimizer update |
| Adam 参数 | `beta1=0.9`、`beta2=0.95`、`weight_decay=1e-8` |
| scheduler | AF3 scheduler；`decay_every_n_steps=25000`、factor `0.95` |
| gradient accumulation | 4；每个 micro-step 都 backward，第 4 个才更新 |
| 全局 target batch | `128 ranks × 1 target/rank × 4 = 512` |
| micro-step / target draw | 200,000 micro-step/rank / 25,600,000 global draw |
| sampler epoch | `epoch_size=10240`；每 rank 80 micro-step/20 update；共 2,500 epoch |
| gradient clipping | norm 10 |
| EMA | decay `0.999`，shadow 常驻模型所在 GPU |
| checkpoint | 每 1,000 step；正式训练结束时保存最终 checkpoint |
| 训练日志 | 每 25 update |
| 验证 | 训练内关闭；独立单卡任务每 5,000 update 验证 EMA checkpoint |

学习率用 1,000 update warmup 到 `1e-3`，在 25,000-update 边界将后续
learning rate 乘到 `9.5e-4`。第二个乘法边界位于 50,000-update 训练结束处，
不会再开启第三段训练。

60,000–75,000 update 是读取 50k EMA 验证结果后才可决定的 continuation
上限，不是当前 `max_steps`。本入口和 strict-resume 入口目前都 fail-closed
锁定 50,000；若验证后决定继续，应新增明确的 continuation stage/contract，
不能修改已有 v8 checkpoint 的训练身份。

当前阶段的数据 profile 和 loss 已锁定：`weighted_pdb_only_v1` +
official initial-training loss。后续蒸馏数据不会原地修改本 profile，而会使用
新的命名 profile、阶段入口和 checkpoint 合同。

### 数据组成

| 数据集 | 权重 | `max_nuc_tokens` |
| --- | ---: | ---: |
| `weightedPDB_before210930_v20260101` | 1.00 | 128 |

`max_nuc_tokens=128` 是 crop 384 内允许保留的核酸 token 上限，不会把没有核酸的
样本 padding 成 128 个核酸 token，也不会改变总 crop 大小。

正式数据管线中 `BENCHMARK_EXACT_CROP=false`。短于 384 的结构保持较短；空间 crop
为保留完整 ligand 或限制核酸 token 还可能得到少于 384 的结果。因此
exact-crop-384 benchmark 是显存和吞吐的上界压力负载，不是正式训练的平均
样本长度；旧三数据集混合统计不再用于推断当前 weightedPDB-only 分布。

### MSA 语义

- 每个训练 target occurrence 只做一次确定性三分类 gate：20% 使用
  query-only MSA，30% 使用旧 `true/true` unpaired-shuffle，50% 使用正常
  species-paired MSA；三类互斥；
- query-only 对 protein 与 RNA 同时生效，不读取 homolog MSA；多链 target
  仍是一个整体，按 crop 后 token 顺序组成唯一一条全局 query row，不会按 chain
  各保留一行；
- query-only 会同步裁成一行的所有 row feature，并从唯一 query row 重算
  `profile` 和 `deletion_mean`，因此不会通过统计摘要泄露 homolog 信息；
- gate 对一个 target 的所有 chain 和全部 recycle 固定不变；不是每条 chain、
  每个 recycle 或每个 MSA row 独立抽样；
- 30% 旧分支逐 occurrence 使用
  `pair_msa_as_unpaired_only=true` 和
  `shuffle_pair_msa_when_unpaired_only=true`：关闭 species pairing，把 paired
  hits 并入每条 chain 的 unpaired 候选，并在保留 query row 的前提下 shuffle；
- 50% normal 分支逐 occurrence 显式使用 `false/false`，继续正常 species
  pairing；launcher 的静态默认也保持 `false/false`；
- raw protein A3M 会先解析；经过 dedup、分支对应的 pairing、头部截断和多链合并后，
  **最终 assembled pool** 才限制为最多 16,384 行。这个值不是 raw 文件读取
  上限，也不会把 pool padding 到 16,384；
- 设 assembled pool 实际深度为 D，每个 recycle 独立采
  `K ~ UniformInteger[1, min(D, 2048)]`，随后从完整 D 行中执行
  `randperm(D)` 无放回选择 K 行；当 `K<=1024` 时，固定保留 index 0
  的全局 query，再从剩余随机排列中取 `K-1` 行。最终深度仍为 K；
- sampled feature、embedding 和 OuterProductMean 按真实 K 行计算；
- 4 个 MSA block 中前 3 个的 MSAStack 只在 pair-weighted averaging 与
  Transition 内部临时补零到 2,048，计算后切回 K；末 block 不做这类 padding；
- `msa_chunk_size=2048` 仅表示上述临时 2,048 高度计算为一个 chunk，不改变
  pool、K 或抽样行；
- train/inference 都在 `K<=1024` 时保证保留 query row，`K>1024`
  时保持原始全行随机策略；
- `profile` 和 `deletion_mean` 在抽 K 行前已由完整 unpaired MSA 计算，故
  两个 homolog 分支的 `K=1` 也不是严格的 single-sequence/no-MSA 输入。

这个 gate 只用于训练 dataset。Inference 继续走独立
`InferenceMSAFeaturizer` 与当前推理配置，不读取上述三个训练概率。

这里的 2,048 不是“多轮 recycle 合计使用 2,048 行”。训练每步根据当前 step
确定性选择 1–4 个实际 recycle，已执行的每一轮都单独进行一次上述随机高度选择；
pool 不够深时，上限自然变为该 target 的实际 MSA depth，只有最后一轮保留梯度。
同一 accumulation window 的四个 micro-batch 使用相同 recycle 轮数，但 MSA
torch RNG 继续前进，通常会得到不同的 K 和行子集。

### Template occurrence gate

- 与 MSA gate 相互独立，每个 target occurrence 有 20% 概率请求真实 template，
  80% 概率强制 no-template；
- 真实 template 搜索沿用现有实现，只针对 target 中的 protein chain；RNA/DNA
  本身不会因此获得结构模板；
- no-template 使用官方已有的固定 4-slot dummy 表示：坐标/mask 为零并保留固定
  shape，而不是删除 TemplateEmbedder 或改变模型参数；
- 请求真实 template 时，若过滤后存在有效候选，则随机选择 1–4 个，而不会再选
  0 个；若该 target 根本没有有效候选，输出仍是 dummy，所以日志中的“实际有真实
  template 比例”可能低于 20%；
- 正式入口把旧 `data.template.template_dropout_rate` 锁为 `0.0`，避免模板层
  再做第二次随机 dropout。

MSA categorical gate 与 template Bernoulli gate 使用
`sample_occurrence_sha256_v1(seed, sampler_epoch, global_position, stream)`
生成互相独立的决策，不推进 Python、NumPy 或 torch RNG。名义联合比例
为：

| MSA / template 请求 | 名义比例 |
| --- | ---: |
| query-only + 请求真实 template | 4% |
| query-only + no-template | 16% |
| unpaired-shuffle + 请求真实 template | 6% |
| unpaired-shuffle + no-template | 24% |
| species-paired + 请求真实 template | 10% |
| species-paired + no-template | 40% |

`train_metrics.jsonl` 的 `dataset_window.augmentation` 会记录三种 MSA 模式、
template 请求比例、实际有效 template 比例和六种联合模式；日志间隔采样到的
sample 字段也写入 `data_samples.jsonl`/`runtime.jsonl`。strict resume 恢复相同 sampler cursor 后，
相同 occurrence 的两个 gate 完全一致；旧 `true/true` 分支内部的 paired-hit
shuffle 仍使用 DataLoader worker NumPy RNG，因此其逐行顺序以及其他随机
crop/MSA 行选择仍遵循原有 worker RNG 恢复边界。

### 数据 profile 扩展接口

当前不可变 profile 定义在：

```text
configs/crop384_db48_data_profiles.sh
```

现有正式入口永久锁定 `weighted_pdb_only_v1`；即使复用 shell 时传入其他
`PROTENIX_TRAINING_DATA_PROFILE` 也会在 `torchrun` 前失败。未来加入多个蒸馏集
时，需要：

1. 在 `configs/configs_data.py` 注册每个版本化 dataset；
2. 在 registry 新增有唯一名字、weights、contract tag、每数据集 override 和
   required paths 的 profile；
3. 建立新的阶段 scratch/resume 入口，不能用本入口切换 profile；
4. 新增 dry-run/strict-contract 测试并重新计算各数据源暴露量。

底层仍支持任意数量的 train set；这是对正式科学配置做版本化锁定，不是删除
多数据集能力。

## 3. AIStudio 正式启动

AIStudio 任务必须分配 16 个 worker、每 worker 8 张 H100。每个 worker 都执行
下面完全相同的命令。启动脚本读取平台注入的
`MLP_WORKER_*`/`MLP_ROLE_INDEX` 坐标建立 DDP，并在 `torchrun` 前拒绝任何不是
16×8=128 的拓扑；child 还会再次校验 `WORLD_SIZE=128` 和
`LOCAL_WORLD_SIZE=8`。所有 worker 必须使用相同 `RUN_NAME` 和 `BASE_DIR`。
外层日志名包含节点编号，避免共享目录中的日志互相覆盖。

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_50k \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_scratch_50k.sh \
  > "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_50k.node${LOG_NODE_ID}.log" 2>&1
```

`diffusion_batch_size=48` 是单个 target 内的 diffusion noise sample 数，不参与
上面的全局 target batch 512 计算。累计窗口中四个 micro-step 都执行 forward 和
backward，且每个 loss 都除以 4。每次 scratch 启动或 strict resume 新建 DDP
wrapper 后，第一次 backward 强制同步，以完成 PyTorch static-graph reducer
初始化；首个累计窗口的第 2–3 个 micro-step 使用 `no_sync()`，第 4 个按正常
更新边界同步并执行 optimizer、scheduler 和 EMA update。从下一个窗口起进入
稳态：前三个 micro-step 使用 `no_sync()`，第 4 个跨 128 卡归约并更新，即
`no_sync × 3 → sync/update`。额外通信在每次进程启动时只发生一次，不改变四个
`loss / 4` 的求和结果、有效 batch 512 或后续稳态吞吐。

正式提交前应先用同一套 16×8 资源做 20-update 生命周期 smoke。它仍然是随机
初始化，只把总步数临时锁为 20（每 rank 80 个 micro-step），并禁用 checkpoint
写入：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_smoke20 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
PROTENIX_SMOKE_TEST=1 \
PROTENIX_SMOKE_UPDATES=20 \
bash run_crop384_db48_ddp_scratch_50k.sh \
  > "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_smoke20.node${LOG_NODE_ID}.log" 2>&1
```

smoke 成功只证明 128 卡初始化、数据、累计、`no_sync`、forward、backward 和
optimizer update 生命周期可运行；它不能证明 5 万次更新的稳定性。PF8+c48 的既有
10+20 benchmark 使用的是 48 层 Pairformer，只能作为单机 8 卡、
accumulation=1 的旧模块性能证据，不能替代这次 PF64、128 卡
accumulation=4 的显存、扩展效率和通信验证。

针对 2026-07-24 首次 128 卡启动在 step 0 暴露的 PyTorch 2.7.1
`static_graph + first-backward no_sync` 断言，仓库提供了独立两卡 NCCL 回归：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k/

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$(pwd)" \
  /mnt/project_abd/klp/miniconda3/envs/protenix/bin/torchrun \
  --standalone --nproc_per_node=2 \
  tests/distributed_ddp_static_graph_accumulation_smoke.py
```

该回归已经通过，覆盖 scratch 首次同步、下一窗口 `no_sync × 3 → sync`、非零
global step 的新 DDP/resume reducer，并精确校验跨 rank 梯度。它证明 reducer
修复和梯度数学正确，但不替代上述完整 PF64、真实数据、128 卡 20-update smoke。

## 4. 最终生产加速配置：PF64 + PF8 grouping + c48

下列纯实现设置已经写入正式 scratch 与 strict-resume 入口。checkpoint grouping
和 chunk 策略在旧 PF48 网络上做过 matched benchmark；PF64 尚需重新测实际显存
与速度：

| 加速项 | 最终生产值 | 证据 |
| --- | --- | --- |
| Pairformer/MSA/Diffusion checkpoint 分组 | `8 / 4 / 3` | PF8+c48 matched run |
| checkpoint input CPU offload | PF/MSA 均关闭（`null`） | 峰值仍有约 38 GiB 卡上余量 |
| diffusion chunk size | `48` | DB48 单 chunk |
| diffusion shared cache | 开启 | matched speed profile |
| EMA shadow | 模型所在 GPU | matched speed profile |
| optimizer / DDP | fused Adam / `gradient_as_bucket_view=true` | matched speed profile |
| Transition chunk/checkpoint | `null / false` | matched speed profile |
| diffusion fine checkpoint | 开启 | matched speed profile |
| 旧 PF48 的 10+20 正式窗口 | `11.694 s/step`、`262.70 token/s` | 20 measured steps |
| 旧 PF48 的后 10 步稳态 | `10.390 s/step`、`295.67 token/s` | 动态编译长尾消退后 |
| 旧 PF48 峰值显存 | allocated `38.83 GiB`；reserved `39.08 GiB` | 8×H100 单机 DDP |

该 benchmark 覆盖 240/240 个唯一的 exact-crop-384 样本，并把 MSA lower bound
也锁为 2,048，所以 648/648 次实际采样都是 2,048 行；正式 weightedPDB profile
的 lower bound 是 1，真实 K 按上一节随机。因此 benchmark 是 matched 实现对比
和压力负载，不代表生产平均 step 时间。benchmark 包含 28 个核酸样本（最大 83
token）；loss、gradient 和 optimizer update 均有限，错误诊断目录为空。旧版与
修复后版本的后 10 步分别为
`10.361` 与 `10.390 s/step`，只差 0.28%，说明原子置换修复没有改变 PF8
稳态性能。

上述 benchmark 网络仍是 48 层 Pairformer。本次增加到 64 层并增加
65,460,224 个参数，因此不能外推 PF64 的速度或峰值显存。正式提交前必须使用
锁定入口做同拓扑 20-update PF64 smoke；必要时再做 exact-crop matched
benchmark。

仓库内用于候选筛选和 matched 比较的入口为：

```text
run_crop384_db48_ddp_speed_trial.sh
run_crop384_db48_randominit_smoke_benchmark.sh
run_crop384_db48_randominit_matched_benchmark.sh
```

## 5. 独立单卡 EMA 验证

正式训练始终保持 `eval_interval=0 / eval_at_end=false`。不要在 128 卡任务中
打开 inline evaluation：它会暂停全部训练 rank、占用训练卡，而且当前 eval
还会消耗训练进程 RNG。固定验证通过共享存储读取已经原子提交的 EMA checkpoint，
在另一个单 GPU AIStudio 任务中运行：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k/

CUDA_VISIBLE_DEVICES=0 \
TRAIN_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<actual_training_run_dir> \
bash run_crop384_db48_ema_eval_queue.sh \
  > crop384_db48_v8_ema_eval_queue.log 2>&1
```

这里的 `CUDA_VISIBLE_DEVICES=0` 是独立验证任务分配到的 GPU，不是训练 worker
上的 GPU。evaluator 会再次要求 `torch.cuda.device_count()==1`，防止误占多卡。
队列按下列固定映射顺序等待：

```text
5,000 updates  ->  4999_ema_0.999.pt
10,000 updates ->  9999_ema_0.999.pt
...
50,000 updates -> 49999_ema_0.999.pt
```

EMA 文件由训练进程先写临时文件，再通过 `os.replace` 原子提交；最终文件出现即可
视为 ready。每个 checkpoint 在加载前会校验 v8 training contract、world size
128、accumulation 4、weightedPDB-only profile 和 PF64/c_z256/c_m128 架构。

验证固定为单 rank、PoseBusters `max_n_token=512`、135 个 target、seed 42、
`N_sample=5`、`N_step=20`，并关闭 MC dropout 与 test reference-position
augmentation。`eval_only` 只加载 EMA，不加载官方或 raw 参数，不保存新
checkpoint。结果位于独立输出目录的 `logs/eval_metrics.jsonl`、
`logs/eval_samples_rank0.jsonl` 和 `logs/eval_worst_samples.jsonl`。
params-only evaluation 不恢复训练 step，因此这些日志内部的 `step=0` 不能当作
源 update；真实 update 已写入 run name、queue marker 和 evaluator 外层日志。

只验证一个 checkpoint 时使用：

```bash
CUDA_VISIBLE_DEVICES=0 \
EMA_CHECKPOINT=/mnt/project_abd/klp/proteinx_output/<run>/checkpoints/4999_ema_0.999.pt \
RUN_NAME=<run>_ema_eval_u5000 \
bash eval_crop384_db48_ema_checkpoint.sh
```

独立验证不会占用 128 张训练卡，也不会改变训练参数或训练 RNG；两项任务仍会共享
checkpoint 文件系统 I/O。

## 6. 严格 full-state resume

随机初始化任务首次启动后，后续恢复只能使用：

```text
resume_crop384_db48_ddp_scratch_50k.sh
```

以下示例从源 run 的 `999.pt` 与 `999_ema_0.999.pt` 恢复，并从 step 1,000
继续。AIStudio 多机任务的每个 worker 同样执行相同命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=protenix_v2_scratch_crop384_db48_ddp_resume1000 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<source_run> \
RESUME_STEP=999 \
bash resume_crop384_db48_ddp_scratch_50k.sh \
  > "protenix_v2_scratch_crop384_db48_ddp_resume1000.node${LOG_NODE_ID}.log" 2>&1
```

默认路径为：

```text
raw: <RESUME_RUN_DIR>/checkpoints/<RESUME_STEP>.pt
EMA: <RESUME_RUN_DIR>/checkpoints/<RESUME_STEP>_ema_0.999.pt
```

严格入口会恢复并验证：

- raw model；
- Adam optimizer moments；
- AF3 scheduler state；
- step/global step；
- GradScaler mapping（BF16 下可以是合法的 disabled-state 空 mapping）；
- 每个 rank 的 Python、NumPy、torch CPU/CUDA 主进程 RNG；
- weighted sampler epoch、精确的下一个 local offset 和 sampler 配置；
- 与 raw 属于同一次保存的 EMA 0.999 shadow。

raw 与 EMA 必须具有一致的 `checkpoint_pair_id`、源 run、step/global step、
world size、训练 profile ID、canonical data/model architecture contract hash
和训练合约
`crop384_db48_ddp_scratch_50k_ws128_acc4_data_weightedpdb_only_v1_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_q20_u30_p50_v8`。
严格入口还要求当前
16×8 拓扑、world size 128 与 checkpoint 保存的 world size 及 128 份 rank-state
一致；本训练合约不允许在 resume 时改变物理节点拆分。

以下情况会直接失败，不会退化成只加载模型参数：

- legacy checkpoint 或缺少严格 `resume_contract`；
- raw 缺少 optimizer、scheduler、step、GradScaler、rank RNG 或 sampler state；
- raw/EMA 不是同一次保存，或者 step、EMA decay、训练合约不匹配；
- 当前 world size、worker 数或 sampler 配置与 checkpoint 不匹配；
- 请求 `load_params_only`、跳过任意训练状态或关闭 strict load。

正式 resume 的终点仍是绝对 50,000 次 optimizer update。比如从 `999.pt`
恢复，剩余更新数是 49,000，而不是再运行 50,000 次更新。

### Resume 一步 smoke

下面的命令完整加载并校验 checkpoint pair，只做一次更新且不写新 checkpoint：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=protenix_v2_scratch_crop384_db48_ddp_resume_smoke1 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<source_run> \
RESUME_STEP=999 \
RESUME_SMOKE_STEPS=1 \
bash resume_crop384_db48_ddp_scratch_50k.sh \
  > "protenix_v2_scratch_crop384_db48_ddp_resume_smoke1.node${LOG_NODE_ID}.log" 2>&1
```

严格 resume 可保证 sampler cursor 与各 rank 主进程 RNG 恢复，但
`num_dl_workers=8` 时无法 checkpoint PyTorch DataLoader worker 内部 RNG 和已经
prefetch 的随机增强状态。因此它不承诺中断前后 worker 数据增强逐 bit 连续。若实验
必须保证这一级别的连续性，需要从首次训练起使用可恢复的确定性 worker feature seed
方案；不能在中途仅通过 resume 参数补回。

更完整的 checkpoint 合约、成功日志字段和失败条件见
[严格断点续训说明](crop384_db48_ddp_strict_resume_cn.md)。

数据集比例、两级采样、world-size/epoch-size 边界、有效 loss 系数和官方分阶段
课程见
[数据集与 loss 决策记录](crop384_db48_dataset_loss_decision_cn.md)。
