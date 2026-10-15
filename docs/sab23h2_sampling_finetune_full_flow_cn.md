# SAb23H2 Sampling Finetune 代码级实现文档

> **范围提示：**本文主要解释当前 `Protenix_train2` 代码路径。已于 2026-06-03 实际执行的 `sampling_immune_heavy` run 是由 wrapper 转到 `Protenix_train` 运行的，其实际 MSA cutoff 为 16384，不是本文中当前 `Protenix_train2` 默认的 2048。该已执行 run 的权威配置、起始 checkpoint 来源链、文件哈希和完成状态见 [`sampling_immune_heavy_20260603_executed_run_cn.md`](sampling_immune_heavy_20260603_executed_run_cn.md)。

> 本文面向首席工程师、首席算法研究员和后续优化 partner。目标不是复述 AF3/Protenix 背景，而是把当前 SAb23H2 sampling finetune 的真实实现拆到可以审查、复现和修改的粒度：入口脚本如何解析为配置，数据如何按概率进入 batch，模型每个主要层如何改变张量，loss 如何形成梯度，训练系统和评估系统之间有哪些错配。

## 0. 文档边界和写法

本文基于当前工作目录 `Protenix_train2` 下的源码、脚本和既有 SAb23H2 评估脚本整理。对方 `/root/Supplementary.md` 的优点是技术主线清楚：先定义数据管线，再讲模型模块、训练目标和评估；它有公式和算法结构，但刻意不暴露仓库级实现细节。我们这份文档采用相同的主线顺序，但每个结论都尽量绑定到真实文件、函数、字段、配置和运行语义。

本文只区分三类信息：

| 标签 | 含义 |
|---|---|
| 代码事实 | 当前源码或脚本直接决定的行为，例如 `run_sab23h2_sampling_finetune.sh:4` 硬编码了另一个 `repo_dir`。 |
| 实验解释 | 对这些实现选择的技术含义，例如当前 loss 更偏 sampling/oracle-best，而不是 ranking top-1。 |
| 待决策项 | 下一轮优化前必须确认或选择的工程/算法 knob，例如是否修正 repo 路径、是否加入 confidence-only 阶段。 |

正文不保留泛泛的尾部材料、模板化话术或为字数服务的章节。如果某个信息不能帮助定位实现、解释指标或决定下一步实验，就不放入正文。

## 1. Resolved Experiment Manifest

### 1.1 一句话定义本次 run

当前 `run_sab23h2_sampling_finetune.sh` 定义的是一个 sampling-focused continuation finetune：从已有 `999_ema_0.999.pt` 权重重新以 step 0 开始训练，使用 `weightedPDB_before210930_v20260101`、`sabdab-train`、`distill-sabdab` 的混合数据，通过低学习率、短训练步数、密集 EMA checkpoint 保存来寻找能提高 SAb23H2 oracle-best DockQ 的 checkpoint。

它不是新模型结构训练，也不是 ranking/confidence-only 训练。脚本没有覆盖 `loss.weight.*`，没有加入 DockQ/interface loss，也没有直接优化 `ranking_score`。真正改变的是数据混合、训练步长、学习率、训练时 recycle/diffusion 计算预算，以及 checkpoint 选择方式。

### 1.2 入口链路

真实调用链是：

```text
run_sab23h2_sampling_finetune.sh
  -> finetune_demo.sh
  -> torchrun ./runner/train.py
  -> parse base/data/model_type configs
  -> AF3Trainer(configs)
  -> Protenix.forward(mode="train")
  -> ProtenixLoss.forward(mode="train")
```

关键文件和职责：

| 文件 | 职责 |
|---|---|
| `run_sab23h2_sampling_finetune.sh:4-73` | sampling finetune wrapper；设置 base checkpoint、训练集、采样权重、学习率、步数、DDP 环境变量，并转交给 `finetune_demo.sh`。 |
| `finetune_demo.sh:17-217` | 激活环境、设置数据根、检查路径、拼接 `torchrun ./runner/train.py` 的 CLI override。 |
| `runner/train.py:2101-2157` | 两轮解析配置：先拿 `model_name`，再合并 `model_configs[model_name]`，最后再次应用 CLI override。 |
| `runner/train.py:1025-1041` | `AF3Trainer.__init__` 主流程：环境、基础目录、日志、模型、loss、data、checkpoint 加载。 |
| `protenix/model/protenix.py:850-934` | `Protenix.forward`，训练时决定随机 recycle 次数并进入 `main_train_loop`。 |
| `protenix/model/loss.py:1802-1897` | `ProtenixLoss.forward`，预计算 label mask/distance，再聚合 diffusion、distogram、confidence 相关 loss。 |

### 1.3 当前最重要的路径风险

代码事实：`Protenix_train2/run_sab23h2_sampling_finetune.sh:4` 把 `repo_dir` 写死为：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train
```

不是当前文档所在的：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2
```

`run_sab23h2_sampling_checkpoint_eval_queue.sh:4` 也写死到 `Protenix_train`，但 `run_sab23h2_infer_checkpoint.sh:4` 默认 `REPO_DIR` 是 `Protenix_train2`。这意味着如果两份目录代码不完全一致，训练、推理、本文分析可能来自不同源码。下一轮正式实验前必须把 dry-run 输出、训练日志中的 `repo_dir`、git commit/worktree 状态放进 run manifest。

### 1.4 关键有效配置

| 参数 | 当前有效值 | 来源和语义 |
|---|---:|---|
| `model_name` | `protenix-v2` | `finetune_demo.sh:149`。触发 `configs_model_type.py:51-88` 的 v2 覆盖。 |
| base checkpoint | `/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt` | `run_sab23h2_sampling_finetune.sh:5`。 |
| normal checkpoint path | 同 base checkpoint | `LOAD_CHECKPOINT_PATH`，`run_sab23h2_sampling_finetune.sh:23`。 |
| EMA checkpoint path | 同 base checkpoint | `LOAD_EMA_CHECKPOINT_PATH`，`run_sab23h2_sampling_finetune.sh:24`。active model 和 EMA shadow 都从 EMA 权重初始化。 |
| `load_params_only` | `True` | `configs_base.py:42-47`，脚本未覆盖。只加载参数，不恢复 optimizer/scheduler/step。 |
| `train_sets` | 默认 `weightedPDB_before210930_v20260101,sabdab-train,distill-sabdab` | `run_sab23h2_sampling_finetune.sh:9`、`finetune_demo.sh:94,171`。 |
| `train_sample_weights` | 必须由 `TRAIN_SAMPLE_WEIGHTS` 传入 | `run_sab23h2_sampling_finetune.sh:8,28`。没有该变量脚本退出。 |
| `lr` | 默认 `3e-6` | `run_sab23h2_sampling_finetune.sh:10,29`，覆盖 base `0.0018`。 |
| `max_steps` | 默认 `2000` | `run_sab23h2_sampling_finetune.sh:11,30`。 |
| `checkpoint_interval` | 默认 `250` | `run_sab23h2_sampling_finetune.sh:12,31`。由于保存用 zero-based `self.step` 命名，第 250 个 update 保存为 `249_ema_0.999.pt`。 |
| `dtype` | `bf16` | `finetune_demo.sh:150`。训练 autocast 用 bf16，GradScaler 不启用。 |
| `diffusion_batch_size` | `48` | `protenix-v2` 预设是 64，`finetune_demo.sh:153` 再覆盖为 48。 |
| `train_crop_size` | `768` | `finetune_demo.sh` 显式传入 `--train_crop_size 768`。比 base 默认 256 大很多。 |
| `model.N_cycle` | `4` | `protenix-v2` 预设是 10，`finetune_demo.sh:167` 覆盖为 4。训练时还会在 1 到 4 间按 step 伪随机采样。 |
| `sample_diffusion.N_step` | `20` | `protenix-v2` 预设是 200，`finetune_demo.sh:168` 覆盖为 20。主要影响训练 mini-rollout；训练 diffusion objective 本身是单步 denoise，不跑 20-step sampler。 |
| train MSA cutoff | `2048` | `finetune_demo.sh:175-178` 覆盖 `data.msa.sample_cutoff.train=2048`。 |
| eval | 训练内完全关闭 | `run_sab23h2_sampling_finetune.sh:32-36` 设置 `eval_interval=0`、`eval_at_end=false`、`eval_only=false`。 |
| EMA decay | `0.999` | `finetune_demo.sh:162`。保存时生成 `{step}_ema_0.999.pt`。 |

### 1.5 配置合并顺序

`runner/train.py:2101-2157` 的配置解析不是简单读一个 yaml。顺序是：

1. 导入 `configs_base` 和 `configs_data`，形成 base config。
2. 第一次 `parse_configs` 只为了拿到 CLI 中的 `model_name`。
3. 用 `model_configs[model_name]` 深度合并，例如 `protenix-v2` 把 `c_z` 从 128 改为 256，把 `model.N_cycle` 预设为 10，把 template blocks 从 0 改为 2。
4. 第二次 `parse_configs` 再应用所有 CLI，因此 `finetune_demo.sh` 的 `--model.N_cycle 4`、`--diffusion_batch_size 48`、`--sample_diffusion.N_step 20` 优先级最高。

这解释了一个容易讲错的点：`protenix-v2` 预设里确实有 10 recycle、64 diffusion samples、200 inference steps，但本次 sampling finetune 训练不是这个计算预算；它使用 v2 模型宽度和模块数，同时用 CLI 降低训练预算。

## 2. Trainer Runtime Contract

### 2.1 初始化顺序和副作用

`AF3Trainer.__init__` 的顺序是：

```text
init_env()
init_basics()
init_log()
init_model()
init_loss()
init_data()
try_load_checkpoint()
```

| 阶段 | 关键行为 |
|---|---|
| `init_env()` | 根据 `DIST_WRAPPER.world_size` 初始化 NCCL；device 取 local rank；如果不是 deterministic seed，rank 间 seed 会偏移。见 `runner/train.py:1176-1224`。 |
| `init_model()` | 构造 `Protenix(configs)`，DDP 时用 `DistributedDataParallel(..., static_graph=True)`；如果 `ema_decay > 0`，先 `EMAWrapper.register()`；再创建 optimizer 和 scheduler。见 `runner/train.py:1236-1301`。 |
| `init_loss()` | 构造 `ProtenixLoss`、`SymmetricPermutation`、`LDDTMetrics`。见 `runner/train.py:1226-1234`。 |
| `init_data()` | 调用 `get_dataloaders(configs)`。训练 dataloader 的 batch size 固定是 1。见 `runner/train.py:1303-1312`、`dataloader.py:328-350`。 |
| `try_load_checkpoint()` | 先加载 EMA checkpoint 到 EMA wrapper，再加载 normal checkpoint 到 active model。当前两个 path 相同。见 `runner/train.py:1348-1433`。 |

### 2.2 checkpoint 加载语义

`load_params_only=True` 是当前 sampling finetune 的关键语义。`runner/train.py:1395-1414` 只有在 `load_params_only=False` 时才恢复 optimizer、scheduler 和 step。本次脚本没有覆盖该字段，所以：

```text
999_ema_0.999.pt 只提供参数初始化
optimizer state 不恢复
scheduler state 不恢复
trainer.step 从 0 开始
warmup 从 0 开始
保存文件名从 249/499/... 开始
```

实验解释：这不是“从第 999 步继续跑到第 2999 步”的训练，而是“用第 999 步 EMA 权重作为初始化，重新跑一个 2000 update 的短 finetune”。如果我们讨论学习率曲线、EMA half-life 或 checkpoint 编号，必须按这个语义解释。

### 2.3 optimizer 和 scheduler

当前默认 `configs.adam.use_adamw=False`，因此 `protenix/utils/training.py:73-115` 会创建 `torch.optim.Adam`。由于 `finetune_params_with_substring` 默认为 `[""]`，`param_names` 会被置为 `None`，optimizer param group 包含全模型参数，学习率取 `configs.lr`。这意味着当前不是“只微调某些层”，而是全模型低学习率微调。

Scheduler 路径由 `runner/train.py:1281-1301` 决定：只有 `finetune_params_with_substring[0]` 非空才用 `FinetuneLRScheduler`。当前不满足，所以走 `get_lr_scheduler(configs, optimizer)`。base 默认 `lr_scheduler="af3"`，`AlphaFold3LRScheduler` 在 warmup 内使用：

```text
lr(step) = step / warmup_steps * lr
```

脚本把 `warmup_steps=2000`、`max_steps=2000`，所以整个短 finetune 几乎都在 warmup 区间。若 `lr=3e-6`，最后一个 update 附近才接近目标学习率。这是保守 continuation 的重要性质：名义学习率是 `3e-6`，平均实际学习率更低。

### 2.4 AMP、梯度和保存

训练步在 `runner/train.py:1764-1870`：

1. 设置 model train。
2. 根据 `dtype=bf16` 进入 CUDA autocast。
3. `model_forward` 产生 `pred_dict`。
4. `get_loss` 调 `ProtenixLoss`；`skip_amp.loss=True` 时 loss 计算退出 AMP。
5. bf16/fp32 下显式检查 loss 是否 NaN/Inf；非有限时把 loss 替换为零 dummy loss。
6. `loss / iters_to_accumulate` backward。
7. 到 update step 后执行 gradient clipping、optimizer step、scheduler step、zero grad。
8. `run` 主循环在 update step 后更新 EMA。

保存逻辑在 `runner/train.py:1314-1346` 和 `runner/train.py:1944-2089`。保存条件用 `(self.step + 1) % checkpoint_interval == 0` 判断，但文件名用当前 `self.step`，然后才 `self.step += 1`。因此默认 steps `249,499,749,999,1249,1499,1749,1999` 是正确的 checkpoint 文件名，而不是 off-by-one 的错误输出。

### 2.5 分布式初始化伪代码

当前分布式训练是标准 PyTorch DDP，不是 ZeRO/FSDP。环境变量由 launcher 或 AIStudio 注入，代码通过 `DIST_WRAPPER` 读取 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`LOCAL_WORLD_SIZE`。关键实现点是：rank seed 默认不同，sampler seed 默认相同；前者让模型内随机增强/dropout/rand recycle 在 rank 间不同，后者让 DDP sampler 先生成同一条 global sampled list 再按 rank 切开。

```python
def init_env(configs):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    if cuda_available():
        device = cuda(local_rank)
        torch.cuda.set_device(device)
    else:
        device = cpu()

    if world_size > 1:
        timeout = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=timeout),
        )

    if configs.deterministic_seed:
        rank_seed = configs.seed
    else:
        # 每个 rank 一个稳定但不同的 seed。
        h = sha256(f"({configs.seed},{rank},init_seed)")
        rank_seed = int(h, 16) % 2**32

    seed_everything(
        seed=rank_seed,
        deterministic=configs.deterministic,
    )

    return device
```

这个设计的后果：

1. 每个 rank 的模型随机性不同：random recycle depth、dropout、diffusion noise、crop 内部随机选择等不会完全一致。
2. 每个 rank 的 dataloader index 切片互斥：`DistributedWeightedSampler` 用相同 `seed+epoch` 生成同一条 sampled list，再按 rank stride。
3. 如果 `deterministic_seed=True`，rank seed 会相同，但 sampler 仍按 rank 切样本；这种模式更利于 debug，不一定是默认训练最佳设置。

### 2.6 DDP model、optimizer 和训练主循环伪代码

模型初始化：

```python
def init_model(configs, device):
    raw_model = Protenix(configs).to(device)

    if world_size > 1:
        model = DistributedDataParallel(
            raw_model,
            find_unused_parameters=configs.find_unused_parameters,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=True,
        )
    else:
        model = raw_model

    if configs.ema_decay > 0:
        ema_wrapper = EMAWrapper(
            model,
            decay=configs.ema_decay,
            mutable_param_keywords=configs.ema_mutable_param_keywords,
        )
        ema_wrapper.register()

    optimizer = get_optimizer(
        configs=configs,
        model=model,
        param_names=configs.get("finetune_params_with_substring", [""]),
    )

    if is_finetune_param_subset(configs.finetune_params_with_substring):
        scheduler = FinetuneLRScheduler(optimizer, configs, configs.finetune)
    else:
        scheduler = get_lr_scheduler(configs, optimizer)

    return model, raw_model, optimizer, scheduler, ema_wrapper
```

训练主循环：

```python
def run_training_loop():
    maybe_evaluate_if_eval_first()
    use_ema = hasattr(self, "ema_wrapper")

    while self.step < configs.max_steps:
        for batch in train_dl:
            is_update_step = ((global_step + 1) % iters_to_accumulate == 0)
            is_last_step = ((step + 1) == configs.max_steps) and is_update_step

            need_log = ((step + 1) % log_interval == 0) and is_update_step
            need_save = ((step + 1) % checkpoint_interval == 0) and is_update_step
            need_eval = eval_interval > 0 \
                     and ((step + 1) % eval_interval == 0) \
                     and is_update_step

            batch = to_device(batch, device)
            batch_summary = summarize_batch(batch)
            log_msa_debug_batch(batch, batch_summary, mode="train")

            train_info = train_step(batch)

            train_dataset_window_stats.add(batch_summary, train_info["loss_dict"])
            train_runtime_window_stats.add(
                rank=rank,
                step=step,
                global_step=global_step,
                batch_summary=batch_summary,
                train_info=train_info,
                cuda_memory=get_cuda_memory_stats(device),
            )

            if use_ema and is_update_step:
                ema_wrapper.update()

            if need_log or is_last_step:
                metrics = train_metric_wrapper.calc()
                dataset_stats = train_dataset_window_stats.calc()
                runtime_stats = train_runtime_window_stats.calc()
                write_rank_runtime_jsonl(...)
                if rank == 0:
                    write_train_metrics_jsonl(metrics, dataset_stats, runtime_stats)
                    update_run_status(status="running")

            if need_save or is_last_step:
                save_checkpoint()
                if use_ema:
                    ema_wrapper.apply_shadow()
                    save_checkpoint(ema_suffix=f"_ema_{ema_decay}")
                    ema_wrapper.restore()

            if should_run_evaluation(need_eval, is_last_step, eval_at_end):
                evaluate()

            global_step += 1
            if is_update_step:
                step += 1

            if step >= max_steps:
                break
```

单步训练：

```python
def train_step(batch):
    model.train()

    with autocast(dtype=configs.dtype):
        batch, model_log = model_forward(batch, mode="train")
        loss, loss_dict, _ = get_loss(batch, mode="train")

    if configs.dtype in ["bf16", "fp32"] and not isfinite(loss):
        log_numeric_error("nan_or_inf_loss", batch, loss_dict)
        loss = tensor(0.0, requires_grad=True, device=device)

    scaler.scale(loss / iters_to_accumulate).backward()

    if is_update_step:
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_if_needed(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    return {
        "loss": detach(loss),
        "loss_dict": loss_dict,
        "grad_norm": grad_norm,
        "N_cycle": model_log.get("N_cycle"),
    }
```

DDP 讨论里要强调 `static_graph=True`：如果某个分支在不同 iteration 或不同 rank 上改变参数使用集合，DDP 可能出错；因此当前实现尽量让 trunk、diffusion、confidence、distogram 在训练路径里稳定出现。若后续想做 loss/head dropout、冻结部分模块、按 dataset 跳过某些 head，需要重新验证 DDP static graph。

### 2.7 训练规模、卡数和有效样本量

Supplementary 给出的原始 AF3 训练资源口径是 256 张 A100，训练阶段表中给出每阶段样本数和耗时：

| 阶段 | training samples | diffusion batch size | 训练资源/耗时 |
|---|---:|---:|---|
| Initial training | 约 20M | 48 | 约 10 days on 256 A100s |
| Fine tuning 1 | 约 1.5M | 32 | 约 3 days on 256 A100s |
| Fine tuning 2 | 约 1.5M | 32 | 约 5 days on 256 A100s |
| Fine tuning 3 | 约 1.8M | 32 | 约 2 days on 256 A100s |

原始训练还明确写了 model training batch size 为 256，并在 diffusion module 上使用比 trunk 更大的 diffusion batch。这个信息需要和当前 run 对比，而不是只写学习率和数据集。

当前 SAb23H2 sampling finetune 的资源口径：

| 项 | 当前 run |
|---|---:|
| GPU 数 | 32 张卡，用户确认；脚本侧由 AIStudio/torchrun 的 `NNODES`、`NPROC_PER_NODE=gpu`、`WORLD_SIZE/RANK` 组织。 |
| DDP per-rank dataloader batch | 1 个 cropped complex。 |
| gradient accumulation | `iters_to_accumulate=1`，所以每个 optimizer update 消费约 `32` 个 cropped complexes。 |
| max updates | `max_steps=2000`。 |
| nominal cropped complex 数 | `32 * 2000 = 64,000` 个 DDP 样本位点，实际去重数取决于 replacement sampler。 |
| diffusion training samples | 每个 crop 内部 `diffusion_batch_size=48`，所以约 `64,000 * 48 = 3,072,000` 个 noisy denoise samples。 |
| mini-rollout samples | 每个 crop 训练中 no-grad `N_sample_mini_rollout=1`、`N_step_mini_rollout=20`，用于 permutation/confidence，不是主 diffusion loss 的反传步数。 |
| checkpoint density | 每 250 updates 保存一次 normal + EMA，约 8 个 EMA candidate checkpoints。 |

解释时要注意两个“batch”不是同一个概念：

1. DDP/global batch：32 张卡每张 1 个 crop，所以一个 optimizer update 聚合 32 个 cropped complexes 的梯度。
2. Diffusion batch：每个 cropped complex 内部生成 48 个 noisy coordinates，共享同一份 trunk 表示，反传进 diffusion denoiser 和最后 recycle trunk。

因此我们当前 run 不是接近 Supplementary 的大规模训练复现，而是 32 卡、短程、密集 checkpoint 的 targeted continuation finetune。它的数据口径和模型/loss 方法沿用同类 AF3/Protenix 训练机制，但训练规模比原始 256 A100、千万级 sample 训练小很多。

## 3. 数据管线和有效采样分布

### 3.1 数据集组合

当前默认训练集来自 `run_sab23h2_sampling_finetune.sh:9` 和 `finetune_demo.sh:94`：

```text
weightedPDB_before210930_v20260101,sabdab-train,distill-sabdab
```

| 数据集 | 配置位置 | 角色 |
|---|---|---|
| `weightedPDB_before210930_v20260101` | `configs_data.py:272-293` | 通用结构能力来源，使用 2021-09-30 前的 weighted PDB index。 |
| `sabdab-train` | `configs_data.py:187-226` | 真实 SAbDab 训练集合，提供免疫复合物和抗体相关链路。 |
| `distill-sabdab` | `configs_data.py:146-185` | distillation SAbDab 相关集合，补充免疫结构/接口分布。 |

三者都继承 weighted PDB 风格的 row sampler 和 crop 配置：`sampler_type="weighted"`，`crop.method_weights=[0.2,0.4,0.4]`，`crop.crop_size=train_crop_size`。本次脚本把 `train_crop_size` 覆盖为 768，所以 crop 后 token 上限显著大于 base 默认 256。

#### 3.1.1 Supplementary 中原始训练数据口径

按用户给定的讨论口径，我们当前 finetune 使用的数据类型和训练方法与 `/root/Supplementary.md` 描述的 AF3 训练数据口径一致：先按 dataset-level weight 选数据集，再在该数据集内部按 weighted 或 uniform 选 example，最后对 example 做 structural crop。本文不把 Supplementary 原文照搬为“竞争方工程实现”，而是把它翻译成我们当前代码可执行的数据 contract。

Supplementary 的原始训练混合数据可以组织为下表：

| 数据源 | 内容 | dataset-level weight | dataset 内采样 |
|---|---|---:|---|
| Weighted PDB | PDB ground-truth structures | 0.5 | weighted |
| Disordered protein PDB distillation | 带 unresolved protein 的 PDB distillation | 0.02 | weighted |
| Protein monomer distillation | MGnify 长蛋白单体预测 | 0.495 | uniform |
| Short protein monomer distillation | MGnify 4-200 aa 短蛋白单体预测 | 0.005 | uniform |
| RNA distillation | Rfam RNA 单体 AF3 预测 | 0.05 | uniform |
| Transcription factor negatives | MGnify protein + random DNA negative complexes | 0.011 | uniform |
| Transcription factor positives | JASPAR protein-DNA positive predictions | 0.021 | uniform |

这里有一个需要和 partner 讲清楚的点：Supplementary 文字说 structural datasets mixture，但表里把 transcription factor positive/negative 作为 finetuning 时启用的额外数据源列出。因此讨论“原始训练数据”时不能只说 PDB + distillation，还必须说明 DNA motif / transcription factor 数据是在 fine tuning 阶段打开的。

另一个需要避免误读的点：上表的 weight 是 Supplementary 给出的 dataset 配置权重口径，不应机械理解为每个训练阶段同时把所有行激活后直接相加。训练阶段会开关 transcription factor 数据、改变 disordered PDB distillation 的权重，并在当前 active datasets 上形成有效采样分布。我们当前脚本也是同一逻辑：`TRAIN_SAMPLE_WEIGHTS` 只对 `data.train_sets` 中实际启用的集合生效，再由 `WeightedMultiDataset` 归一化。

原始训练数据规模方面，Supplementary 明确披露和未明确披露的信息要分开写：

| 数据源 | Supplementary 明确规模 |
|---|---:|
| Weighted PDB | 表中给了 weight 和采样公式，但未在 2.5 段直接披露总 row/example 数。 |
| Protein monomer distillation + short protein monomer distillation | 两者合计约 41,000,000 structures。 |
| Disordered protein PDB distillation | 约 25,000 structures。 |
| RNA distillation | 约 65,000 structures after filtering。 |
| Transcription factor positives | 16,439 protein-DNA complexes，1,165 unique protein sequences。 |
| Transcription factor negatives | 2.5.2 给了构造流程，但在当前可读段落未见明确总 N。 |

不能用我们本地 `weightedPDB_before210930_v20260101` 的 2026 index 行数倒推 DeepMind 原始 Weighted PDB 规模；两者 cutoff、index row 定义、过滤和 release 版本都不完全等价。讨论时可以说“我们当前 Protenix weighted PDB finetune index 是多少”，但不要把它写成 AF3 原始训练 Weighted PDB 的官方规模。

Weighted PDB 的内部权重公式是：

```text
w(row) proportional to
    beta(type) / cluster_size
  * (alpha_prot * n_prot + alpha_nuc * n_nuc + alpha_ligand * n_ligand)

beta_chain = 0.5
beta_interface = 1
alpha_prot = 3
alpha_nuc = 3
alpha_ligand = 1
```

我们代码里的 `get_sample_weights(... sampler_type="weighted")` 和 `calc_weights_for_df` 正是这个口径的工程化版本，只是多除了一项 `pdb_sorted_entity_id_member_num`，用于避免同一 PDB/entity pair 被多个 row 重复放大。换句话说，当前 finetune 不是“均匀抽 SAbDab/PDB”，而是沿用 AF3 weighted PDB 的 redundancy-reduction 和 molecule-type stratification 思路。

#### 3.1.2 Supplementary 中训练阶段口径

Supplementary 的原始 AF3 训练分四段：

| 阶段 | crop size | 初始化 | 数据/loss 特征 |
|---|---:|---|---|
| Initial training | 384 | random | 结构、distogram 训练；diffusion batch size 48；约 20M samples。 |
| Fine tuning 1 | 640 | initial training | disordered PDB distillation weight 降到 0.01；非 protein 的 masked diffusion 设置改变；diffusion batch size 32；约 1.5M samples。 |
| Fine tuning 2 | 768 | fine tuning 1 | 打开 transcription factor distillation；disordered PDB distillation weight 回到 0.02；diffusion batch size 32；约 1.5M samples。 |
| Fine tuning 3 | 768 | fine tuning 2 | 去掉 structure/distogram loss，训练 PAE head，max chains 到 50；约 1.8M samples。 |

我们当前 SAb23H2 run 的定位应写成：从已有 EMA checkpoint 出发，在 `train_crop_size=768`、`diffusion_batch_size=48`、`N_cycle=4`、短程 2000 updates 的设置下，对同类 weighted PDB / antibody-related real / antibody-related distillation 数据做 sampling-focused continuation finetune。它不是从随机初始化重跑 Supplementary 四阶段，但数据选择、weighted sampling、crop、MSA/template、loss 组合都按同一类 AF3/Protenix 训练机制执行。

#### 3.1.3 当前 finetune 数据集实际规模

以下规模来自当前本地 index 文件统计：

| 数据集 | index path | rows | unique PDB | unique cluster | chain/interface rows | token min/median/p95/max |
|---|---|---:|---:|---:|---|---|
| `weightedPDB_before210930_v20260101` | `/mnt/project_abd/public_databases/protenix.20260101/indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz` | 1,417,835 | 166,168 | 267,080 | chain 850,823 / interface 567,012 | 4 / 796 / 5,151 / 66,060 |
| `sabdab-train` | `/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix/indices/sabdab-train.csv` | 51,436 | 8,812 | 454 | chain 26,015 / interface 25,421 | 125 / 533 / 1,359 / 3,666 |
| `distill-sabdab` | `/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix/indices/distill-sabdab.csv` | 19,682 | 3,383 | 2 | chain 9,953 / interface 9,729 | 222 / 449 / 566 / 1,463 |

分子类型分布也要说明，因为它决定 weighted row 公式中的 `alpha` 是否会偏向某类 row：

| 数据集 | dominant mol-pair rows |
|---|---|
| `weightedPDB_before210930_v20260101` | intra-protein 346,273；intra-ligand 278,821；intra-ion 202,942；protein-protein 181,854；ligand-protein 164,055；nuc-protein 34,617。 |
| `sabdab-train` | intra-protein 26,015；protein-protein interface 25,421。 |
| `distill-sabdab` | intra-protein 9,953；protein-protein interface 9,729。 |

这里有一个风险点：`distill-sabdab` 当前 `cluster_id_nunique=2`，如果继续使用 weighted PDB 的 `1/cluster_size` 机制，它的 row-level distribution 主要由 row 类型和 entity-pair duplicate 修正决定，而不是细粒度 cluster 去冗余。这个现象需要和数据构建方确认：是 distillation index 本来只有两个粗 cluster，还是 cluster 字段在导出时没有填到足够细。

### 3.2 两级采样公式

训练样本概率不是简单把三个 CSV 拼起来均匀抽。`protenix/data/pipeline/dataset.py:1201-1274` 先为每个 train set 创建 `BaseSingleDataset` 和 row weights，再交给 `WeightedMultiDataset` 合并。

设第 `d` 个 dataset 的 dataset-level weight 为 `S_d`，该 dataset 内第 `i` 行的 row weight 为 `w_{d,i}`。`WeightedMultiDataset.init_datasets` 的实际合并权重是：

```text
global_weight(d,i) = S_d * w_{d,i} / sum_j w_{d,j}
P(d,i) = global_weight(d,i) / sum_{d,i} global_weight(d,i)
       = S_d * w_{d,i} / sum_j w_{d,j} / sum_d S_d
```

所以如果 `TRAIN_SAMPLE_WEIGHTS=0.4,0.4,0.2`，三个 dataset 的名义 mass 是 40/40/20；每个 dataset 内部再按 row weight 分配。`configs_data.py` 中某些 dataset 自带的 `sample_weight` 字段不直接参与这条训练链路，真正生效的是 `data.train_sampler.train_sample_weights` 和每个 dataset 的 `sampler_configs.sampler_type`。

### 3.3 row-level weightedPDB 权重

row weight 由 `dataset.py:1095-1152` 计算。核心逻辑是：

1. 对 interface 行，按 `pdb_id(+assembly_id)+sorted(entity_1,entity_2)` 去重，避免同一 PDB/entity pair 重复支配训练。
2. `cluster_size` 是同一个 `cluster_id` 下 unique entity-pair 的数量。
3. 根据样本类型取 `beta`：chain 默认 0.5，interface 默认 1。
4. 根据分子类型取 `alpha`：protein 3，nucleotide 3，ligand 1。
5. 临时权重：

```text
tmp = beta(type) * sum(alpha(mol_1_type), alpha(mol_2_type)) / cluster_size
```

6. 最终再除以同一 PDB/entity pair 的 row 数：

```text
w = tmp / pdb_sorted_entity_id_member_num
```

实验解释：这个公式会相对提升 interface、蛋白/核酸分子，降低大 cluster 和重复 entity-pair 的影响。SAbDab train 和 distill-sabdab 继承该机制时，也会在其自身 index 内做同类 reweighting。

### 3.4 sampler 和 DDP 切片

`get_dataloaders` 在 `dataloader.py:303-373` 构建训练 dataloader：

| 场景 | sampler |
|---|---|
| 单机/单进程 | `WeightedSampler(weights, num_samples=epoch_size, replacement=True)` |
| DDP | `DistributedWeightedSampler(weights, num_samples=epoch_size, replacement=True)` |

两者都调用 `torch.multinomial` 做 replacement 抽样。`epoch_size` 默认 10000，不是 dataset 长度；每个 epoch 抽 10000 个 global indices。`IterDataLoader.__iter__` 每轮会 `sampler.set_epoch(epoch)`，seed 是 `seed + epoch`，保证每个 epoch 的 replacement 抽样不同。

DDP sampler 的实现是：每个 rank 用同一个 seed 生成同一条全局 sampled list，然后用 `indices[rank:total_size:num_replicas]` 切片。若 `epoch_size` 不能被 `world_size` 整除，`num_samples_per_replica=ceil(epoch_size/world_size)`，但 `__len__` 返回 `epoch_size // world_size`，这里存在可观测长度语义不一致风险。当前文档不判断是否已导致 hang，但下一轮应记录每 rank 实际迭代数。

### 3.5 失败重采样改变有效分布

`BaseSingleDataset.__getitem__` 在 `dataset.py:356-383` 最多尝试 10 次 `process_one`。失败时会保存错误信息；如果 `random_sample_if_failed=True`，就随机换一个 index 继续。多个训练集默认开启该选项。

这意味着名义采样概率 `P(d,i)` 不是最终有效分布。高失败率 dataset 或高失败率 row 会被悄悄替换，替换样本来自同一个 dataset 的随机 index，而不是按原 row weight 重新抽。因此有效分布会向“更容易成功解析、crop、featurize 的样本”偏移。对免疫接口任务尤其要记录失败率，因为复杂 antibody-antigen assembly、template/MSA 缺失、crop 后无足够 resolved atoms 都可能造成偏移。

### 3.6 数据侧必须新增的观测

| 观测 | 目的 |
|---|---|
| 每个 dataset 的 row weight sum/top-k/zero count | 确认 `TRAIN_SAMPLE_WEIGHTS` 之外，内部 row 权重没有异常集中。 |
| replacement 后每 1000 samples 的 dataset/type/mol_group 分布 | 确认 sampler 真实抽样 mass 是否接近设定。 |
| `random_sample_if_failed` 的原始 idx、替换 idx、异常类型、dataset_name | 量化有效分布漂移。 |
| crop method、reference chains、interface token 数、是否同时包含抗体/抗原侧 | 判断 SAbDab 样本是否真的把梯度放到 immune-other interface。 |
| MSA raw count、sampled count、template raw/selected count | 解释训练-推理 MSA/template 分布差异。 |

### 3.7 Dataloader 构建伪代码

这一节把训练数据从 config 到 batch 的入口写成伪代码。核心结论是：训练 dataloader 的 `batch_size=1` 指的是一次只取一个 cropped complex；真正的 diffusion batch 是模型内部对同一个 crop 生成 `N_sample=48` 个 noisy coordinates。

#### 3.7.1 `get_datasets`

模块职责：根据 `configs.data.train_sets` 构造多个 `BaseSingleDataset`，为每个 dataset 计算 row weights，再合成一个 `WeightedMultiDataset`。

```python
def get_datasets(configs, error_dir):
    data_config = configs.data
    assert len(data_config.train_sets) == len(data_config.train_sampler.train_sample_weights)

    train_datasets = []
    datapoint_weights = []

    for train_name in data_config.train_sets:
        cfg = data_config[train_name]

        dataset_param = {
            "name": train_name,
            **cfg["base_info"],
            "cropping_configs": cfg["cropping_configs"],
            "error_dir": error_dir,
            "msa_featurizer": get_msa_featurizer(configs, train_name, stage="train"),
            "template_featurizer": get_template_featurizer(configs, train_name, stage="train"),
            "lig_atom_rename": cfg.get("lig_atom_rename", False),
            "shuffle_mols": cfg.get("shuffle_mols", False),
            "shuffle_sym_ids": cfg.get("shuffle_sym_ids", False),
            "constraint": cfg.get("constraint", {}),
            "ref_pos_augment": data_config.get("train_ref_pos_augment", True),
            "limits": data_config.get("limits", -1),
        }

        dataset = BaseSingleDataset(**dataset_param)
        train_datasets.append(dataset)

        row_weights = get_sample_weights(
            **cfg["sampler_configs"],
            indices_df=dataset.indices_list,
        )
        datapoint_weights.append(row_weights)

    train_dataset = WeightedMultiDataset(
        datasets=train_datasets,
        dataset_names=data_config.train_sets,
        datapoint_weights=datapoint_weights,
        dataset_sample_weights=data_config.train_sampler.train_sample_weights,
    )

    test_datasets = {}
    for test_name in data_config.test_sets:
        cfg = data_config[test_name]
        test_datasets[test_name] = BaseSingleDataset(
            name=test_name,
            **cfg["base_info"],
            cropping_configs=cfg["cropping_configs"],
            msa_featurizer=get_msa_featurizer(configs, test_name, "test"),
            template_featurizer=get_template_featurizer(configs, test_name, "test"),
        )

    return train_dataset, test_datasets
```

对当前 SAb23H2 run，`train_sets` 默认是：

```text
weightedPDB_before210930_v20260101
sabdab-train
distill-sabdab
```

这三个集合都使用 `default_weighted_pdb_configs`：`sampler_type="weighted"`，`beta_chain=0.5`，`beta_interface=1`，`alpha_prot=3`，`alpha_nuc=3`，`alpha_ligand=1`，`method_weights=[0.2,0.4,0.4]`，`crop_size=train_crop_size=768`。

#### 3.7.2 `WeightedMultiDataset`

模块职责：把多个 dataset 的内部 row weight 和外部 dataset weight 合成一个全局 sampling weight vector。PyTorch sampler 看到的是一个扁平 dataset。

```python
class WeightedMultiDataset:
    def __init__(datasets, dataset_names, datapoint_weights, dataset_sample_weights):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.datapoint_weights = datapoint_weights
        self.dataset_sample_weights = tensor(dataset_sample_weights)
        self.init_datasets()

    def init_datasets():
        merged_datapoint_weights = []
        dataset_indices = []
        within_dataset_indices = []

        for d, (dataset, row_w, dataset_w) in enumerate(zip(datasets, datapoint_weights, dataset_sample_weights)):
            # 先把 dataset 内 row weight 归一化，再乘 dataset-level mass。
            row_sum = sum(row_w)
            normalized_global_w = [dataset_w * w / row_sum for w in row_w]

            merged_datapoint_weights.extend(normalized_global_w)
            dataset_indices.extend([d] * len(row_w))
            within_dataset_indices.extend(range(len(row_w)))

        self.merged_datapoint_weights = tensor(merged_datapoint_weights, dtype=float64)
        self.dataset_indices = dataset_indices
        self.within_dataset_indices = within_dataset_indices

    def __len__():
        return len(self.merged_datapoint_weights)

    def __getitem__(global_index):
        d = dataset_indices[global_index]
        local_i = within_dataset_indices[global_index]
        data = datasets[d][local_i]
        data["basic"]["dataset_name"] = dataset_names[d]
        return data
```

这解释了两层概率：

```text
P(dataset=d) = dataset_sample_weight[d] / sum(dataset_sample_weight)
P(row=i | d) = row_weight[d,i] / sum_j row_weight[d,j]
P(global row=(d,i)) = P(dataset=d) * P(row=i | d)
```

如果我们把 `TRAIN_SAMPLE_WEIGHTS` 从 `0.4,0.4,0.2` 改成 `0.25,0.5,0.25`，改变的是 dataset-level mass；每个 dataset 内部的 interface/chain、cluster、protein/nuc/ligand reweighting 仍然存在。

#### 3.7.3 `get_sample_weights` 和 weighted PDB row weight

模块职责：把 index CSV 的每一行变成 row-level sampling weight。

```python
def get_sample_weights(sampler_type, indices_df, beta_dict, alpha_dict, force_recompute_weight):
    if sampler_type == "uniform":
        return [1 / len(indices_df)] * len(indices_df)

    if sampler_type == "weighted":
        if "weights" not in indices_df.columns or force_recompute_weight:
            indices_df = calc_weights_for_df(indices_df, beta_dict, alpha_dict)
        return indices_df["weights"].astype(float32)

    raise ValueError("unknown sampler_type")
```

`calc_weights_for_df` 的核心：

```python
def calc_weights_for_df(df, beta_dict, alpha_dict):
    # 同一 PDB assembly + unordered entity pair 被视为一组。
    df["pdb_sorted_entity_id"] = pdb_id_or_assembly \
                               + min(entity_1, entity_2) \
                               + max(entity_1, entity_2)

    # 同一 entity pair 在 index 里有多少 row，用于防止重复 row 放大概率。
    member_num = groupby(df, "pdb_sorted_entity_id").size()

    # cluster_size 是一个 cluster 下 unique entity pair 数。
    cluster_size = groupby(df, "cluster_id").nunique("pdb_sorted_entity_id")

    beta = map(df["type"], {"chain": 0.5, "interface": 1.0})

    weighted_count = 0
    for mol_type, alpha in {"prot": 3, "nuc": 3, "ligand": 1}.items():
        weighted_count += alpha * (
            (df["mol_1_type"] == mol_type) + (df["mol_2_type"] == mol_type)
        )

    tmp = beta * weighted_count / cluster_size
    df["weights"] = tmp / member_num
    return df
```

与 Supplementary 的公式相比，我们代码额外显式处理了同一个 PDB/entity-pair 的多 row 重复问题；这对 SAbDab 尤其重要，因为 antibody-heavy/light/antigen 的 chain/interface row 可能在同一 complex 中形成多个相关 row。

#### 3.7.4 `WeightedSampler`

模块职责：单进程训练时，按全局 weight vector 做 replacement 抽样。

```python
class WeightedSampler:
    def __init__(weights, num_samples, replacement=True, seed=0):
        self.weights = tensor(weights, dtype=double)
        self.num_samples = num_samples          # configs.data.epoch_size，默认 10000
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

    def set_epoch(epoch):
        self.epoch = epoch

    def __iter__():
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            input=self.weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
            generator=g,
        )
        return iter(indices.tolist())

    def __len__():
        return self.num_samples
```

`IterDataLoader.__iter__` 每次开始一个 dataloader epoch 都会：

```python
sampler.set_epoch(counter)
counter += 1
return DataLoader.__iter__()
```

所以 epoch 不是遍历完整 dataset 的概念，而是“按 replacement 抽 `epoch_size` 个样本”的窗口。

#### 3.7.5 `DistributedWeightedSampler`

模块职责：DDP 训练时，每个 rank 拿到互斥的 sampled indices，避免多个 rank 重复消费同一个 global sample slot。

```python
class DistributedWeightedSampler:
    def __init__(dataset, weights, num_samples, num_replicas, rank, replacement=True, seed=0):
        self.weights = tensor(weights, dtype=double)
        self.num_samples = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        self.num_samples_per_replica = ceil(num_samples / num_replicas)
        self.total_size = self.num_samples_per_replica * num_replicas

    def set_epoch(epoch):
        self.epoch = epoch

    def __iter__():
        # 每个 rank 用同一个 seed 生成同一条 global sampled list。
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=g,
        ).tolist()

        # 再按 rank 做 stride 切片。
        local_indices = global_indices[self.rank : self.total_size : self.num_replicas]
        return iter(local_indices)

    def __len__():
        return self.num_samples // self.num_replicas
```

需要注意这个实现细节：`total_size` 可能大于 `num_samples`，但 `global_indices` 长度是 `num_samples`。切片不会 pad，只会返回实际存在的元素。若 `epoch_size % world_size != 0`，不同 rank 的实际 sample 数可能有差异；`__len__` 又返回 floor。当前默认 `epoch_size=10000`，如果 world size 能整除 10000，问题不显著；如果不能整除，应在日志里记录每 rank dataloader 迭代数，避免最后一个 rank 过早/过晚结束导致同步风险。

#### 3.7.6 `get_dataloaders`

模块职责：把 dataset 和 sampler 封装成训练/测试 dataloader。

```python
def get_dataloaders(configs, world_size, seed, error_dir):
    train_dataset, test_datasets = get_datasets(configs, error_dir)

    if world_size > 1:
        train_sampler = DistributedWeightedSampler(
            dataset=train_dataset,
            weights=train_dataset.merged_datapoint_weights,
            num_samples=configs.data.epoch_size,
            replacement=True,
            seed=seed,
        )
        train_dl = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=configs.data.num_dl_workers,
            collate_fn=collate_fn_first,
            sampler=train_sampler,
        )
    else:
        train_sampler = WeightedSampler(
            weights=train_dataset.merged_datapoint_weights,
            num_samples=configs.data.epoch_size,
            replacement=True,
            seed=seed,
        )
        train_dl = IterDataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=configs.data.num_dl_workers,
            collate_fn=collate_fn_first,
            sampler=train_sampler,
        )

    test_dls = {}
    for test_name, test_dataset in test_datasets.items():
        if world_size > 1:
            test_sampler = KeySumBalancedSampler(
                test_dataset,
                key="num_tokens",
                seed=configs.seed,
            )
        else:
            test_sampler = None

        test_dls[test_name] = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=configs.data.num_dl_workers,
            sampler=test_sampler,
            collate_fn=collate_fn_first,
        )

    return train_dl, test_dls
```

`collate_fn_first` 表示 dataloader 虽然返回 batch，但 batch size 固定为 1，因此直接取第一个样本，避免把可变 shape 的 protein complex 做 PyTorch 默认 stack。真实的 batch contract 是单样本 dict，而不是 `[B,...]` 的 dense tensor batch。

#### 3.7.7 `BaseSingleDataset.__getitem__`

模块职责：把 sampled index 转成可训练样本；失败时最多重试 10 次。

```python
def __getitem__(idx):
    for attempt in range(10):
        try:
            return process_one(idx)
        except Exception as e:
            save_error_data(idx, traceback=e)

            if random_sample_if_failed:
                # 注意：这里是同一个 BaseSingleDataset 内均匀随机换 idx，
                # 不是按 row weight 重新抽。
                idx = random.choice(range(len(indices_list)))
                continue
            else:
                raise

    return data
```

这段代码是名义分布和有效分布之间的最大偏差来源之一。假设某类 antibody-antigen complex 更容易在 crop、MSA/template、featurizer 或 label 构造中失败，那么它的有效训练概率会下降；替代样本又来自同 dataset 内均匀随机 index，而不是 weighted distribution。因此数据日志必须记录失败前 idx、替换后 idx、dataset、异常类型。

#### 3.7.8 `BaseSingleDataset.process_one`

模块职责：读取 index row、bioassembly、crop、feature、label，形成模型输入。

```python
def process_one(idx):
    sample_indice, bioassembly_dict, bioassembly_path = _get_bioassembly_data(idx)

    if sample_indice["type"] == "interface" and bioassembly_has_entity_pair_to_chain_pairs:
        # 同一个 entity pair 可能有多个具体 chain pair，训练时随机挑一个。
        chain_1, chain_2 = random.choice(entity_pair_to_chain_pairs[(entity_1, entity_2)])
        sample_indice = replace_chain_ids(sample_indice, chain_1, chain_2)

    if use_reference_chains_only:
        bioassembly_dict = keep_only_reference_chains(
            bioassembly_dict,
            chain_1=sample_indice.chain_1_id,
            chain_2=sample_indice.chain_2_id,
        )

    if shuffle_mols:
        bioassembly_dict["token_array"], bioassembly_dict["atom_array"] = shuffle_by_mol_id(...)

    if shuffle_sym_ids:
        bioassembly_dict["atom_array"] = assign_random_sym_id(...)

    if reassign_continuous_chain_ids:
        bioassembly_dict["atom_array"] = reassign_contiguous_asym_entity_sym_ids(...)

    max_entity_mol_id = bioassembly_dict["atom_array"].entity_mol_id.max()

    crop_method, token_array, atom_array, msa_features, template_features, ref_token_i = crop(
        sample_indice=sample_indice,
        bioassembly_dict=bioassembly_dict,
        crop_size=train_crop_size,
        method_weights=[0.2, 0.4, 0.4],
        max_nuc_tokens=128,
    )

    feat, label, label_full = get_feature_and_label(
        idx=idx,
        token_array=token_array,
        atom_array=atom_array,
        msa_features=msa_features,
        template_features=template_features,
        full_atom_array=bioassembly_dict["atom_array"],
        is_spatial_crop=("spatial" in crop_method.lower()),
        max_entity_mol_id=max_entity_mol_id,
    )

    basic = {
        "pdb_id": bioassembly_dict["pdb_id"] or sample_indice["pdb_id"],
        "N_asym": nunique(feat["asym_id"]),
        "N_token": feat["token_index"].shape[0],
        "N_atom": feat["atom_to_token_idx"].shape[0],
        "N_msa": feat["msa"].shape[0],
        "N_msa_prot_pair": feat["prot_pair_num_alignments"],
        "N_msa_prot_unpair": feat["prot_unpair_num_alignments"],
        "N_msa_rna_pair": feat["rna_pair_num_alignments"],
        "N_msa_rna_unpair": feat["rna_unpair_num_alignments"],
        "N_prot_atom/token": count_by_mol_type(feat, "protein"),
        "N_lig_atom/token": count_by_mol_type(feat, "ligand"),
        "N_rna_atom/token": count_by_mol_type(feat, "rna"),
        "N_dna_atom/token": count_by_mol_type(feat, "dna"),
        "chain_id": chain_ids_after_crop,
    }

    return {
        "input_feature_dict": feat,
        "label_dict": label,
        "label_full_dict": label_full,
        "basic": basic,
    }
```

当前 `basic` 已经记录 dataset、PDB、token/atom/MSA 数和分子类型统计，但没有记录 `crop_method`、`reference_token_index`、selected chain pair、interface token count。对下一轮优化，这几个字段比再看平均 loss 更关键，因为它们能直接回答：训练样本到底有没有把 768 token 预算用在 SAb23H2 关心的 antibody-antigen interface 上。

#### 3.7.9 `DataPipeline.crop`

模块职责：按 crop 配置把 full bioassembly 裁成模型可承受的 token 子图，并同步裁剪 MSA/template。

```python
def crop(one_sample, bioassembly_dict, crop_size, msa_featurizer, template_featurizer, method_weights):
    if crop_size <= 0:
        selected_indices = None
        msa = get_msa_raw_features(bioassembly_dict, selected_indices=None, msa_featurizer=msa_featurizer)
        template = get_template_raw_features(bioassembly_dict, selected_indices=None, template_featurizer=template_featurizer)
        return "no_crop", token_array, atom_array, msa, template, -1

    # 把 index row 的 chain_1/chain_2 映射到 crop 使用的 asym_id_int。
    ref_chain_indices = _map_ref_chain(one_sample, bioassembly_dict)

    cropper = CropData(
        crop_size=crop_size,
        ref_chain_indices=ref_chain_indices,
        token_array=bioassembly_dict["token_array"],
        atom_array=bioassembly_dict["atom_array"],
        method_weights=method_weights,
        contiguous_crop_complete_lig=True,
        spatial_crop_complete_lig=True,
        drop_last=True,
        remove_metal=True,
        max_nuc_tokens=128,
    )

    crop_method = cropper.random_crop_method()
    selected_token_indices, reference_token_index = cropper.get_crop_indices(crop_method)

    # MSA/template 先按 selected token 取 raw features。
    cropped_msa = get_msa_raw_features(
        bioassembly_dict,
        selected_indices=selected_token_indices,
        msa_featurizer=msa_featurizer,
    )
    cropped_template = get_template_raw_features(
        bioassembly_dict,
        selected_indices=selected_token_indices,
        template_featurizer=template_featurizer,
    )

    cropped_token_array, cropped_atom_array = cropper.crop_by_indices(selected_token_indices)

    if crop_method == "ContiguousCropping":
        assert sum(cropped_atom_array.is_resolved) > 4

    return crop_method, cropped_token_array, cropped_atom_array, cropped_msa, cropped_template, reference_token_index
```

Supplementary 中 crop 策略也是三类：contiguous、spatial、spatial interface。我们当前 weightedPDB/SAbDab/distill-SAbDab 训练集都用 `[0.2,0.4,0.4]`，即 40% nominal mass 会尝试 spatial interface crop。但最终是否成为有效 interface crop，还取决于 full assembly 是否有可用 inter-chain token、reference chain 是否正确、crop fallback 是否发生、失败重采样是否替换样本。

## 4. 从 index row 到 batch contract

### 4.1 `BaseSingleDataset.process_one` 的顺序

`dataset.py:491-628` 是单样本主流程：

```text
读取 sample index
读取 bioassembly pickle
若 interface row 有 entity_pair_to_chain_pairs，则随机选具体 chain pair
可选 reference-only / shuffle mols / shuffle sym ids / reassign chain id
DataPipeline.crop()
Featurizer.get_all_input_features()
Featurizer.get_labels()
Featurizer.get_gt_full_complex_features()
MSA/template features 合并
data_type_transform
返回 input_feature_dict / label_dict / label_full_dict / basic
```

`basic` 记录 `pdb_id`、`N_asym`、`N_token`、`N_atom`、`N_msa`、各 molecule token/atom 数等，但当前没有记录 `crop_method` 和 interface token 数。这是 sampling finetune 讨论的缺口：我们无法从训练日志直接判断 SAbDab 样本是否以 interface crop 为主。

### 4.2 crop 实现

`DataPipeline.crop` 在 `data_pipeline.py:245-360`。如果 `crop_size<=0`，不 crop；否则：

1. `_map_ref_chain()` 把 index row 中的 chain id 映射到 `asym_id_int`。
2. 创建 `CropData`。
3. `CropData.random_crop_method()` 用 `random.choices` 按 `method_weights` 选择 crop 方法。
4. 根据 token distance 和 reference chain 选 crop indices。
5. 用 crop indices 同步裁剪 token array、atom array、MSA、template。

三种 crop 方法来自 `utils/cropping.py`：

| 方法 | 默认权重 | 行为 |
|---|---:|---|
| ContiguousCropping | 0.2 | 按链和 token 顺序连续截取，适合保持局部序列上下文，但可能错过跨链接口。 |
| SpatialCropping | 0.4 | 在 reference chain resolved token 中随机选中心，取空间最近 top-k token。 |
| SpatialInterfaceCropping | 0.4 | 先找 reference chain 中距离其他 chain 小于 15A 的 interface tokens，再围绕 interface token 空间 top-k。 |

对 SAb23H2 这类 antibody-antigen benchmark，`SpatialInterfaceCropping` 是最直接把 crop 预算投到免疫接口的实现点。但是否真正命中 immune-other interface 取决于 index row 的 reference chain、bioassembly chain pair、resolved atom、crop size 和 fallback 行为。文档层面不能只写“使用 interface crop”，必须要求日志证明 crop 中同时包含两侧 interface chain，并且 interface token 在 crop 内。

### 4.3 MSA 和 template

MSA featurizer在 `dataset.py:927-965` 创建。脚本覆盖：

```text
--data.msa.max_size.train 16384
--data.msa.sample_cutoff.train 2048
--data.msa.max_size.test 16384
--data.msa.sample_cutoff.test 16384
```

`MSAModule.forward` 在模型内还会根据 training/eval 选择 `train_cutoff` 或 `test_cutoff`，并用 `sample_msa_feature_dict_random_without_replacement` 对 `msa/has_deletion/deletion_value` 采样。训练只看最多 2048 条 MSA，推理默认可看 16384 条，这会带来 MSA 深度分布差异。

Template featurizer输出模型真正使用的字段包括：

```text
template_aatype: [N_template, N_token]
template_distogram: [N_template, N_token, N_token, 39]
template_unit_vector: [N_template, N_token, N_token, 3]
template_pseudo_beta_mask: [N_template, N_token, N_token]
template_backbone_frame_mask: [N_template, N_token, N_token]
```

`TemplateEmbedder` 不读取旧 schema 里的 `template_restype/template_all_atom_*` 名字。这里应加 batch schema assertion，区分 no-template dummy 和真实 template，避免 silent feature mismatch。

### 4.4 batch 字段 contract

`BaseSingleDataset.process_one()` 返回顶层 dict：

```text
{
  "input_feature_dict": ...,
  "label_dict": ...,
  "label_full_dict": ...,
  "basic": ...
}
```

训练 `runner/train.py:1491-1545` 把 `input_feature_dict/label_dict/label_full_dict` 传给模型，再把 `input_feature_dict/pred_dict/label_dict` 传给 loss。

| 字段 | shape | 来源/用途 |
|---|---:|---|
| `token_index/residue_index/asym_id/entity_id/sym_id` | `[N_token]` | relative position encoding、pair mask、chain/interface 语义。 |
| `restype` | `[N_token,32]` | 输入 embedder 拼接进 `s_inputs`。 |
| `profile` | `[N_token,32]` | MSA profile，输入 embedder 拼接。 |
| `deletion_mean` | `[N_token]` | 输入 embedder 拼接，reshape 后是 1 维特征。 |
| `ref_pos` | `[N_atom,3]` | atom attention reference coordinate；也用于构造 local atom pair `d_lm`。 |
| `ref_mask/ref_charge/ref_space_uid` | `[N_atom]` | atom reference mask、charge、局部空间 UID。 |
| `ref_element` | `[N_atom,128]` | atom 元素 one-hot。 |
| `ref_atom_name_chars` | `[N_atom,4,64]` | atom name 字符 one-hot。 |
| `atom_to_token_idx` | `[N_atom]` | atom 到 token 的聚合和 broadcast 索引。 |
| `atom_to_tokatom_idx` | `[N_atom]` | token 内 atom 序号，用于 pLDDT/resolved head 的 atom-position 权重。 |
| `token_bonds` | `[N_token,N_token]` | 加到 pair init。 |
| `bond_mask` | `[N_atom,N_atom]` | bond loss mask；当前权重为 0，但仍可能计算。 |
| `msa/has_deletion/deletion_value` | `[N_msa,N_token]` | MSA module 输入。 |
| `template_*` | 见上节 | TemplateEmbedder 输入。 |
| `is_protein/is_ligand/is_dna/is_rna` | `[N_atom]` | MSE 原子权重、lDDT 半径、confidence target。 |
| `resolution` | scalar/`[1]` | confidence loss 是否启用的 gate。distillation 样本设为 `-1.0`。 |
| `frame_atom_index` | `[N_token,3]` | PAE frame 构造。 |
| `has_frame` | `[N_token]` | PAE/ptm/iptm 有效 token gate。 |
| `distogram_rep_atom_mask` | `[N_atom]` | distogram/PDE/contact 代表原子。 |
| `plddt_m_rep_atom_mask` | `[N_atom]` | atom lDDT target 中 polymer representative atom。 |
| `pae_rep_atom_mask` | `[N_atom]` | PAE token representative atom。 |
| `label_dict.coordinate` | `[N_atom,3]` | crop 后 GT coordinate。 |
| `label_dict.coordinate_mask` | `[N_atom]` | GT coordinate 是否存在。 |
| `label_full_dict` | full complex/crop-asym fields | chain permutation 使用；mini-rollout 时用 full/cropped-asym 信息重排 label。 |

模型前向会额外在 `input_feature_dict` 中写入：

| 字段 | shape | 生成位置 |
|---|---:|---|
| `relp` | `[N_token,N_token,139]` | `RelativePositionEncoding.generate_relp`。 |
| `d_lm` | `[n_blocks,32,128,3]` | `update_input_feature_dict`，atom-local dense pair displacement。 |
| `v_lm` | `[n_blocks,32,128,1]` | local atom pair valid mask。 |
| `pad_info` | dict | local atom pair padding/mask 信息。 |

### 4.5 Featurizer 实现细节和字段生成伪代码

这一节把 `Featurizer` 写到接近代码级别。首席们如果不看源码，至少要能从这里判断每个模型输入字段的来源、随机性、mask 语义和潜在失效点。

#### 4.5.1 `get_feature_and_label` 总入口

`BaseSingleDataset.get_feature_and_label` 负责把 crop 后的 `TokenArray/AtomArray` 转成模型可消费的 feature/label。核心顺序是：

```python
def get_feature_and_label(
    cropped_token_array,
    cropped_atom_array,
    bioassembly_dict,
    selected_token_indices,
    is_spatial_crop,
):
    # 约束类 feature 最先生成，后面会合并到 input feature。
    if constraint_pipeline_enabled:
        feature_dict.update(make_constraint_features(...))

    featurizer = Featurizer(
        cropped_token_array=cropped_token_array,
        cropped_atom_array=cropped_atom_array,
        ref_pos_augment=config.ref_pos_augment,
        lig_atom_rename=config.lig_atom_rename,
        include_discont_poly_poly_bonds=config.include_discont_poly_poly_bonds,
    )

    feature_dict.update(featurizer.get_all_input_features())
    label_dict.update(featurizer.get_labels())

    # atom_perm_list 来自 residue-level symmetry annotation，
    # 后续 atom permutation 会逐 residue 枚举这些 permutation。
    feature_dict["atom_perm_list"] = featurizer.get_atom_permutation_list()

    # label_full_dict 给 chain permutation 使用。
    # spatial crop 时可以只取 cropped asym 涉及的 full-complex 子集。
    label_full_dict, full_atom_array = featurizer.get_gt_full_complex_features(
        bioassembly_dict=bioassembly_dict,
        get_cropped_asym_only=is_spatial_crop,
    )

    if find_pocket:
        label_dict.update(make_interested_ligand_and_pocket_masks(...))

    if find_eval_chain_interface:
        label_dict.update(make_eval_interface_masks(...))

    feature_dict.update(make_dummy_features_for_missing_optional_inputs(...))
    return feature_dict, label_dict, label_full_dict
```

这里最关键的工程事实是：模型主 loss 使用 `label_dict`，而 chain permutation 使用 `label_full_dict`。如果 crop、full label、`mol_id/mol_atom_index` 三者不一致，symmetry 对齐会直接改变训练 target，且这种错误不一定能在 shape 检查里暴露。

#### 4.5.2 `get_all_input_features`

`Featurizer.get_all_input_features()` 的实际执行顺序是固定的：

```python
def get_all_input_features():
    features = {}
    features.update(get_token_features())
    features.update(get_bond_features())
    features.update(get_reference_features())
    features.update(get_extra_features())
    features.update(get_chain_perm_features())
    features.update(get_mask_features())
    return features
```

这个顺序意味着 `token_bonds` 和 `bond_mask` 是两套不同语义：

1. `token_bonds` 是 token-token feature，进入 pair 初始化。
2. `bond_mask` 是 atom-atom mask，进入 bond loss。

它们都和 bond 有关，但一个是输入特征，一个是监督 mask，不能混用。

#### 4.5.3 token features

token features 使用每个 token 的 centre atom 作为代表原子，从 `cropped_atom_array` 的 annotation 中取字段：

```python
def get_token_features():
    centre_atom_indices = token_array.get_annotation("centre_atom_index")
    centre_atoms = cropped_atom_array[centre_atom_indices]

    restype = centre_atoms.cano_seq_resname
    restype_onehot = restype_onehot_encoded(restype, num_classes=32)

    return {
        "token_index": token_array.get_values(),
        "residue_index": centre_atoms.res_id,
        "asym_id": centre_atoms.asym_id_int,
        "entity_id": centre_atoms.entity_id_int,
        "sym_id": centre_atoms.sym_id_int,
        "restype": restype_onehot,
    }
```

实现里 `restype` 的类别数是 32，来自 `STD_RESIDUES_WITH_GAP`。ligand 没有标准 residue type，通常会落到 UNK 语义。因此 ligand 的结构信息更多依赖 atom reference features、bond features、atom attention 和 tokenization，而不是 protein residue one-hot。

#### 4.5.4 reference atom features

reference features 是 diffusion atom attention 的核心输入，字段包括 `ref_pos/ref_mask/ref_element/ref_charge/ref_atom_name_chars/ref_space_uid/has_frame/frame_atom_index`。

```python
def get_reference_features():
    ref_pos = zeros([N_atom, 3])

    for uid in unique(cropped_atom_array.ref_space_uid):
        mask = cropped_atom_array.ref_space_uid == uid
        # 每个 ref_space_uid 内独立做 random_transform。
        # centralize=True，ref_pos_augment 控制是否加随机旋转平移增强。
        ref_pos[mask] = random_transform(
            cropped_atom_array.ref_pos[mask],
            apply_augmentation=ref_pos_augment,
            centralize=True,
        )

    if lig_atom_rename:
        atom_names = rename_ligand_atoms_by_element_count(atom_names)

    ref_element = one_hot(atomic_number, num_classes=128)
    ref_atom_name_chars = one_hot_each_char(atom_name, num_chars=4, alphabet=64)

    has_frame, frame_atom_index = get_token_frame(
        token_array=token_array,
        atom_array=cropped_atom_array,
        ref_mask=cropped_atom_array.ref_mask,
    )

    return {
        "ref_pos": ref_pos,
        "ref_mask": cropped_atom_array.ref_mask,
        "ref_element": ref_element,
        "ref_charge": cropped_atom_array.charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": cropped_atom_array.ref_space_uid,
        "has_frame": has_frame,
        "frame_atom_index": frame_atom_index,
    }
```

`lig_atom_rename=True` 时，ligand atom name 会按元素计数重命名，例如同一 ligand 内的碳按 `C1/C2/...`。这个设计的目的不是增强表达能力，而是降低 ligand 原始 atom name 对 label 的泄漏风险。

#### 4.5.5 token frame 构造

PAE、PTM/IPTM 和局部 frame 相关逻辑依赖 `has_frame/frame_atom_index`。不同 molecule 类型的 frame 来源不同：

```python
def get_token_frame(token):
    if token_is_protein_residue:
        frame_atoms = ["N", "CA", "C"]

    elif token_is_dna_or_rna_residue:
        frame_atoms = ["C1'", "C3'", "C4'"]

    else:
        # ligand、非标准 residue、per-atom token：
        # 以 centre atom 作为 b，在同一个 ref_space_uid 内用 KDTree 找最近邻。
        b = centre_atom
        neighbors = nearest_atoms_in_same_ref_space_uid(b)
        a, c = choose_two_nearest_neighbors(neighbors)

        if num_atoms_in_ref_space < 3:
            return has_frame=False
        if any_required_atom_ref_mask_is_false:
            return has_frame=False
        if norm(a-b) == 0 or norm(c-b) == 0:
            return has_frame=False
        if angle(a-b, c-b) <= 25 degree or >= 155 degree:
            return has_frame=False

        frame_atoms = [a, b, c]

    return has_frame=True, frame_atom_index=atom_indices(frame_atoms)
```

这段实现的讨论价值在于：ligand 和非标准 residue 的 frame 是近邻几何构造出来的，不是化学键规则构造出来的。若 ligand frame 质量影响 PAE/ranking 或 pocket 评估，应该优先检查 `has_frame` 覆盖率、近邻角度过滤比例、以及 ligand tokenization 是否让 ref_space 内可用 atom 少于 3。

#### 4.5.6 bond features

`get_bond_features()` 从 `cropped_atom_array.bonds.as_array()` 读取 atom bond，再映射到 token-token bond：

```python
def get_bond_features():
    token_bonds = zeros([N_token, N_token])
    atom_to_token = build_atom_to_token_idx(token_array)

    for atom_i, atom_j, bond_type in cropped_atom_array.bonds.as_array():
        token_i = atom_to_token[atom_i]
        token_j = atom_to_token[atom_j]

        # 默认丢弃标准 polymer-polymer bond，避免把主链连续性重复编码。
        if is_standard_polymer_polymer_bond(atom_i, atom_j):
            if not include_discont_poly_poly_bonds:
                continue

        # 保留 polymer-ligand、ligand-ligand、非标准相关 bond。
        token_bonds[token_i, token_j] = 1
        token_bonds[token_j, token_i] = 1

    return {"token_bonds": token_bonds}
```

这里有一个容易被忽略的点：`token_bonds` 是对称矩阵，进入 `linear_no_bias_token_bond` 后加到 pair init。它不是 bond length，也不是 bond type 多分类。若要增强 ligand chemistry，现有实现最直接的改造点是把 bond order/type、aromatic、formal charge 或 torsion 相关信息扩成更丰富的 pair feature，而不是只调 loss。

#### 4.5.7 extra、chain permutation 和 mask features

`get_extra_features()` 提供 atom 到 token 的索引和 molecule 类型：

```python
def get_extra_features():
    return {
        "atom_to_token_idx": atom_to_token_idx,       # [N_atom]
        "atom_to_tokatom_idx": atom_to_tokatom_idx,   # [N_atom]
        "is_protein": atom_array.is_protein,
        "is_ligand": atom_array.is_ligand,
        "is_dna": atom_array.is_dna,
        "is_rna": atom_array.is_rna,
        "resolution": resolution_or_minus_one,
    }
```

`get_chain_perm_features()` 给 chain permutation 使用，不直接等价于 `asym_id/entity_id`：

```python
def get_chain_perm_features():
    return {
        "mol_id": atom_array.mol_id,
        "mol_atom_index": atom_array.mol_atom_index,
        "entity_mol_id": atom_array.entity_mol_id,
    }
```

`mol_id` 是 permutation 的 chain identity，`mol_atom_index` 是同一 molecule 内的 atom 顺序。chain permutation 构造 indices 时会用 predicted crop 中的 `mol_atom_index` 去 full label 对应链里取 atom，因此这两个字段一旦在 crop/full label 之间错位，会造成错误 label 对齐。

`get_mask_features()` 生成下游 loss 和 head 使用的 representative atom mask：

```python
def get_mask_features():
    centre_atom_mask = mask_of_token_centre_atoms()

    return {
        "pae_rep_atom_mask": centre_atom_mask,
        "plddt_m_rep_atom_mask": build_plddt_representative_mask(),
        "distogram_rep_atom_mask": build_distogram_representative_mask(),
        "modified_res_mask": atom_array.modified_res_mask,
        "bond_mask": get_ligand_polymer_bond_mask(atom_array),
    }
```

`bond_mask` 是 atom-atom 对称矩阵，主要覆盖 ligand-polymer bond。当前 bond loss 权重为 0，但如果后续要打开 bond loss，必须先确认这个 mask 的覆盖范围是否只包含我们想约束的键。

#### 4.5.8 label 和 atom permutation list

label 很直接：

```python
def get_labels():
    return {
        "coordinate": cropped_atom_array.coord,
        "coordinate_mask": cropped_atom_array.is_resolved,
    }
```

atom permutation list 来自每个 atom 的 `res_perm` annotation：

```python
def get_atom_permutation_list():
    out = []
    for atom in cropped_atom_array:
        # res_perm 是字符串，以下划线分隔，例如 "0_1_2"。
        perm = [int(x) for x in atom.res_perm.split("_")]
        out.append(perm)
    return out
```

这个字段的含义是 residue 内 atom 对称置换候选，而不是全局 atom permutation。后续 atom permutation 会先按 `ref_space_uid` 把 atom 切成 residue，再对每个 residue 枚举候选置换。

### 4.6 MSA 和 template 的在线 featurization 伪代码

#### 4.6.1 MSA feature 生成

`MSAFeaturizer.make_msa_features()` 的输入是 full bioassembly、crop 后 token indices、以及 `entity_to_asym_id_int` 映射。实现不是简单读一个 MSA 文件，而是按 crop 中出现的 asym/entity 逐条组装。

```python
def make_msa_features(bioassembly_dict, selected_indices, entity_to_asym_id_int):
    atom_array = bioassembly_dict["atom_array"]
    token_array = bioassembly_dict["token_array"]

    centre_atom_indices = token_array[selected_indices].centre_atom_index \
        if selected_indices is not None else token_array.centre_atom_index
    selected_asym_ids = set(atom_array[centre_atom_indices].asym_id_int)

    msa_meta = {}
    for entity_id, asym_ids in entity_to_asym_id_int.items():
        if not any(aid in selected_asym_ids for aid in asym_ids):
            continue

        sequence = bioassembly_dict["sequences"].get(entity_id)
        chain_type = map_entity_poly_type(entity_id)

        if sequence is None:
            # ligand/non-polymer fallback，长度按 atom 数补 X。
            sequence = "X" * atom_count_for_entity(entity_id)

        if chain_type == "RNA":
            paired_msa = []
            unpaired_msa = rna_mgr.fetch_msas(sequence, pdb_id="", chain_type=RNA)

        elif chain_type == "PROTEIN":
            paired_msa, unpaired_msa = prot_mgr.fetch_msas(
                sequence=sequence,
                pdb_id=bioassembly_dict["pdb_id"],
                paired_dbs=prot_p_dbs,
                unpaired_dbs=prot_np_dbs,
            )

        else:
            paired_msa, unpaired_msa = [], []

        msa_meta[entity_id] = {
            "sequence": sequence,
            "chain_type": chain_type,
            "paired_msa": paired_msa,
            "unpaired_msa": unpaired_msa,
        }

    std_idxs = map_crop_tokens_to_standard_sequence_indices(
        token_array=token_array,
        atom_array=atom_array,
        selected_indices=selected_indices,
    )

    msa_feat = FeatureAssemblyLine(...).assemble(msa_meta, std_idxs).to_dict()
    return select_keys(msa_feat, [
        "msa",
        "has_deletion",
        "deletion_value",
        "deletion_mean",
        "profile",
        "prot_pair_num_alignments",
        "prot_unpair_num_alignments",
        "rna_pair_num_alignments",
        "rna_unpair_num_alignments",
    ])
```

MSA deletion value 的数学实现是：

```text
deletion_value = arctan(deletion_matrix / 3.0) * (2 / pi)
has_deletion = clip(deletion_matrix, 0, 1)
```

这两个字段进入 `MSAModule`。训练时 `MSAModule` 又会按 `sample_cutoff.train=2048` 从 `msa/has_deletion/deletion_value` 中无放回采样，因此 raw MSA 最大 16384 不等于模型每步实际使用 16384。

#### 4.6.2 Template feature 生成

`TemplateFeaturizer` 在 train/eval 阶段有不同的日期和采样行为：

```python
def init_template_featurizer(dataset_name, stage):
    if stage != "train":
        max_template_date = "2021-09-30"
    else:
        if "distillation" in dataset_name or "openprotein" in dataset_name:
            max_template_date = "2018-04-30"
        else:
            max_template_date = "3000-01-01"

    online_featurizer = TemplateHitFeaturizer(
        max_hits=None if stage == "train" else 4,
        max_template_date=max_template_date,
        _shuffle_top_k_prefiltered=20 if stage == "train" else None,
    )
```

生成逻辑：

```python
def make_template_features(bioassembly_dict, selected_token_indices, entity_to_asym_id_int):
    selected_asym_ids = get_selected_asym_ids_from_centre_atoms(...)
    asym_to_entity_id = restrict_entity_asym_mapping_to_selected_asym(...)
    asym_id_to_sequence = fill_missing_ligand_sequence_with_X(...)

    query_release_date = parse_release_date(bioassembly_dict) if stage == "train" else None

    drop_template = False
    if stage == "train" and template_dropout_rate > 0:
        drop_template = random_uniform() < template_dropout_rate

    template_meta_infos = {}
    seq_to_templates_cache = {}

    for asym_id, sequence in asym_id_to_sequence.items():
        chain_type = map_poly_type_to_chain_type(entity_id)

        if drop_template or chain_type != PROTEIN:
            templates = []
            raw_hit_count = 0
        else:
            if sequence in seq_to_templates_cache:
                templates = seq_to_templates_cache[sequence]
            else:
                templates = get_template(
                    pdb_id=pdb_id,
                    query_sequence=sequence,
                    sequence_uid=f"{pdb_id}_{asym_id}",
                    query_release_date=query_release_date,
                    chain_entity_type=PROTEIN,
                )
                seq_to_templates_cache[sequence] = templates

        template_meta_infos[asym_id] = {
            "entity_id": entity_id,
            "chain_id": chain_id,
            "sequence": sequence,
            "chain_entity_type": chain_type,
            "templates": templates,
        }

    copied_or_assembled = copy_same_entity_templates_when_safe(template_meta_infos)
    return reduce_template_features_to_crop(copied_or_assembled, selected_token_indices)
```

`get_template()` 内部还有 hit 过滤：

```python
def get_template(pdb_id, query_sequence, query_release_date):
    template_paths = source_mgr.fetch_template_paths(pdb_id, query_sequence)
    hhr_or_a3m = read_template_search_outputs(template_paths)

    if query_release_date is None:
        cutoff_date = max_template_date
    else:
        cutoff_date = min(max_template_date, query_release_date - DAYS_BEFORE_QUERY_DATE)

    hits = parse_hhr_if_available_else_parse_hmmsearch_a3m(hhr_or_a3m)

    valid_hits = []
    for hit in hits:
        pdb_code, chain_id = get_pdb_id_and_chain(hit)
        pdb_code = obsolete_pdbs.get(pdb_code, pdb_code)

        reject_if_release_date_after_cutoff(pdb_code, cutoff_date)
        reject_if_alignment_ratio_le_0_1(hit, query_sequence)
        reject_if_hit_sequence_is_large_subsequence_duplicate(hit, query_sequence)
        reject_if_template_sequence_shorter_than_10(hit)
        valid_hits.append(hit)

    valid_hits = sort_by_sum_probs_desc(valid_hits)
    deduped_hits = dedupe_by_hit_sequence(valid_hits)

    if stage == "train":
        shuffle_first_20_hits(deduped_hits)
        max_to_collect = min(4, randint(0, min(len(deduped_hits), 20)))
    else:
        max_to_collect = 4

    return process_hits_until_collected(max_to_collect)
```

生成给模型的 template tensor 时，会先把 atom37 转为 dense atom，再构造 AF3 风格的 pair features：

```python
def template_as_protenix_dict(template_features):
    for each_template in templates:
        dense_atom_mask, dense_atom_positions = atom37_to_dense(...)

        pseudo_beta_pos, pseudo_beta_mask = pseudo_beta_fn(
            aatype=template_aatype,
            dense_atom_positions=dense_atom_positions,
            dense_atom_masks=dense_atom_mask,
            is_ligand=is_ligand_if_available,
        )

        template_distogram = dgram_from_positions(
            pseudo_beta_pos,
            min_bin=3.25,
            max_bin=50.75,
            num_bins=39,
        ) * pseudo_beta_pair_mask

        template_unit_vector, backbone_frame_mask = compute_template_unit_vector(
            aatype=template_aatype,
            atom_positions=dense_atom_positions,
            atom_mask=dense_atom_mask,
        )

    return {
        "template_aatype": ...,
        "template_atom_mask": ...,
        "template_atom_positions": ...,
        "template_distogram": ...,
        "template_unit_vector": ...,
        "template_pseudo_beta_mask": ...,
        "template_backbone_frame_mask": ...,
        "template_release_timestamp": ...,
    }
```

`compute_template_unit_vector()` 的局部 frame 是以 CA 为原点，`C-CA` 为第一轴，`N-CA` 正交化后为第二轴，第三轴是叉乘。然后把 residue j 的 CA 相对 residue i 的 CA 的向量投到 i 的局部 frame，并归一化。这就是 template unit vector 的几何含义。

### 4.7 CropData 内部算法伪代码

#### 4.7.1 crop 总入口

`CropData` 固定有三种候选方法：

```python
cand_crop_methods = [
    "ContiguousCropping",
    "SpatialCropping",
    "SpatialInterfaceCropping",
]
method_weights = [0.2, 0.4, 0.4]
```

`get_crop_indices()` 的公共预处理：

```python
def get_crop_indices(crop_method):
    tokens = token_array.get_values()
    centre_atom_indices = token_array.centre_atom_index
    centre_atoms = atom_array[centre_atom_indices]

    chain_id = centre_atoms.asym_id_int
    token_dist_mask_1d = centre_atoms.is_resolved
    token_indices_in_ref = where(centre_atoms.asym_id_int in ref_chain_indices)
    ref_space_uid_token = atom_array.ref_space_uid[centre_atom_indices]

    atom_num_in_tokens = [len(token.atom_indices) for token in token_array]
    atom_sums = count_total_atoms_per_ref_space_uid(ref_space_uid_token, atom_num_in_tokens)

    if crop_method == "ContiguousCropping":
        selected, ref_idx = contiguous_crop(...)
        selected = limit_nuc_tokens(selected, reference_token_index=None)
        return selected, token_indices_in_ref[-1]

    else:
        token_distance = cdist(ref_chain_token_coords, all_token_coords)
        token_distance_mask = token_dist_mask_1d[ref_tokens, None] * token_dist_mask_1d[None, :]
        selected, ref_idx_in_ref = spatial_crop(..., interface_crop=(crop_method == "SpatialInterfaceCropping"))
        abs_ref_idx = token_indices_in_ref[ref_idx_in_ref]
        selected = limit_nuc_tokens(selected, reference_token_index=abs_ref_idx)
        return selected, abs_ref_idx
```

这里 `reference_token_index` 是记录 crop 中心/参考 token 的绝对 token index。当前文档建议把它写进日志，因为它能解释 spatial crop 到底围绕哪条链和哪个 residue 展开。

#### 4.7.2 连续裁剪

连续裁剪按链随机顺序迭代，每条链随机取一个连续片段：

```python
def get_continues_crop_index(tokens, chain_id, ref_space_uid_token, atom_sums, crop_size):
    unique_chain_id = unique(chain_id)
    chain_lengths = bincount(chain_id)
    chain_offsets = first_index_of_each_chain(chain_id)

    is_metal, uid_first_indices, uid_last_indices = identify_mol_type(
        ref_space_uid_token, atom_sums, chain_id, chain_lengths
    )

    chain_order = randperm(len(unique_chain_id))
    selected = []
    N_added = 0
    N_remaining = len(tokens) - count_removed_metals_if_enabled(is_metal)

    for chain_idx in chain_order:
        if N_added >= crop_size:
            break
        if remove_metal and chain_is_metal(chain_idx):
            continue

        chain_length = chain_lengths[chain_idx]
        N_remaining -= chain_length

        crop_size_min = min(chain_length, max(0, crop_size - (N_added + N_remaining)))
        crop_size_max = min(crop_size - N_added, chain_length)

        chain_crop_size = randint(crop_size_min, crop_size_max + 1)
        chain_crop_start = randint(0, chain_length - chain_crop_size + 1)

        start = chain_offset + chain_crop_start
        end = start + chain_crop_size

        if crop_complete_ligand_unstdRes:
            start, end = adjust_start_end_to_ref_space_uid_boundary(
                start, end, uid_first_indices, uid_last_indices,
                crop_size_min, N_added, crop_size, drop_last,
            )

        selected.append(arange(start, end))
        N_added += end - start

    return sort(concat(selected))
```

`identify_mol_type()` 的 metal 判定非常具体：如果某个 `ref_space_uid` 只对应一个 token，且所在 chain 长度为 1，且 atom 数为 1，则认为是 metal/ion。这个判断影响 `remove_metal=True` 时的裁剪结果。

#### 4.7.3 空间裁剪和界面空间裁剪

空间裁剪的核心是“先选 reference token，再按距离取 top-k”：

```python
def get_spatial_crop_index(
    tokens,
    chain_id,
    token_distance,
    token_distance_mask,
    reference_chain_id,
    ref_space_uid_token,
    crop_size,
    interface_crop,
):
    if interface_crop:
        reference_token_indices = get_interface_token(
            chain_id=chain_id,
            reference_chain_id=reference_chain_id,
            token_distance=token_distance,
            token_distance_mask=token_distance_mask,
            interface_minimal_distance=15,
        )

        if len(reference_token_indices) < 1 and len(reference_chain_id) == 1:
            # fallback：单 reference chain 没找到跨链接触时，用该链所有 resolved token。
            reference_token_indices = where(token_distance_mask.any(axis=-1))
    else:
        reference_token_indices = where(token_distance_mask.any(axis=-1))

    assert len(reference_token_indices) > 0
    reference_token_idx = random_choice(reference_token_indices)
    assert token_distance_mask[reference_token_idx].any()

    d = token_distance[reference_token_idx]
    d = where(token_distance_mask[reference_token_idx], d, inf)

    # 加微小 index 噪声打破距离相同的 tie。
    d = d + arange(len(d)) * 1e-3
    selected = topk_smallest(d, k=min(crop_size, len(tokens)))
    selected = sort(selected)

    if crop_complete_ligand_unstdRes:
        selected = drop_ref_space_uid_if_not_fully_selected(selected)

    return selected, reference_token_idx
```

`get_interface_token()` 的 mask 是：

```text
distance(token_i, token_j) < 15A
and chain_id_i != chain_id_j
and token_i belongs to reference_chain_id
and both tokens have valid distance
```

因此 `SpatialInterfaceCropping` 只保证 reference token 来自界面候选，不保证最后 top-k 一定覆盖我们关心的 antibody-antigen 双方，也不保证不会被大 chain 的近邻 token 占满。对 SAb23H2 优化而言，日志至少应统计：

1. `crop_method`。
2. `reference_token_index`、reference token 所在 chain 和 molecule type。
3. crop 内 antibody token 数、antigen token 数、ligand token 数。
4. interface token 是否真的存在 fallback。
5. crop 后 `has_frame`、`coordinate_mask`、`bond_mask` 的有效比例。

#### 4.7.4 核酸 token 上限

当前 CropData 还有一个 `max_nuc_tokens` 保护：

```python
def limit_nuc_tokens(selected_token_indices, reference_token_index=None):
    if max_nuc_tokens <= 0:
        return selected_token_indices

    nuc_positions = positions_in_selected_where_token_is_dna_or_rna()
    if len(nuc_positions) <= max_nuc_tokens:
        return selected_token_indices

    if reference_token_index is not None:
        ref_coord = centre_coord(reference_token_index)
        nuc_coords = centre_coords(selected_nuc_tokens)
        keep_nuc = nearest_by_distance(nuc_coords, ref_coord, max_nuc_tokens)
    else:
        keep_nuc = first_nuc_positions(max_nuc_tokens)

    keep all non-nuc tokens and keep_nuc
    return sort(selected_token_indices[keep_mask])
```

如果我们后续把 benchmark 扩到 RNA/DNA complex，这个限制会改变实际 crop distribution。当前抗体 benchmark 通常不是主风险，但仍应该在 dataset summary 中记录是否触发。

## 5. 模型总前向和梯度路径

### 5.1 训练模式真实执行序列

`Protenix.forward(mode="train")` 的高层顺序：

```text
relp = RelativePositionEncoding.generate_relp(input_feature_dict)
update_input_feature_dict(input_feature_dict)  # d_lm/v_lm/pad_info
N_cycle = RandomState(current_step).randint(1, self.N_cycle + 1)
main_train_loop(...)
```

`main_train_loop` 在 `protenix.py:657-848`：

```text
s_inputs, s, z = get_pairformer_output(...)
可选 prepare diffusion shared cache: pair_z, p_lm, c_l
no_grad mini-rollout -> coordinate_mini
coordinate_mini 驱动 label permutation
ConfidenceHead(coordinate_mini) -> plddt/pae/pde/resolved
sample_diffusion_training(... N_sample=diffusion_batch_size) -> coordinate/noise_level
DistogramHead(z) -> distogram
permute_diffusion_sample_to_match_label(stage="train")
return pred_dict, log_dict
```

正常训练的 `pred_dict` 字段：

| 字段 | shape | 梯度/用途 |
|---|---:|---|
| `coordinate_mini` | `[1,N_atom,3]` | no-grad mini rollout 输出，detach；用于 label permutation 和 confidence target。 |
| `plddt` | `[1,N_atom,50]` | confidence head 输出；loss 权重 `1e-4`。 |
| `pae` | `[1,N_token,N_token,64]` | confidence head 输出；当前 loss 权重 0，但 inference ranking 依赖 PAE 概率。 |
| `pde` | `[1,N_token,N_token,64]` | confidence head 输出；loss 权重 `1e-4`。 |
| `resolved` | `[1,N_atom,2]` | confidence head 输出；loss 权重 `1e-4`。 |
| `coordinate` | `[48,N_atom,3]` | diffusion training denoised 坐标；主导 diffusion loss。 |
| `noise_level` | `[48]` | 每个 diffusion sample 的 sigma；MSE/bond per-sample scale 使用。 |
| `distogram` | `[N_token,N_token,64]` | distogram logits；loss 权重 `0.03`。 |

### 5.2 recycle 梯度策略

`get_pairformer_output` 中每个 recycle cycle 使用：

```python
with torch.set_grad_enabled(
    self.training and not self.train_confidence_only and cycle_no == (N_cycle - 1)
):
    ...
```

也就是说训练时随机抽 `N_cycle` 次 recycle，但只有最后一个 cycle 记录 autograd。早期 cycles 产生 recycling state，但不反传。脚本把 `model.N_cycle` 覆盖为 4，所以每个 step 的 trunk 计算深度在 1 到 4 间伪随机变化，最后一轮有梯度。推理脚本默认 `N_CYCLE=10`，因此训练和评估的 recycle 深度不同。

实验解释：这是一种显存/速度折中。它保留多 recycle 状态对最终 trunk 的影响，但只让最后 cycle 承担梯度。对于短 finetune，若 SAb23H2 评估主要使用 10 recycle，训练 1-4 recycle 是否足够适配 interface sampling，需要用 checkpoint-level oracle/top-k 曲线验证。

### 5.3 总前向伪代码

下面伪代码覆盖 `Protenix.forward(mode="train")`、`main_train_loop` 和 loss 前的 `pred_dict` 形成。它不是可运行代码，而是为了让读者能逐行对照 `protenix/model/protenix.py` 的真实调用顺序。

```python
def protenix_forward_train(input_feature_dict, label_dict, label_full_dict, current_step):
    # 1. no-grad feature construction
    input_feature_dict["relp"] = RelativePositionEncoding.generate_relp(
        asym_id=input_feature_dict["asym_id"],
        residue_index=input_feature_dict["residue_index"],
        entity_id=input_feature_dict["entity_id"],
        token_index=input_feature_dict["token_index"],
        sym_id=input_feature_dict["sym_id"],
    )                                            # [N_token, N_token, 139]

    update_input_feature_dict(input_feature_dict)
    # writes d_lm/v_lm/pad_info for sequence-local atom attention

    # 2. deterministic-per-step random recycle depth
    n_cycle = RandomState(current_step).randint(1, model.N_cycle + 1)
    # sampling finetune: model.N_cycle=4, so n_cycle in {1,2,3,4}

    # 3. trunk
    s_inputs, s, z = get_pairformer_output(
        input_feature_dict=input_feature_dict,
        n_cycle=n_cycle,
        train_confidence_only=False,
    )
    # s_inputs: [N_token,449], s: [N_token,384], z: [N_token,N_token,256]

    # 4. optional shared diffusion cache
    if configs.enable_diffusion_shared_vars_cache:
        pair_z = diffusion_conditioning.prepare_cache(input_feature_dict["relp"], z)
        p_lm, c_l = diffusion_atom_encoder.prepare_cache(
            ref_features=input_feature_dict,
            d_lm=input_feature_dict["d_lm"],
            v_lm=input_feature_dict["v_lm"],
            pad_info=input_feature_dict["pad_info"],
        )
    else:
        pair_z, p_lm, c_l = None, None, None

    # 5. no-grad mini rollout for permutation and confidence input
    with no_grad():
        coordinate_mini = sample_diffusion(
            denoise_net=diffusion_module,
            s_inputs=detach(s_inputs),
            s_trunk=detach(s),
            z_trunk=detach(z),
            pair_z=detach(pair_z),
            p_lm=detach(p_lm),
            c_l=detach(c_l),
            N_sample=1,
            N_step=20,
        )
    pred_dict = {"coordinate_mini": detach(coordinate_mini)}

    # 6. symmetric label permutation driven by mini rollout
    label_dict, label_full_dict, perm_log = symmetric_permutation.permute_label_to_match_mini_rollout(
        pred_dict={"coordinate": pred_dict["coordinate_mini"]},
        label_dict=label_dict,
        label_full_dict=label_full_dict,
        input_feature_dict=input_feature_dict,
    )

    # 7. confidence head; trunk inputs are detached inside ConfidenceHead by default
    plddt, pae, pde, resolved = confidence_head(
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        x_pred_coords=pred_dict["coordinate_mini"],
    )
    pred_dict.update({"plddt": plddt, "pae": pae, "pde": pde, "resolved": resolved})

    # 8. diffusion training objective: one-step denoise for 48 noisy samples
    x_gt_aug, x_denoised, sigma = sample_diffusion_training(
        noise_sampler=TrainingNoiseSampler(p_mean=-1.2, p_std=1.5, sigma_data=16),
        denoise_net=diffusion_module,
        label_dict=label_dict,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        N_sample=configs.diffusion_batch_size,   # sampling finetune: 48
    )
    pred_dict["coordinate"] = x_denoised          # [48,N_atom,3]
    pred_dict["noise_level"] = sigma              # [48]

    # 9. pair-level distogram logits from final trunk z
    pred_dict["distogram"] = distogram_head(z)    # [N_token,N_token,64]

    # 10. train-stage diffusion sample permutation normally only logs
    pred_dict, sample_perm_log = symmetric_permutation.permute_diffusion_sample_to_match_label(
        pred_dict=pred_dict,
        label_dict=label_dict,
        label_full_dict=label_full_dict,
        stage="train",
    )

    return pred_dict, label_dict, {**perm_log, **sample_perm_log}
```

### 5.4 Trunk recycle 伪代码

`get_pairformer_output` 是模型主体最重要的状态机。早期 recycle no-grad，最后一个 recycle 有梯度。

```python
def get_pairformer_output(input_feature_dict, n_cycle):
    s_inputs = input_embedder(input_feature_dict)      # [N,449]

    z_constraint = constraint_embedder(input_feature_dict)  # normally None in this run

    s_init = linear_sinit(s_inputs)                    # [N,384]
    z_init = linear_zinit_i(s_init)[:, None, :] + linear_zinit_j(s_init)[None, :, :]
    z_init = z_init + relpos(input_feature_dict["relp"])
    z_init = z_init + input_feature_dict["token_bonds"][..., None]
    if z_constraint is not None:
        z_init = z_init + z_constraint

    s = zeros_like(s_init)
    z = zeros_like(z_init)
    pair_mask = token_mask[:, None] * token_mask[None, :]

    for cycle_no in range(n_cycle):
        grad_enabled = training and (cycle_no == n_cycle - 1)
        with set_grad_enabled(grad_enabled):
            z = z_init + recycle_z_linear(layernorm_z(z))

            template_update = template_embedder(input_feature_dict, z, pair_mask)
            z = z + template_update

            z = msa_module(
                input_feature_dict=input_feature_dict,
                z=z,
                s_inputs=s_inputs,
                pair_mask=pair_mask,
            )

            s = s_init + recycle_s_linear(layernorm_s(s))
            s, z = pairformer_stack(s=s, z=z, pair_mask=pair_mask)

    return s_inputs, s, z
```

## 6. Trunk 模型层级实现

### 6.1 v2 宽度和脚本覆盖

`configs_model_type.py:51-88` 的 `protenix-v2` 覆盖项：

| 维度/模块 | v2 有效值 |
|---|---:|
| `c_z` | 256 |
| `c_s` | 384，未被 v2 覆盖 |
| InputFeatureEmbedder `c_token` | 384，未被 v2 覆盖 |
| `c_s_inputs` | 449 |
| TemplateEmbedder blocks | 2 |
| MSAModule blocks | 4 |
| MSAModule `c_m` | 128 |
| PairformerStack blocks | 48 |
| DiffusionModule internal `c_token` | 768 |
| ConfidenceHead blocks | 4 |

这里必须避免混淆：输入 embedder 的 atom-to-token 输出 `a` 是 384 维，拼上 restype/profile/deletion 后 `s_inputs=449`；diffusion module 内部 token channel `c_token=768` 是扩散 denoiser 的 token 表示，不是 input embedder 的 `c_token`。

### 6.2 InputFeatureEmbedder

文件：`protenix/model/modules/embedders.py:28-121`。

输入字段：

```text
atom_to_token_idx
ref_pos/ref_charge/ref_mask/ref_atom_name_chars/ref_element
d_lm/v_lm/pad_info
restype/profile/deletion_mean
```

步骤：

1. `AtomAttentionEncoder(has_coords=False)` 用 atom reference features 和 local atom pair features 得到 token 聚合表示 `a [...,N_token,384]`。
2. `restype [N_token,32]`、`profile [N_token,32]`、`deletion_mean [N_token,1]` 与 `a` 拼接。
3. 输出：

```text
s_inputs [..., N_token, 384+32+32+1] = [..., N_token, 449]
```

AtomAttentionEncoder 内部在 `transformer.py:597-949`：

| 子步骤 | 说明 |
|---|---|
| atom single `c_l` | `ref_pos`、`arcsinh(ref_charge)`、`ref_mask/ref_element/ref_atom_name_chars` 线性投影相加。 |
| atom pair `p_lm` | `d_lm`、inverse distance、valid mask `v_lm` 线性投影，并用 `pad_info["mask_trunked"]` mask。 |
| local AtomTransformer | 3 blocks、4 heads、local query/key window 默认 32/128。 |
| atom -> token | `linear_q(q_l)` 后按 `atom_to_token_idx` mean 聚合成 token 表示。 |

### 6.3 RelativePositionEncoding

文件：`embedders.py:124-213`。

`generate_relp` 生成 139 维 pair feature：

```text
relative residue index one-hot: 2*r_max + 2 = 66
relative token index one-hot:   2*r_max + 2 = 66
same_entity bit:                1
relative chain/sym one-hot:     2*s_max + 2 = 6
total:                          139
```

随后 `RelativePositionEncoding.forward` 用线性层投到 `c_z=256`，加到 pair 初始化 `z_init`。`generate_relp` 在 `Protenix.forward` 前置阶段 no-grad 执行，因为这些是离散索引派生特征，不需要梯度。

### 6.4 s/z 初始化

`get_pairformer_output` 在 `protenix.py:170-229`：

```text
s_inputs = InputFeatureEmbedder(...)
s_init = Linear(s_inputs)                         # [N,384]
z_init = Linear(s_init)_i + Linear(s_init)_j      # [N,N,256]
z_init += RelativePositionEncoding(relp)
z_init += token_bonds[...,None]
z_init += optional z_constraint
s = zeros_like(s_init)
z = zeros_like(z_init)
```

recycle 的状态注入方式：

```text
z = z_init + linear_z_recycle(LN(z))
s = s_init + linear_s_recycle(LN(s))
```

recycling 线性层在初始化时 zero-init，因此刚开始 continuation 时 recycle state 的增量受 checkpoint 参数控制，而不是随机大扰动。

### 6.5 TemplateEmbedder

文件：`pairformer.py:977-1156`。

v2 中 `TemplateEmbedder.n_blocks=2`。如果 batch 没有 `template_aatype` 或 blocks 小于 1，则返回 0。否则每个 template：

1. 输入当前 `z` 先 LayerNorm 后线性投到 template hidden `c=64`。
2. 拼接 template pair 特征：

```text
template_distogram: 39
template_pseudo_beta_mask: 1
template_restype_i: 32
template_restype_j: 32
template_unit_vector: 3
template_backbone_frame_mask: 1
```

3. 用 `multichain_mask = asym_id[:,None] == asym_id[None,:]` 乘在 distogram、unit vector、mask 上。也就是说 template pair 几何只保留同链信息，跨链 template 几何被置零。
4. `linear_z(z) + linear_a(template_features)` 后跑一个 `PairformerStack(c_s=0,c_z=64,n_blocks=2)`。
5. 多 template 求平均，ReLU 后线性回 `c_z=256`，加到 trunk pair `z`。

对 antibody-antigen interface 的含义：template 主要加强链内结构和局部几何先验；跨链 docking/interface 关系不能依赖 template unit vector 直接注入，仍主要由 MSA/pairformer/diffusion 学到。

### 6.6 MSAModule

文件：`pairformer.py:701-974`。

MSA module 输入：

```text
msa: [N_msa,N_token]，32类 one-hot
has_deletion: [N_msa,N_token]
deletion_value: [N_msa,N_token]
s_inputs: [N_token,449]
z: [N_token,N_token,256]
```

步骤：

1. 若没有 `msa` 或 MSA 维度异常，直接返回原 `z`。
2. 按 training/eval 选择 cutoff。当前训练 cutoff 是 2048，推理/eval 默认可到 16384。
3. 对 `msa/has_deletion/deletion_value` 随机无放回采样。
4. `msa` one-hot 到 32 维，拼上 deletion 2 维，总 34 维。
5. `Linear(34 -> c_m=128)`，再加 `Linear(s_inputs -> c_m)`。
6. 进入 4 个 `MSABlock`：
   - `OuterProductMean(m)` 加到 `z`。
   - 非最后 block：`MSAStack` 用 pair bias 对 MSA 行加权平均，再 transition。
   - 每个 block 都跑 `PairformerBlock(c_s=0)` 更新 pair。
7. 最后 block 返回 `m=None,z`，避免 MSA 表示被后续使用。

训练时为了 DDP `static_graph=True`，`MSAStack.forward` 会把 MSA 行 pad 到 `msa_max_size` 后按 `msa_chunk_size=2048` chunk/checkpoint，再截回原 MSA 行数。这是一个工程稳定性实现，不是算法层面的必要操作。

### 6.7 PairformerStack

文件：`pairformer.py:61-359`。

主 PairformerStack 是 48 blocks，输入 `s [N,384]`、`z [N,N,256]`。每个 `PairformerBlock` 顺序是：

```text
TriangleMultiplicationOutgoing(z)
TriangleMultiplicationIncoming(z)
TriangleAttentionStartingNode(z)
TriangleAttentionEndingNode(z^T)
PairTransition(z)
AttentionPairBias(s, z)
SingleTransition(s)
```

v2 设置 `hidden_scale_up=True`，因此 triangle multiplicative hidden 会随 `c_z` 扩展，pair attention head 数也会按 `c_z/c_hidden_pair_att` 调整。脚本传入：

```text
--triangle_attention cuequivariance
--triangle_multiplicative cuequivariance
```

所以 triangle kernel 走 cuequivariance 实现；`init_env` 里还有相关环境检查。文档讨论性能或数值时，需要把 kernel 类型作为 run manifest 一部分。

### 6.8 Trunk 子模块伪代码

以下伪代码覆盖训练前向实际调用的 trunk 模块。为了便于审查，每段都保留输入/输出张量语义。

#### 6.8.1 `update_input_feature_dict`

该函数不是神经网络层，但它生成后续 atom-local attention 需要的 dense local pair 表示。

```python
def update_input_feature_dict(feat):
    # q/k both come from reference atom coordinates and local space uid
    q = {"ref_pos": feat["ref_pos"], "ref_space_uid": feat["ref_space_uid"]}
    k = {"ref_pos": feat["ref_pos"], "ref_space_uid": feat["ref_space_uid"]}

    # rearrange atom pairs into trunked local windows:
    # n_queries=32, n_keys=128
    dense = rearrange_qk_to_dense_trunk(
        q=q,
        k=k,
        n_queries=32,
        n_keys=128,
        compute_mask=True,
    )

    feat["d_lm"] = dense["ref_pos_q"] - dense["ref_pos_k"]      # [B_l,32,128,3]
    feat["v_lm"] = same_space_uid_and_valid_mask(dense)         # [B_l,32,128,1]
    feat["pad_info"] = dense["pad_info"]
    return feat
```

工程含义：`d_lm/v_lm/pad_info` 是根据 reference atom 顺序和 local window 构造的，不是 learned feature。crop 改变 atom 顺序和 token/atom 数时，这些 local windows 也随之改变。

#### 6.8.2 `InputFeatureEmbedder`

```python
def input_feature_embedder(feat):
    # AtomAttentionEncoder(has_coords=False)
    a_token, q_l, c_l, p_lm = atom_attention_encoder_ref_only(
        atom_to_token_idx=feat["atom_to_token_idx"],
        ref_pos=feat["ref_pos"],
        ref_charge=feat["ref_charge"],
        ref_mask=feat["ref_mask"],
        ref_atom_name_chars=feat["ref_atom_name_chars"],
        ref_element=feat["ref_element"],
        d_lm=feat["d_lm"],
        v_lm=feat["v_lm"],
        pad_info=feat["pad_info"],
        r_l=None,
        s=None,
        z=None,
    )                                                        # a_token: [N,384]

    deletion_mean = feat["deletion_mean"][..., None]         # [N,1]
    s_inputs = concat(
        a_token,                                             # [N,384]
        feat["restype"],                                     # [N,32]
        feat["profile"],                                     # [N,32]
        deletion_mean,                                       # [N,1]
        dim=-1,
    )                                                        # [N,449]
    return s_inputs
```

#### 6.8.3 `AtomAttentionEncoder`

这个模块在两个地方复用：InputFeatureEmbedder 用 `has_coords=False`；DiffusionModule 用 `has_coords=True`。

```python
def atom_attention_encoder(feat, r_l=None, s=None, z=None, p_lm=None, c_l=None):
    if p_lm is None or c_l is None:
        c_l = linear_ref_pos(feat["ref_pos"]) \
            + linear_ref_charge(arcsinh(feat["ref_charge"])) \
            + linear_ref_features(concat(ref_mask, ref_element, ref_atom_name_chars))
        c_l = c_l * feat["ref_mask"][..., None]              # [N_atom,128]

        p_lm = linear_d(feat["d_lm"]) * feat["v_lm"]
        p_lm = p_lm * feat["pad_info"]["mask_trunked"][..., None]
        p_lm = p_lm + linear_invd(1 / (1 + squared_norm(feat["d_lm"]))) * feat["v_lm"]
        p_lm = p_lm + linear_valid(feat["v_lm"])             # [B_l,32,128,16]

    if r_l is not None:
        # diffusion path only
        z_local = broadcast_token_pair_to_local_atom_pair(
            linear_z(layernorm_z(z)),
            atom_to_token_idx=feat["atom_to_token_idx"],
            n_queries=32,
            n_keys=128,
        )
        p_lm = p_lm[None, ...] + z_local                     # [N_sample,B_l,32,128,16]

        c_l = c_l[None, ...] + broadcast_token_to_atom(
            linear_s(layernorm_s(s)),
            feat["atom_to_token_idx"],
        )                                                    # [N_sample,N_atom,128]
        q_l = c_l + linear_r(r_l)                            # noisy coordinate injection
    else:
        q_l = clone(c_l)                                     # [N_atom,128]

    c_l_q, c_l_k = local_qk(c_l, n_queries=32, n_keys=128)
    p_lm = p_lm + linear_cl(relu(c_l_q)) + linear_cm(relu(c_l_k))
    p_lm = p_lm + small_mlp(p_lm)

    q_l = atom_transformer(q_l, c_l, p_lm)                   # local pair-bias attention
    a_token = aggregate_atom_to_token(
        relu(linear_q(q_l)),
        atom_to_token_idx=feat["atom_to_token_idx"],
        reduce="mean",
    )
    return a_token, q_l, c_l, p_lm
```

审查点：InputFeatureEmbedder 的 `a_token` 是 384 维；DiffusionModule 的 `a_token` 是 768 维，因为两个实例的 `c_token` 配置不同。

#### 6.8.4 `RelativePositionEncoding`

```python
def generate_relp(asym_id, residue_index, entity_id, token_index, sym_id):
    same_chain = asym_id[:, None] == asym_id[None, :]
    same_entity = entity_id[:, None] == entity_id[None, :]

    rel_residue = clip(residue_index[:, None] - residue_index[None, :], -r_max, r_max)
    rel_token = clip(token_index[:, None] - token_index[None, :], -r_max, r_max)
    rel_chain = clip(sym_id[:, None] - sym_id[None, :], -s_max, s_max)

    a_rel_pos = one_hot_with_out_of_range(rel_residue, bins=2 * r_max + 2)
    a_rel_token = one_hot_with_out_of_range(rel_token, bins=2 * r_max + 2)
    a_rel_chain = one_hot_with_out_of_range(rel_chain, bins=2 * s_max + 2)

    relp = concat(a_rel_pos, a_rel_token, same_entity[..., None], a_rel_chain)
    return relp                                               # [N,N,139]

def relative_position_encoding(relp):
    return linear_relp(relp)                                  # [N,N,c_z]
```

#### 6.8.5 `TemplateEmbedder`

```python
def template_embedder(feat, z, pair_mask):
    if "template_aatype" not in feat or n_template_blocks < 1:
        return 0

    z_norm = layernorm_z(z)
    multichain_mask = feat["asym_id"][:, None] == feat["asym_id"][None, :]
    updates = []

    for t in range(num_templates):
        dgram = feat["template_distogram"][t] * multichain_mask[..., None] * pair_mask[..., None]
        pseudo_beta_mask = feat["template_pseudo_beta_mask"][t] * multichain_mask * pair_mask
        unit_vector = feat["template_unit_vector"][t] * multichain_mask[..., None] * pair_mask[..., None]
        backbone_mask = feat["template_backbone_frame_mask"][t] * multichain_mask * pair_mask

        restype = one_hot(feat["template_aatype"][t], 32)
        restype_i = expand_i(restype)                         # [N,N,32]
        restype_j = expand_j(restype)                         # [N,N,32]

        template_pair_feat = concat(
            dgram,
            pseudo_beta_mask[..., None],
            restype_i,
            restype_j,
            unit_vector,
            backbone_mask[..., None],
        )                                                     # [N,N,108]

        v = linear_z(z_norm) + linear_template_feat(template_pair_feat)
        _, v = pairformer_stack_template(s=None, z=v, pair_mask=pair_mask)
        updates.append(layernorm_v(v))

    u = mean(updates)
    return linear_to_cz(relu(u))                              # [N,N,256]
```

审查点：`multichain_mask` 会把跨链 template 几何清零；这对 antibody-antigen interface 是重要限制。

#### 6.8.6 `MSAModule`

```python
def msa_module(feat, z, s_inputs, pair_mask):
    if n_blocks < 1 or "msa" not in feat or feat["msa"].dim() < 2:
        return z

    cutoff = train_cutoff if training else test_cutoff
    lower_bound = train_lowerb if training else test_lowerb
    msa_feat = sample_msa_without_replacement(
        feat_dict=feat,
        keys=["msa", "has_deletion", "deletion_value"],
        cutoff=cutoff,
        lower_bound=lower_bound,
        strategy=msa_strategy,
    )

    msa_one_hot = one_hot(msa_feat["msa"], 32)
    m = concat(msa_one_hot, msa_feat["has_deletion"], msa_feat["deletion_value"])
    m = linear_msa(m)                                         # [N_msa_sample,N,128]
    m = m + linear_s_inputs(s_inputs)                         # broadcast over MSA rows

    for block_idx in range(n_blocks):
        is_last = block_idx == n_blocks - 1
        m, z = msa_block(m, z, pair_mask, is_last=is_last)

    return z

def msa_block(m, z, pair_mask, is_last):
    z = z + outer_product_mean(m)
    if not is_last:
        m = msa_stack(m, z)
    _, z = pairformer_block(s=None, z=z, pair_mask=pair_mask)
    if is_last:
        return None, z
    return m, z

def msa_stack(m, z):
    if training:
        m_pad = pad_msa_rows_to_static_size(m, msa_max_size)
        update = chunk_checkpoint(msa_pair_weighted_averaging, m_pad, z)
        m = dropout_add_rowwise(m, update[:m.shape[-3]], p=0.15)
        m_pad = pad_msa_rows_to_static_size(m, msa_max_size)
        trans = chunk_checkpoint(msa_transition, m_pad)
        m = m + trans[:m.shape[-3]]
    else:
        for msa_chunk in chunks(m, msa_chunk_size):
            msa_chunk += msa_pair_weighted_averaging(msa_chunk, z)
            msa_chunk += msa_transition(msa_chunk)
    return m
```

#### 6.8.7 `MSAPairWeightedAveraging`

```python
def msa_pair_weighted_averaging(m, z):
    m_norm = layernorm_m(m)
    v = reshape_heads(linear_v(m_norm))                       # [M,N,H,C]
    gate = sigmoid(reshape_heads(linear_gate(m_norm)))         # [M,N,H,C]

    pair_bias = linear_z(layernorm_z(z))                       # [N,N,H]
    weights = softmax(pair_bias, dim=-2)                       # attention over source token

    averaged = einsum("ijh,mjhc->mihc", weights, v)
    out = linear_out(flatten_heads(gate * averaged))           # [M,N,c_m]
    return out
```

#### 6.8.8 `PairformerBlock` 和 `PairformerStack`

```python
def pairformer_stack(s, z, pair_mask):
    for block in checkpointed_blocks(pairformer_blocks, blocks_per_ckpt=1):
        s, z = pairformer_block(s, z, pair_mask)
    return s, z

def pairformer_block(s, z, pair_mask):
    z = dropout_add_rowwise(
        z,
        triangle_multiplication_outgoing(z, mask=pair_mask, implementation="cuequivariance"),
        p=0.25,
    )
    z = dropout_add_rowwise(
        z,
        triangle_multiplication_incoming(z, mask=pair_mask, implementation="cuequivariance"),
        p=0.25,
    )
    z = dropout_add_rowwise(
        z,
        triangle_attention_starting_node(z, mask=pair_mask, implementation="cuequivariance"),
        p=0.25,
    )

    z_t = transpose_token_pair(z)
    z_t = dropout_add_rowwise(
        z_t,
        triangle_attention_ending_node(z_t, mask=transpose(pair_mask), implementation="cuequivariance"),
        p=0.25,
    )
    z = transpose_token_pair(z_t)
    z = z + pair_transition(z)

    if s is not None:
        s = s + attention_pair_bias(a=s, s=None, z=z)
        s = s + single_transition(s)
    return s, z
```

#### 6.8.9 `AttentionPairBias`

该模块在 Pairformer、DiffusionTransformer、AtomTransformer 中复用。区别在于是 token-level full attention 还是 atom-level local attention。

```python
def attention_pair_bias(a, s, z, n_queries=None, n_keys=None):
    if has_s:
        q = adaptive_layernorm(a, s)
    else:
        q = layernorm(a)

    if cross_attention_mode:
        kv = adaptive_layernorm(a, s) if has_s else layernorm(a)
    else:
        kv = q

    if n_queries is not None and n_keys is not None:
        # local atom attention: z is [B_l,32,128,c_z]
        bias = linear_z(layernorm_z(z))                       # [B_l,32,128,n_heads]
        bias = move_head_dim_to_attention_layout(bias)
        out = local_attention(q, kv, trunked_attn_bias=bias, n_queries=32, n_keys=128)
    else:
        # token attention: z is [N,N,c_z]
        bias = linear_z(layernorm_z(z))                       # [N,N,n_heads]
        bias = move_head_dim_to_attention_layout(bias)
        out = standard_attention(q, kv, attn_bias=bias)

    if has_s:
        out = sigmoid(linear_gate_from_s(s)) * out
    return out
```

## 7. Diffusion denoiser 实现

### 7.1 训练 diffusion 不是 20-step rollout

本次脚本设置 `--sample_diffusion.N_step 20`。这个值影响训练中的 no-grad mini-rollout，用于 permutation/confidence；真正的 diffusion training objective 由 `sample_diffusion_training` 实现，是对每个 noisy sample 做一次 denoise：

```text
x_gt -> random rigid augmentation
sigma ~ LogNormal(p_mean,p_std)*sigma_data
x_noisy = x_gt_aug + sigma * epsilon
x_denoised = D_theta(x_noisy, sigma, trunk conditioning)
loss(x_denoised, x_gt)
```

所以不能把训练 objective 描述成“用 20 步 diffusion 反传”。20-step sampler在训练里只用于 mini-rollout，且 no-grad。

### 7.2 训练噪声和 sample 维

`TrainingNoiseSampler` 在 `generator.py:26-61`：

```text
sigma = exp(randn * p_std + p_mean) * sigma_data
默认 p_mean=-1.2, p_std=1.5, sigma_data=16
```

`sample_diffusion_training` 在 `generator.py:289-402`：

```text
x_gt_augment = centre_random_augmentation(label["coordinate"], N_sample=48)
sigma = noise_sampler(size=[48])
noise = randn_like(x_gt_augment) * sigma[...,None,None]
x_denoised = denoise_net(x_gt_augment + noise, sigma, ...)
```

这里的 `N_sample=48` 是 diffusion sample 维，不是 dataloader batch。一次 dataloader batch 只有一个 crop，但该 crop 上并行造 48 个 noisy samples，共享同一套 trunk 表示 `s_inputs/s/z`。

### 7.3 DiffusionConditioning

文件：`diffusion.py:31-177`。

pair conditioning：

```text
pair_z = concat(z_trunk, relpe(relp))      # [N,N,2*c_z]
pair_z = Linear(LN(pair_z))               # [N,N,c_z]
pair_z = pair_z + Transition(pair_z)
pair_z = pair_z + Transition(pair_z)
```

single conditioning：

```text
single_s_base = Linear(LN(concat(s_trunk, s_inputs)))   # [N,c_s]
noise_emb = Fourier(log(sigma/sigma_data)/4)            # [N_sample,c_noise]
single_s = single_s_base[None,:,:] + Linear(LN(noise_emb))[:,None,:]
single_s = single_s + Transition(single_s)
single_s = single_s + Transition(single_s)
```

`use_conditioning=False` 时，代码只在 `pair_z is None` 分支把 `s_trunk/z_trunk` 清零。如果 `enable_diffusion_shared_vars_cache=True` 且 `pair_z` 已缓存，conditioning dropout 对 cached pair 可能不彻底。这是一个真实工程风险，尤其当我们想调 condition dropout rate 时必须验证。

### 7.4 DiffusionModule 的 atom-token-atom 路径

文件：`diffusion.py:231-600`。

denoiser 的 `forward` 先做 EDM 输入缩放：

```text
r_noisy = x_noisy / sqrt(sigma_data^2 + sigma^2)
```

然后 `f_forward`：

1. `DiffusionConditioning` 得到 `s_single [N_sample,N_token,384]` 和 `z_pair [N_token,N_token,256]`。
2. `z_pair` expand 到 sample 维。
3. `AtomAttentionEncoder(has_coords=True)`：
   - 使用 reference atom features 初始化 atom single/pair。
   - 加入 noisy coordinates `r_noisy`。
   - 加入 trunk single `s_trunk` 和 pair `z_pair`。
   - local AtomTransformer 后聚合到 token，得到 `a_token [N_sample,N_token,768]`。
4. `a_token += Linear(LN(s_single))`。
5. `DiffusionTransformer` 跑 token-level full self-attention，pair bias 来自 `z_pair`，每 block 是 pair-bias attention 加 conditioned transition。
6. `AtomAttentionDecoder` 把 token 表示 broadcast 回 atom，加 encoder skip，local AtomTransformer 后线性到 3 维 `r_update`。

最终 EDM 输出组合：

```text
s_ratio = sigma / sigma_data
x_denoised =
    1 / (1 + s_ratio^2) * x_noisy
  + sigma / sqrt(1 + s_ratio^2) * r_update
```

### 7.5 mini-rollout 与训练 denoise 的差异

mini-rollout 调用 `sample_diffusion`，默认 `N_sample_mini_rollout=1`、`N_step_mini_rollout=20`。它从高噪声初始坐标开始做 predictor-corrector/Euler 更新，输出一套 `coordinate_mini`。该过程在训练中 no-grad，主要作用是：

1. 提供一个当前模型预测结构，用于 chain/atom symmetric permutation，使 label 与预测对齐。
2. 给 confidence head 提供输入坐标，使 pLDDT/PDE/PAE/resolved head 学习“当前模型 20-step mini rollout 的质量”。

真正 diffusion loss 用的是 48 个单步 denoise samples，不是 `coordinate_mini`。这会造成一个潜在错配：confidence head 的 target 坐标质量来自 20-step mini rollout，而推理评估用 200-step、5 seeds、5 samples。

### 7.6 Diffusion 子模块伪代码

这一节把 diffusion 部分拆到可实现级别。讨论优化时，建议把每个小模块当成独立检查点：输入张量是否对、sample 维是否对、conditioning 是否真的被使用、checkpoint 是否改变梯度路径、哪些位置用了 no-grad 或 detach。

#### 7.6.1 `TrainingNoiseSampler`

模块职责：给训练 diffusion 的每个 noisy sample 采一个独立噪声强度。这里不是 schedule rollout，而是 log-normal 单步训练噪声。

```python
class TrainingNoiseSampler:
    def __init__(p_mean=-1.2, p_std=1.5, sigma_data=16.0):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data

    def __call__(size, device):
        # size 通常是 batch_shape + [N_sample]
        eps = randn(size, device=device)
        sigma = exp(eps * p_std + p_mean) * sigma_data
        return sigma
```

对本次 run，`size=[48]`，因为 dataloader crop batch 通常是 1，但 diffusion sample 维是 48。优化时要重点看 sigma 分布在目标体系上是否太宽：大 sigma 主导粗结构，小 sigma 主导局部几何和 clash。当前文档里的 loss scale 只进 MSE/bond，不进 smooth lDDT，所以 sigma 分布会以不同方式影响两个主 loss。

#### 7.6.2 `sample_diffusion_training`

模块职责：构造训练用的 noisy coordinate，并调用 denoiser 做一次预测。它是主 diffusion objective 的数据入口。

```python
def sample_diffusion_training(
    noise_sampler,
    denoise_net,
    label_dict,
    input_feature_dict,
    s_inputs,
    s_trunk,
    z_trunk,
    pair_z,
    p_lm,
    c_l,
    N_sample=48,
    diffusion_chunk_size=None,
    use_conditioning=True,
    enable_efficient_fusion=False,
):
    x_gt = label_dict["coordinate"]             # [N_atom,3]
    x_mask = label_dict["coordinate_mask"]      # [N_atom]

    # 对同一个 crop 复制 N_sample 份，并做随机刚体增强。
    # 输出维度是 [N_sample,N_atom,3]。
    x_gt_aug = centre_random_augmentation(
        x_input_coords=x_gt,
        N_sample=N_sample,
        mask=x_mask,
    )

    sigma = noise_sampler(size=[N_sample], device=x_gt.device)
    noise = randn_like(x_gt_aug) * sigma[:, None, None]
    x_noisy = x_gt_aug + noise

    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_noisy,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )
    else:
        pieces = []
        for sample_slice in split_sample_dim(N_sample, diffusion_chunk_size):
            pieces.append(
                denoise_net(
                    x_noisy=x_noisy[sample_slice],
                    t_hat_noise_level=sigma[sample_slice],
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    pair_z=pair_z,
                    p_lm=p_lm,
                    c_l=c_l,
                    use_conditioning=use_conditioning,
                    enable_efficient_fusion=enable_efficient_fusion,
                )
            )
        x_denoised = cat(pieces, dim=sample_dim)

    return x_gt_aug, x_denoised, sigma
```

这里有三个实现细节需要明确：

1. `x_gt_aug` 是 loss 对齐目标，不是原始 label 坐标；MSE/lDDT 比的是 denoised sample 与增强后的 GT。
2. 每个 noisy sample 有自己的 `sigma` 和 noise，但共享同一个 trunk conditioning。
3. 如果启用 `diffusion_chunk_size`，chunk 只切 sample 维，不重新计算 trunk，不改变 loss 数学定义。

#### 7.6.3 `DiffusionConditioning.prepare_cache`

模块职责：把 trunk pair 表示和相对位置编码变成 diffusion transformer 使用的 pair bias 表示。这个结果与 `sigma` 无关，可以在 mini-rollout 和训练 denoise 之间复用。

```python
def prepare_cache(relp_feature, z_trunk, inplace_safe=False):
    relp_z = RelativePositionEncoding(relp_feature)       # [N_token,N_token,c_z]
    pair_z = concat([z_trunk, relp_z], dim=-1)            # [N_token,N_token,2*c_z]

    pair_z = LinearNoBias_z(LayerNorm_z(pair_z))          # [N_token,N_token,c_z]

    if inplace_safe:
        pair_z += Transition_z1(pair_z)
        pair_z += Transition_z2(pair_z)
    else:
        pair_z = pair_z + Transition_z1(pair_z)
        pair_z = pair_z + Transition_z2(pair_z)

    return pair_z
```

工程含义：

1. `pair_z` 是 token-token 表示，不带 `N_sample` 维。
2. `prepare_cache` 内部包含两层 transition，相当于对 pair conditioning 做了额外非线性适配。
3. 如果外层打开 `enable_diffusion_shared_vars_cache`，这里可以只算一次；但如果我们希望做 conditioning dropout，需要确认 cached pair 是否绕过了 dropout 分支。

#### 7.6.4 `DiffusionConditioning.forward`

模块职责：同时生成 diffusion denoiser 的 single conditioning 和 pair conditioning。pair 只依赖 trunk/relp，single 同时依赖 trunk/input feature 和噪声强度。

```python
def diffusion_conditioning_forward(
    t_hat_noise_level,      # [N_sample]
    relp_feature,
    s_inputs,               # [N_token,c_s_inputs]
    s_trunk,                # [N_token,c_s]
    z_trunk,                # [N_token,N_token,c_z]
    pair_z,                 # None or cached [N_token,N_token,c_z]
    inplace_safe=False,
    use_conditioning=True,
):
    if pair_z is None:
        if not use_conditioning:
            # 注意：这个清零只发生在 pair_z is None 的分支。
            s_trunk = zero(s_trunk)
            z_trunk = zero(z_trunk)
        pair_z = prepare_cache(relp_feature, z_trunk, inplace_safe)
    else:
        if inplace_safe:
            pair_z = clone(pair_z)

    # single base 不带 sample 维。
    single_base = concat([s_trunk, s_inputs], dim=-1)      # [N_token,c_s+c_s_inputs]
    single_base = LinearNoBias_s(LayerNorm_s(single_base))# [N_token,c_s]

    # noise embedding 带 sample 维。
    noise_arg = log(t_hat_noise_level / sigma_data) / 4   # [N_sample]
    noise_emb = FourierEmbedding(noise_arg)               # [N_sample,c_noise]
    noise_emb = LinearNoBias_n(LayerNorm_n(noise_emb))    # [N_sample,c_s]

    single_s = single_base[None, :, :] + noise_emb[:, None, :]
    single_s = single_s + Transition_s1(single_s)
    single_s = single_s + Transition_s2(single_s)

    return single_s, pair_z
```

这个模块是 diffusion 条件注入的核心。对算法侧最关键的问题是：噪声信息只进入 `single_s`，pair bias 本身不感知 `sigma`；pair 对不同 sample 共享，sample 差异主要来自 `r_noisy` 和 `single_s`。

#### 7.6.5 `DiffusionModule.forward`

模块职责：实现 EDM 风格的输入缩放、网络残差预测和输出组合。它包装 `f_forward`，真正的大网络在 `f_forward` 中。

```python
def diffusion_module_forward(
    x_noisy,                # [N_sample,N_atom,3]
    t_hat_noise_level,      # [N_sample]
    input_feature_dict,
    s_inputs,
    s_trunk,
    z_trunk,
    pair_z,
    p_lm,
    c_l,
    use_conditioning=True,
):
    sigma = t_hat_noise_level

    # EDM c_in。把坐标缩到近似单位尺度。
    c_in = 1 / sqrt(sigma_data**2 + sigma**2)             # [N_sample]
    r_noisy = x_noisy * c_in[:, None, None]

    r_update = f_forward(
        r_noisy=r_noisy,
        t_hat_noise_level=sigma,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        use_conditioning=use_conditioning,
    )

    # EDM c_skip/c_out。代码写成 s_ratio 形式。
    s_ratio = sigma / sigma_data
    c_skip = 1 / (1 + s_ratio**2)
    c_out = sigma / sqrt(1 + s_ratio**2)

    x_denoised = c_skip[:, None, None] * x_noisy \
               + c_out[:, None, None] * r_update

    return x_denoised
```

这里 `r_update` 不是直接坐标，而是被 `c_out` 缩放后与 noisy coordinate 混合的更新量。调 loss 时不能简单把网络输出解释为绝对坐标；不同 sigma 下 `c_skip/c_out` 改变了梯度对 `r_update` 的有效尺度。

#### 7.6.6 `DiffusionModule.f_forward`

模块职责：完整的 atom-token-atom denoiser。它把 noisy atom coordinate 编码到 token，跑 24 层 token diffusion transformer，再解码回 atom coordinate update。

```python
def f_forward(
    r_noisy,                # [N_sample,N_atom,3]
    t_hat_noise_level,      # [N_sample]
    input_feature_dict,
    s_inputs,               # [N_token,c_s_inputs]
    s_trunk,                # [N_token,c_s]
    z_trunk,                # [N_token,N_token,c_z]
    pair_z,                 # cached or None
    p_lm,
    c_l,
    chunk_size=None,
    use_conditioning=True,
    enable_efficient_fusion=False,
):
    N_sample = r_noisy.shape[-3]
    assert t_hat_noise_level.shape[-1] == N_sample

    # 1. diffusion conditioning。
    # s_single 带 sample 维，z_pair 不带 sample 维。
    s_single, z_pair = DiffusionConditioning(
        t_hat_noise_level=t_hat_noise_level,
        relp_feature=input_feature_dict["relp"],
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_z=pair_z,
        use_conditioning=use_conditioning,
    )

    # 2. 扩展 trunk/pair 到 sample 维。
    s_trunk_sample = expand_at_dim(s_trunk, dim=sample_dim, n=1)
    z_pair_sample = expand_at_dim(z_pair, dim=sample_dim, n=1)

    # 3. atom encoder：局部 atom attention + atom 到 token 聚合。
    a_token, q_skip, c_skip, p_skip = AtomAttentionEncoder(
        atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
        ref_pos=input_feature_dict["ref_pos"],
        ref_charge=input_feature_dict["ref_charge"],
        ref_mask=input_feature_dict["ref_mask"],
        ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
        ref_element=input_feature_dict["ref_element"],
        d_lm=input_feature_dict["d_lm"],
        v_lm=input_feature_dict["v_lm"],
        pad_info=input_feature_dict["pad_info"],
        r_l=r_noisy,
        s=s_trunk_sample,
        z=z_pair_sample,
        p_lm=p_lm,
        c_l=c_l,
        chunk_size=chunk_size,
    )

    # 4. token 表示加入 sigma-aware single conditioning。
    a_token = float32(a_token)
    a_token = a_token + LinearNoBias_s(LayerNorm_s(s_single))

    # 5. token-level diffusion transformer。
    if enable_efficient_fusion:
        z_for_attention = normalize(float32(z_pair_sample))
        z_for_attention = permute_final_dims(z_for_attention, [2, 0, 1]).contiguous()
    else:
        z_for_attention = float32(z_pair_sample)

    a_token = DiffusionTransformer(
        a=float32(a_token),
        s=float32(s_single),
        z=z_for_attention,
        chunk_size=chunk_size,
        enable_efficient_fusion=enable_efficient_fusion,
    )

    a_token = LayerNorm_a(a_token)

    # 6. atom decoder：token broadcast 回 atom，结合 encoder skip，输出 3D update。
    r_update = AtomAttentionDecoder(
        atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
        a=a_token,
        q_skip=q_skip,
        c_skip=c_skip,
        p_skip=p_skip,
        chunk_size=chunk_size,
    )

    return r_update
```

主要维度流：

| 节点 | 维度 | 说明 |
|---|---|---|
| `r_noisy` | `[N_sample,N_atom,3]` | scaled noisy coordinate。 |
| `s_single` | `[N_sample,N_token,c_s]` | 带 sigma embedding 的 single conditioning。 |
| `z_pair` | `[N_token,N_token,c_z]` | 共享 pair conditioning。 |
| `z_pair_sample` | `[N_sample,N_token,N_token,c_z]` | 给 attention bias 使用。 |
| `a_token` | `[N_sample,N_token,c_token]` | diffusion transformer 的 token stream。 |
| `r_update` | `[N_sample,N_atom,3]` | EDM 输出组合前的 update。 |

#### 7.6.7 `AtomAttentionEncoder` 在 diffusion 中的作用

`AtomAttentionEncoder` 在 trunk 和 diffusion 中都会用，但 diffusion 场景 `has_coords=True`，所以 noisy coordinate 是输入信号的一部分。伪代码如下：

```python
def atom_attention_encoder_for_diffusion(
    atom_to_token_idx,
    ref_pos,
    ref_charge,
    ref_mask,
    ref_atom_name_chars,
    ref_element,
    d_lm,
    v_lm,
    pad_info,
    r_l,                    # [N_sample,N_atom,3]
    s,                      # [N_sample,N_token,c_s]
    z,                      # [N_sample,N_token,N_token,c_z]
    p_lm,
    c_l,
):
    # 1. 构造 atom single feature q_l。
    q_l = embed_ref_atom_features(
        ref_pos=ref_pos,
        ref_charge=ref_charge,
        ref_mask=ref_mask,
        ref_atom_name_chars=ref_atom_name_chars,
        ref_element=ref_element,
    )

    # 2. diffusion 特有：把 noisy coordinate 编码到 atom stream。
    q_l = q_l + LinearNoBias_r(r_l)

    # 3. token conditioning broadcast 到 atom。
    s_atom = broadcast_token_to_atom(s, atom_to_token_idx)
    q_l = q_l + LinearNoBias_s(s_atom)

    # 4. 构造局部 atom pair feature p_lm。
    # 它包含局部窗口的 ref geometry、token pair z、语言/配体特征。
    p_lm = build_local_atom_pair_feature(
        atom_to_token_idx=atom_to_token_idx,
        ref_pos=ref_pos,
        z=z,
        d_lm=d_lm,
        v_lm=v_lm,
        p_lm=p_lm,
        c_l=c_l,
        pad_info=pad_info,
    )
    p_lm = p_lm + small_mlp(p_lm)

    # 5. 局部 atom transformer。
    q_l = AtomTransformer(q_l, c_l, p_lm)

    # 6. atom -> token mean aggregation。
    a_token = aggregate_atom_to_token(
        x_atom=relu(LinearNoBias_q(q_l)),
        atom_to_token_idx=atom_to_token_idx,
        reduce="mean",
    )

    # q_l/c_l/p_lm 作为 decoder skip。
    return a_token, q_l, c_l, p_lm
```

优化上要注意：这个模块是 diffusion 中最贴近原子几何的部分。如果 SAb23H2 失败集中在抗体-抗原界面局部构象或 side-chain-like atom placement，优先检查这里的局部窗口、pair feature、atom permutation 和 ligand/protein atom 权重是否把信号传对。

#### 7.6.8 `DiffusionTransformerBlock`

模块职责：在 token stream 上做 pair-bias attention，再做由 single conditioning 调制的 transition。默认 v2 diffusion transformer 是 24 blocks、16 heads、`c_token=768`。

```python
def diffusion_transformer_block(a, s, z):
    # a: [N_sample,N_token,c_token]
    # s: [N_sample,N_token,c_s]
    # z: [N_sample,N_token,N_token,c_z]

    attn_out = AttentionPairBias(
        a=a,
        s=s,
        z=z,
        has_s=True,
    )
    attn_out = DropPath(attn_out)
    a_after_attn = a + attn_out

    ff_out = ConditionedTransitionBlock(
        a=a_after_attn,
        s=s,
    )
    ff_out = DropPath(ff_out)
    out_a = a_after_attn + ff_out

    return out_a, s, z
```

这里的 `z` 不被更新，只作为 attention bias；`s` 也不被更新，只作为 conditioning。真正被 24 层迭代更新的是 `a`。如果想提升 denoiser 表达能力，候选方向包括：调整 `z` 的 conditioning 强度、增加对界面 pair 的 bias、改 `ConditionedTransitionBlock` 的调制方式、或引入 sample-dependent pair modulation。但这些都会改变显存和训练稳定性。

#### 7.6.9 `DiffusionTransformer`

模块职责：把多个 `DiffusionTransformerBlock` 串起来，并按 `blocks_per_ckpt` 做 activation checkpoint。

```python
def diffusion_transformer_forward(a, s, z, blocks, blocks_per_ckpt):
    prepared_blocks = []
    for block in blocks:
        prepared_blocks.append(
            partial(
                block,
                n_queries=None,
                n_keys=None,
                chunk_size=chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
            )
        )

    if not grad_enabled():
        blocks_per_ckpt = None

    a, s, z = checkpoint_blocks(
        prepared_blocks,
        args=(a, s, z),
        blocks_per_ckpt=blocks_per_ckpt,
    )

    return a
```

这个模块不会返回更新后的 `s/z`。因此 diffusion transformer 的优化目标全部压在 token stream `a` 的更新上；pair/single conditioning 是静态条件而不是 trunk 的继续迭代。

#### 7.6.10 `AtomAttentionDecoder`

模块职责：把 diffusion transformer 后的 token 表示映射回 atom 表示，并输出 3D update。

```python
def atom_attention_decoder(atom_to_token_idx, a, q_skip, c_skip, p_skip):
    # 1. token -> atom broadcast。
    a_atom = broadcast_token_to_atom(
        x_token=LinearNoBias_a(a),
        atom_to_token_idx=atom_to_token_idx,
    )

    # 2. 加 encoder atom skip。
    q = a_atom + q_skip

    # 3. 用 encoder 保存的 atom conditioning/pair conditioning 再跑局部 atom transformer。
    q = AtomTransformer(
        q=q,
        c=c_skip,
        p=p_skip,
    )

    # 4. 映射到 3D update。
    r_update = LinearNoBias_out(LayerNorm_q(q))
    return r_update
```

这个 decoder 的输出层是线性到 3 维。它没有直接输出置信度或距离分布；所有 diffusion 主 loss 都通过 `x_denoised` 的坐标反传回来。

#### 7.6.11 `sample_diffusion` mini-rollout / inference sampler

模块职责：从纯噪声坐标开始，按 inference noise schedule 多步调用 denoiser。训练中的 mini-rollout 和推理主采样都走这条路径，只是步数、sample 数、是否 no-grad、是否后续计算 loss 不同。

```python
def sample_diffusion(
    denoise_net,
    input_feature_dict,
    s_inputs,
    s_trunk,
    z_trunk,
    pair_z,
    p_lm,
    c_l,
    noise_schedule,
    N_sample,
    gamma0=0.8,
    gamma_min=1.0,
    noise_scale_lambda=1.003,
    step_scale_eta=1.5,
):
    N_atom = input_feature_dict["atom_to_token_idx"].shape[-1]

    # 初始高噪声坐标。
    x_l = noise_schedule[0] * randn([N_sample, N_atom, 3])

    for step_i, (c_tau_last, c_tau) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
        # 每一步先中心化/随机刚体增强，保持坐标规范。
        x_l = centre_random_augmentation(x_l, N_sample=1).squeeze(sample_axis)

        # churn：在高噪声区额外加噪。
        gamma = gamma0 if c_tau > gamma_min else 0.0
        t_hat = c_tau_last * (1 + gamma)
        delta_sigma = sqrt(t_hat**2 - c_tau_last**2)
        x_noisy = x_l + noise_scale_lambda * delta_sigma * randn_like(x_l)

        t_hat_tensor = full([N_sample], t_hat)

        x_denoised = denoise_net(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat_tensor,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
        )

        # Euler step。代码中特别说明 AF3 line 9 的 x_l_hat 可能是 typo，
        # 当前实现用 (x_noisy - x_denoised) / t_hat。
        delta = (x_noisy - x_denoised) / t_hat_tensor[:, None, None]
        dt = c_tau - t_hat
        x_l = x_noisy + step_scale_eta * dt * delta

    return x_l
```

训练中 mini-rollout 的 `x_l` 进入两个地方：一是 `permute_label_to_match_mini_rollout` 对齐 label，二是 confidence head 生成训练 logits。它不直接进入 diffusion 主 loss；主 loss 使用 `sample_diffusion_training` 的单步 denoise 输出。

### 7.7 EDM 方法对照

本节对照 `/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/EDM` 下的 EDM 方法文档。EDM 的核心贡献不是单个公式，而是把扩散模型拆成彼此相对独立的设计块：denoiser 定义、网络预条件化、训练噪声分布、loss weighting、时间离散、ODE/SDE sampler。Protenix 的 diffusion 模块明显采用了 EDM 的预条件化和训练噪声思想，但在 sampler、loss 形态和数据空间上做了结构预测任务需要的改造。

#### 7.7.1 EDM 的统一 denoiser 视角

EDM 从 denoising score matching 出发。给定干净样本 `y` 和 Gaussian noise `n ~ N(0, sigma^2 I)`，理想 denoiser 是：

```text
D(y + n; sigma) = argmin_D E || D(y + n; sigma) - y ||_2^2
```

在 score-based 公式中，它和 score 的关系是：

```text
score(x; sigma) = ∇_x log p(x; sigma)
                = (D(x; sigma) - x) / sigma^2
```

因此 sampler 不一定要显式预测 score；只要有 denoiser `D_theta(x; sigma)`，就能通过：

```text
d x / d t = (x - D_theta(x; t)) / t
```

在 `sigma(t)=t, s(t)=1` 的设置下做 ODE 反向积分。Protenix 的 `DiffusionModule.forward()` 正是在实现一个条件 denoiser：

```text
x_hat = D_theta(
    x_noisy,
    sigma,
    trunk embeddings,
    atom/reference features,
    local atom pair features
)
```

不同之处是 EDM 原始任务是图像分布，`x` 是 image tensor；Protenix 中 `x` 是 atom coordinate tensor，且 denoiser 还强条件化于 sequence/MSA/template/trunk pair 表示。

#### 7.7.2 EDM 预条件化公式

EDM 不直接训练 raw network `F_theta` 输出 clean sample，而是定义：

```text
D_theta(x; sigma)
  = c_skip(sigma) * x
  + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))
```

其中 EDM 推荐：

```text
c_skip(sigma) = sigma_data^2 / (sigma^2 + sigma_data^2)
c_out(sigma)  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
c_in(sigma)   = 1 / sqrt(sigma^2 + sigma_data^2)
c_noise(sigma)= 1/4 * log(sigma)
```

`sigma_data` 是数据标准差尺度，不是固定物理常数。EDM 论文图像实验表里使用的数值和 Protenix 坐标任务不同；Protenix 配置为：

```text
sigma_data = 16.0
```

这相当于把 atom coordinate 的天然尺度纳入预条件化。配置中也专门标注了这里不同于常见 EDM 图像设置。

#### 7.7.3 Protenix 对 EDM 预条件化的代码映射

Protenix 的代码变量映射如下：

| EDM 符号 | Protenix 变量/代码 | 说明 |
|---|---|---|
| `x` | `x_noisy` | noisy atom coordinates，shape `[N_sample,N_atom,3]`。 |
| `sigma` | `t_hat_noise_level` / `noise_level` | 训练单步 denoise 或推理 sampler 当前噪声。 |
| `sigma_data` | `configs.sigma_data = 16.0` | 坐标尺度。 |
| `c_in * x` | `r_noisy = x_noisy / sqrt(sigma_data^2 + sigma^2)` | 输入坐标缩放。 |
| `F_theta(...)` | `DiffusionModule.f_forward(...) -> r_update` | atom-token-atom denoiser raw update。 |
| `c_skip` | `1 / (1 + (sigma/sigma_data)^2)` | 代码用 `s_ratio` 写。 |
| `c_out` | `sigma / sqrt(1 + (sigma/sigma_data)^2)` | 等价于 `sigma*sigma_data/sqrt(sigma^2+sigma_data^2)`。 |
| `D_theta` | `x_denoised` | 进入 diffusion loss 的坐标。 |
| `c_noise` | `log(sigma/sigma_data)/4` | Protenix 相比 EDM 表格多除以 `sigma_data`，用相对噪声尺度做 Fourier embedding。 |

Protenix 代码：

```python
r_noisy = x_noisy / sqrt(sigma_data**2 + sigma**2)
r_update = f_forward(r_noisy, sigma, conditioning)

s_ratio = sigma / sigma_data
x_denoised = (
    1 / (1 + s_ratio**2) * x_noisy
    + sigma / sqrt(1 + s_ratio**2) * r_update
)
```

这和 EDM 的 `D_theta = c_skip*x + c_out*F_theta(c_in*x,c_noise)` 对齐。唯一需要特别说明的是 `c_noise`：EDM 表格写 `1/4 log(sigma)`，Protenix 写 `1/4 log(sigma/sigma_data)`。这会把 noise embedding 的零点放在 `sigma=sigma_data`，更适合跨坐标尺度迁移。

#### 7.7.4 EDM loss weighting 与 Protenix 的 per-sample scale

EDM 的预条件化训练 loss 可以写成 raw network `F_theta` 的监督问题：

```text
E[ lambda(sigma) * || D_theta(y+n; sigma) - y ||^2 ]
```

代入预条件化形式后，`F_theta` 的有效 loss 权重是：

```text
effective_weight(sigma) = lambda(sigma) * c_out(sigma)^2
```

EDM 为了让不同 sigma 的 raw network 训练权重更均衡，取：

```text
lambda(sigma) = 1 / c_out(sigma)^2
              = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
```

Protenix 训练中对 MSE 和 bond 使用的：

```text
diffusion_per_sample_scale =
    (sigma^2 + sigma_data^2) / (sigma_data * sigma)^2
```

正是 EDM 的 `lambda(sigma)`。这不是随手加的 reweight，而是 EDM 预条件化下为了抵消 `c_out(sigma)` 引起的 raw-network 梯度尺度差异。

但 Protenix 没有把这个 scale 传给所有 diffusion loss：

| Loss | 是否使用 EDM `lambda(sigma)` | 当前实现含义 |
|---|---|---|
| `MSELoss` | 是 | 对坐标 MSE 使用 EDM per-sigma weighting。 |
| `BondLoss` | 是 | 若启用，会对 bond distance error 使用同样 weighting。 |
| `SmoothLDDTLoss` | 否 | 对不同 sigma 的 pairwise distance quality 等权平均。 |
| `DistogramLoss` | 否 | 不依赖 denoised coordinate 或 sigma。 |
| confidence losses | 否 | 输入来自 mini-rollout，不是训练单步 denoise。 |

因此严格说，Protenix 是“EDM 预条件化 + EDM MSE/bond weighting + 结构任务专用 smooth lDDT 辅助目标”，不是纯 EDM L2 denoising objective。

#### 7.7.5 EDM 训练噪声分布与 Protenix

EDM 使用 log-normal 噪声分布：

```text
log(sigma) ~ Normal(P_mean, P_std^2)
```

Protenix 对应代码：

```python
sigma = exp(randn * p_std + p_mean) * sigma_data
```

当前配置：

```text
p_mean = -1.2
p_std = 1.5
sigma_data = 16.0
```

也就是：

```text
log(sigma / sigma_data) ~ Normal(-1.2, 1.5^2)
```

这和 `c_noise = log(sigma/sigma_data)/4` 保持一致：训练采样和噪声条件都在相对噪声尺度 `sigma/sigma_data` 上定义。

优化含义：

1. `sigma_data=16` 会整体平移实际坐标噪声范围。
2. `p_std=1.5` 比 EDM 图像表中常见设置更宽，会覆盖很大的噪声跨度。
3. 如果 SAb23H2 的失败主要来自界面局部误差，可能需要检查低 sigma 样本占比是否足够；如果失败来自全局 docking，可能需要检查中高 sigma 的 denoise 是否更关键。

#### 7.7.6 EDM 时间离散和 Protenix inference schedule

EDM 推荐的 sampling noise levels：

```text
sigma_i = (
    sigma_max^(1/rho)
  + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho))
)^rho
sigma_N = 0
rho = 7
```

Protenix `InferenceNoiseScheduler`：

```python
step_size = 1 / N_step
step_indices = arange(N_step + 1)
t_i = sigma_data * (
    s_max**(1/rho)
    + step_indices * step_size * (s_min**(1/rho) - s_max**(1/rho))
)**rho
t_N = 0
```

当前配置：

```text
s_max = 160
s_min = 4e-4
rho = 7
sigma_data = 16
N_step = 200 推理 / 20 mini-rollout
```

注意这里的 `s_max/s_min` 是相对尺度，最后又乘了 `sigma_data`。所以实际最高噪声是：

```text
sigma_0 = sigma_data * s_max = 16 * 160 = 2560
```

这比训练 log-normal 的典型 sigma 高很多。它符合 EDM/AF3 类 sampler 从极高噪声开始的设计，但也意味着推理早期完全依赖 denoiser 在极高噪声区的泛化。训练分布、推理 schedule 和 `N_step` 之间是否匹配，是 sampling finetune 必须评估的点。

#### 7.7.7 EDM Heun sampler 与当前 Protenix sampler

EDM deterministic sampler 推荐 Heun 二阶方法。简化到 `sigma(t)=t, s(t)=1` 时：

```text
d_i = (x_i - D_theta(x_i; t_i)) / t_i
x_euler = x_i + (t_{i+1} - t_i) * d_i

if t_{i+1} != 0:
    d'_i = (x_euler - D_theta(x_euler; t_{i+1})) / t_{i+1}
    x_{i+1} = x_i + (t_{i+1} - t_i) * (d_i + d'_i) / 2
else:
    x_{i+1} = x_euler
```

EDM stochastic sampler又在每步先做 churn：

```text
t_hat = t_i + gamma_i * t_i
x_hat = x_i + sqrt(t_hat^2 - t_i^2) * S_noise * eps
d_i = (x_hat - D_theta(x_hat; t_hat)) / t_hat
x_euler = x_hat + (t_{i+1} - t_hat) * d_i
if t_{i+1} != 0:
    d'_i = (x_euler - D_theta(x_euler; t_{i+1})) / t_{i+1}
    x_{i+1} = x_hat + (t_{i+1} - t_hat) * (d_i + d'_i) / 2
```

Protenix 当前 `sample_diffusion()` 更接近“EDM stochastic sampler 的 churn + 一阶 Euler”：

```python
gamma = gamma0 if c_tau > gamma_min else 0
t_hat = c_tau_last * (1 + gamma)
x_noisy = x_l + noise_scale_lambda * sqrt(t_hat**2 - c_tau_last**2) * eps
x_denoised = D_theta(x_noisy, t_hat)
delta = (x_noisy - x_denoised) / t_hat
x_l = x_noisy + step_scale_eta * (c_tau - t_hat) * delta
```

差异：

| 设计点 | EDM 文档 | Protenix 当前实现 |
|---|---|---|
| 二阶修正 | Heun：每步通常 2 次 denoiser eval | 当前没有第二次 `D_theta(x_euler,t_{i+1})`。 |
| churn gate | `gamma_i = min(S_churn/N, sqrt(2)-1)` 且只在 `[S_min,S_max]` 内启用 | `gamma = gamma0 if c_tau > gamma_min else 0`。 |
| noise inflation | `S_noise` | `noise_scale_lambda=1.003`，语义相近。 |
| step scale | EDM Euler/Heun 使用自然步长 | Protenix 额外乘 `step_scale_eta=1.5`。 |
| 最后到 0 | EDM 在 `t=0` 回退 Euler，避免除零 | Protenix 一直用当前 `t_hat` 做分母，并让 `c_tau` 可到 0。 |

这说明当前推理 sampler 不是完整 EDM Heun sampler。它更像经过 AF3/Protenix 适配的一阶 stochastic sampler：少一次 denoiser eval，速度更低成本，但在相同步数下局部截断误差可能大于 EDM Heun。若后续讨论 sampler 优化，可以把“加 Heun correction”作为一个明确实验，而不是模糊说“改 diffusion”。

#### 7.7.8 当前实现相对 EDM 的保留、改造和偏离

| 模块 | EDM 标准方法 | Protenix 当前实现 | 评价 |
|---|---|---|---|
| Denoiser parameterization | `D = c_skip*x + c_out*F(c_in*x,c_noise)` | 基本一致 | 核心预条件化被保留。 |
| `sigma_data` | 数据标准差尺度 | 设为 16 | 合理但需要用坐标分布验证。 |
| `c_noise` | `1/4 log(sigma)` | `1/4 log(sigma/sigma_data)` | 使用相对噪声，更稳定。 |
| Training sigma | log-normal | log-normal on `sigma/sigma_data` | 保留 EDM 思路。 |
| Loss weighting | `lambda=1/c_out^2` | MSE/bond 使用同一公式 | 部分保留。smooth lDDT 不使用。 |
| Training target | clean sample L2 | coordinate MSE + smooth lDDT + distogram + confidence | 改为结构任务混合目标。 |
| Augmentation | 非泄漏几何/颜色增强并作为条件 | 坐标随机刚体增强，不作为额外条件 | 结构任务中刚体增强不应泄漏绝对坐标。 |
| Sampler schedule | rho=7, sigma(t)=t | rho=7，相对 schedule 乘 `sigma_data` | 保留主要 schedule 形态。 |
| Sampler solver | deterministic Heun 或 stochastic Heun | churn + Euler + `step_scale_eta` | 明显偏离 EDM 完整二阶 sampler。 |

因此，在给首席们讨论时可以明确说：Protenix diffusion 不是“泛泛用了扩散模型”，而是采用 EDM 预条件化框架；但在三维结构任务上，把原始 EDM 的 L2 denoising objective 扩展为 coordinate/lDDT/distogram/confidence 混合目标，并把 EDM sampler 简化/改造成当前一阶 stochastic update。

#### 7.7.9 基于 EDM 对照的优化建议

这些建议不是结论，而是从 EDM 对照里自然暴露出的可实验变量：

1. **Sampler solver ablation**：实现 EDM Heun correction，比较 `200-step Euler`、`100-step Heun`、`200-step Heun` 在 SAb23H2 oracle/top-1/DockQ/clash 上的变化。若 Heun 明显提升同 NFE 质量，说明当前采样误差是瓶颈。
2. **Churn 参数 ablation**：当前 `gamma0=0.8`、`gamma_min=1.0`、`noise_scale_lambda=1.003`、`step_scale_eta=1.5` 不是 EDM 表格原始参数。需要扫 `gamma0/gamma_min/step_scale_eta`，并记录 oracle-best 与 clash。
3. **训练 sigma 分布 ablation**：固定模型和数据，扫 `p_mean/p_std` 或按 interface 任务缩窄低 sigma 区间，观察局部 interface DockQ、clash 和 global docking 是否分别变化。
4. **smooth lDDT sigma weighting**：当前 smooth lDDT 不用 EDM `lambda(sigma)`。可以实验 `smooth_lddt` 是否也按 sigma reweight，或按低 sigma 加权，检查局部几何是否改善。
5. **sigma_data 校准**：统计训练 crop 的 coordinate RMS scale、pair distance scale、centre_random_augmentation 后尺度，验证 `sigma_data=16` 是否仍是最优。若数据裁剪/居中改变尺度，`sigma_data` 不一定应固定。
6. **mini-rollout 步数与主 sampler 对齐**：训练 label permutation 和 confidence target 用 20-step sampler，推理用 200-step。可比较 `N_step_mini_rollout=20/50/100` 对 permutation 稳定性和 confidence calibration 的影响。

这组实验比“调 diffusion loss 权重”更系统，因为它按照 EDM 的独立设计块拆分：训练噪声、预条件化尺度、loss weighting、sampler 离散、stochasticity。

## 8. Confidence、Distogram 和对称置换

### 8.1 对称置换

`SymmetricPermutation` 在 `protenix/utils/permutation/permutation.py`。

训练里有两次相关调用：

| 位置 | 默认效果 |
|---|---|
| `permute_label_to_match_mini_rollout` | 用 `coordinate_mini` 先做 chain permutation 再做 atom permutation；默认 `train.mini_rollout=True`，所以会更新 `label_dict`。 |
| `permute_diffusion_sample_to_match_label(stage="train")` | 注释说对 diffusion sample 做 per-sample permutation，但默认 `configs_base.py:366-386` 中 train 阶段 `diffusion_sample=False`，通常只记录 log，不更新 `pred_dict["coordinate"]`。 |

这点很重要：loss 的主 diffusion coordinate 通常不是被 per-sample permutation 后再算，而是依赖 mini-rollout 已经把 label 方向对齐。若我们要改 symmetry handling，需要明确改的是 mini-rollout label permutation 还是 diffusion sample permutation。

### 8.2 ConfidenceHead

文件：`protenix/model/modules/confidence.py:26-348`。

输入：

```text
s_inputs [N,449]
s_trunk [N,384]
z_trunk [N,N,256]
x_pred_coords [N_sample,N_atom,3]
```

默认 `stop_gradient=True`，所以 `s_inputs/s_trunk/z_trunk` 都 detach。ConfidenceHead 不会通过 confidence loss 反推 trunk 表示，只训练 head 自己的参数；且 loss 权重极小。

每个 sample 独立运行 `memory_efficient_forward`：

1. 取代表原子坐标，计算 token-token distance。
2. distance one-hot embedding 和 raw distance linear embedding 加到 `z_pair`。
3. 跑 4-block `PairformerStack`。
4. `linear_pae(LN(z_pair))` 输出 PAE logits。
5. `linear_pde(LN(z_pair + z_pair^T))` 输出 PDE logits。
6. token single broadcast 到 atom，按 `atom_to_tokatom_idx` 选 pLDDT/resolved 权重，输出 atom-level pLDDT/resolved logits。

输出：

```text
plddt:   [N_sample,N_atom,50]
pae:     [N_sample,N_token,N_token,64]
pde:     [N_sample,N_token,N_token,64]
resolved:[N_sample,N_atom,2]
```

### 8.3 DistogramHead

文件：`protenix/model/modules/head.py:22-56`。

DistogramHead 很简单：

```text
logits = Linear(z)              # [N,N,64]
logits = logits + logits^T
```

训练中它提供 distogram loss；推理中 `Protenix.inference` 会把 distogram logits 转成 `contact_probs`，再进入 confidence summary 的 gpde 等计算。

### 8.4 ranking score

推理 confidence summary 在 `protenix/model/sample_confidence.py:111-169`：

```text
ranking_score = 0.8 * iptm
              + 0.2 * ptm
              + 0.5 * disorder
              - 100 * has_clash
```

当前 `disorder` 是 0。`iptm/ptm` 都来自 PAE 概率，`has_clash` 来自坐标 clash 检测。这解释了为什么当前 finetune 不应被描述为 ranking 优化：PAE loss 权重是 0，confidence 其他项权重只有 `1e-4`，且 confidence head stop-gradient。若 oracle-best 提升但 top-1 不提升，这是符合当前目标函数设计的，不是意外。

### 8.5 Head 和 permutation 伪代码

#### 8.5.1 `ConfidenceHead.forward`

模块职责：给一组 predicted coordinates 输出 pLDDT、PAE、PDE、resolved logits。训练中它通常接收 `coordinate_mini`，推理中接收 diffusion sampler 最终输出。

```python
def confidence_head_forward(
    input_feature_dict,
    s_inputs,               # [N_token,c_s_inputs]
    s_trunk,                # [N_token,c_s]
    z_trunk,                # [N_token,N_token,c_z]
    pair_mask,              # [N_token,N_token]
    x_pred_coords,          # [N_sample,N_atom,3]
    use_embedding=True,
    stop_gradient=True,
):
    if stop_gradient:
        s_inputs = detach(s_inputs)
        s_trunk = detach(s_trunk)
        z_trunk = detach(z_trunk)

    # 代码先 clamp 再 LN，避免极端 trunk 值影响 confidence head。
    s_trunk = LayerNorm_s(clamp(s_trunk, min=-512, max=512))

    if not use_embedding:
        z_trunk = zero(z_trunk)

    rep_mask = input_feature_dict["distogram_rep_atom_mask"].bool()
    x_rep = x_pred_coords[:, rep_mask, :]                 # [N_sample,N_token,3]
    N_sample = x_rep.shape[-3]

    # 用 s_inputs 初始化 pair，再加 trunk pair。
    z_init = LinearNoBias_s1(s_inputs)[None, :, :] \
           + LinearNoBias_s2(s_inputs)[:, None, :]
    z_pair_base = z_init + z_trunk                         # [N_token,N_token,c_z]

    plddt_list = []
    pae_list = []
    pde_list = []
    resolved_list = []

    for i in range(N_sample):
        plddt_i, pae_i, pde_i, resolved_i = memory_efficient_forward(
            input_feature_dict=input_feature_dict,
            s_trunk=clone_if_needed(s_trunk),
            z_pair=clone_if_needed(z_pair_base),
            pair_mask=pair_mask,
            x_pred_rep_coords=x_rep[i],
        )
        plddt_list.append(plddt_i)
        pae_list.append(pae_i)
        pde_list.append(pde_i)
        resolved_list.append(resolved_i)

    plddt = stack(plddt_list, dim=sample_dim)              # [N_sample,N_atom,50]
    pae = stack(pae_list, dim=sample_dim)                  # [N_sample,N_token,N_token,64]
    pde = stack(pde_list, dim=sample_dim)                  # [N_sample,N_token,N_token,64]
    resolved = stack(resolved_list, dim=sample_dim)        # [N_sample,N_atom,2]

    return plddt, pae, pde, resolved
```

关键点：

1. 默认 `stop_gradient=True`，confidence loss 不训练 trunk，也不训练 diffusion denoiser。
2. 每个 sample 单独跑一次 4-block confidence Pairformer，避免一次性展开导致 OOM。
3. `z_init` 来自 `s_inputs` 的 pairwise projection，因此即使 `z_trunk` 被清零，confidence head 仍有输入 feature pair 信息。

#### 8.5.2 `ConfidenceHead.memory_efficient_forward`

模块职责：对单个 sample 的 predicted coordinate 生成四类 confidence logits。

```python
def memory_efficient_forward(
    input_feature_dict,
    s_trunk,                # [N_token,c_s]
    z_pair,                 # [N_token,N_token,c_z]
    pair_mask,
    x_pred_rep_coords,      # [N_token,3]
):
    # 1. token representative atom distance。
    with autocast_disabled():
        d = cdist(float32(x_pred_rep_coords), float32(x_pred_rep_coords))

    # 2. 距离 one-hot 和 raw distance 都注入 pair。
    d_onehot = one_hot(d, lower_bins, upper_bins)          # [N_token,N_token,num_bins]
    z_pair = z_pair + LinearNoBias_d(d_onehot)
    z_pair = z_pair + LinearNoBias_d_raw(d[..., None])

    # 3. confidence 专用 PairformerStack。
    s_single, z_pair = PairformerStack(
        s=s_trunk,
        z=z_pair,
        mask=pair_mask,
    )

    z_pair = float32(z_pair)
    s_single = float32(s_single)

    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    atom_to_tokatom_idx = input_feature_dict["atom_to_tokatom_idx"]

    # 4. pair-level heads。
    pae_logits = LinearNoBias_pae(LayerNorm_pae(z_pair))
    pde_logits = LinearNoBias_pde(LayerNorm_pde(z_pair + transpose_token_pair(z_pair)))

    # 5. token single broadcast 回 atom。
    a_atom = broadcast_token_to_atom(
        x_token=s_single,
        atom_to_token_idx=atom_to_token_idx,
    )

    # 6. atom-level heads。权重按 atom 在 token 内的位置选择。
    plddt_logits = einsum(
        "...nc,ncb->...nb",
        LayerNorm_plddt(a_atom),
        plddt_weight[atom_to_tokatom_idx],
    )
    resolved_logits = einsum(
        "...nc,ncb->...nb",
        LayerNorm_resolved(a_atom),
        resolved_weight[atom_to_tokatom_idx],
    )

    return plddt_logits, pae_logits, pde_logits, resolved_logits
```

这段实现暴露了一个重要事实：confidence head 的输入坐标只通过 representative atom distance 注入 pair stream；pLDDT/resolved 的 atom-level 差异来自 token broadcast 后按 atom slot 选择不同输出权重，而不是每个 atom 都重新做 full atom attention。

#### 8.5.3 `DistogramHead`

模块职责：把 trunk pair 表示转成 token representative atom 的距离分布 logits。

```python
class DistogramHead:
    def forward(z):                         # [N_token,N_token,c_z]
        logits = Linear(z)                  # [N_token,N_token,no_bins]
        logits = logits + transpose_token_pair(logits)
        return logits
```

它没有额外非线性，也没有 sample 维。梯度从 distogram CE 直接进入 trunk pair `z` 和这一层 linear。因为 linear 用 zero initializer，训练初期 logits 主要由对称化后的零输出开始，distogram loss 初始接近均匀分布 CE。

#### 8.5.4 `permute_label_to_match_mini_rollout`

模块职责：训练时用 no-grad mini-rollout 的预测结构，把 label 的链顺序和对称 atom 顺序改成更匹配当前模型预测的方向。它改变后续 diffusion loss 的 label。

```python
def permute_label_to_match_mini_rollout(
    mini_coord,              # [1,N_atom,3]
    input_feature_dict,
    label_dict,              # cropped label，后续 loss 使用
    label_full_dict,         # full label，chain permutation 使用
):
    assert mini_coord.dim() == 3
    pred_single = mini_coord[0]
    log_dict = {}

    # 1. chain permutation：用预测结构匹配 full label 链排列。
    permuted_label, chain_logs, _, _ = chain_permutation.run(
        pred_coord=pred_single,
        input_feature_dict=input_feature_dict,
        label_full_dict=label_full_dict,
        permute_label=True,
        **chain_permutation_config,
    )

    if config.chain_permutation.train.mini_rollout:
        label_dict.update(permuted_label)
        log_dict.update(prefix(chain_logs, "minirollout_perm/Chain-"))
    else:
        log_dict.update(prefix(chain_logs, "minirollout_perm/Chain.F-"))

    # 2. atom permutation：在已更新的 label 上做对称 atom 匹配。
    permuted_label, atom_logs, _ = atom_permutation.run(
        pred_coord=pred_single,
        true_coord=label_dict["coordinate"],
        true_coord_mask=label_dict["coordinate_mask"],
        ref_space_uid=input_feature_dict["ref_space_uid"],
        atom_perm_list=input_feature_dict["atom_perm_list"],
        permute_label=True,
        global_align_wo_symmetric_atom=config.atom_permutation.global_align_wo_symmetric_atom,
    )

    if config.atom_permutation.train.mini_rollout:
        label_dict.update(permuted_label)
        log_dict.update(prefix(atom_logs, "minirollout_perm/Atom-"))
    else:
        log_dict.update(prefix(atom_logs, "minirollout_perm/Atom.F-"))

    return label_dict, log_dict
```

这一步发生在 diffusion training denoise loss 之前，因此会影响 `label_dict["coordinate"]`、`coordinate_mask` 等后续 loss target。它是训练目标定义的一部分，不是纯 logging。

#### 8.5.5 `permute_diffusion_sample_to_match_label`

模块职责：对最终 predicted diffusion samples 做 per-sample permutation，使预测与 label 对齐。当前 train 阶段默认不开启 diffusion sample permutation，所以多数情况下只记录日志。

```python
def permute_diffusion_sample_to_match_label(
    input_feature_dict,
    pred_dict,               # pred_dict["coordinate"]: [N_sample,N_atom,3]
    label_dict,
    stage,                   # "train" or "test"
    permute_by_pocket=False,
):
    assert pred_dict["coordinate"].shape[-2] == label_dict["coordinate"].shape[-2]
    log_dict = {}
    permute_pred_indices = []
    permute_label_indices = []

    if stage != "train":
        # train 阶段已经在 mini-rollout 上对 label 做过 chain permutation，
        # 因此这里跳过 chain permutation。
        permuted_pred, chain_logs, chain_pred_indices, _ = chain_permutation.run(
            pred_coord=pred_dict["coordinate"],
            input_feature_dict=input_feature_dict,
            label_full_dict=label_dict,
            permute_label=False,
            max_num_chains=-1,
            permute_by_pocket=permute_by_pocket and config.chain_permutation.permute_by_pocket,
        )

        if config.chain_permutation[stage].diffusion_sample:
            pred_dict.update(permuted_pred)
            log_dict.update(prefix(chain_logs, "sample_perm/Chain-"))
            permute_pred_indices = chain_pred_indices
        else:
            log_dict.update(prefix(chain_logs, "sample_perm/Chain.F-"))

    # pocket 模式会缩小 alignment mask；普通训练/测试用全局 atom mask。
    if permute_by_pocket and config.atom_permutation.permute_by_pocket:
        chain_mask, alignment_mask = build_pocket_chain_and_alignment_mask(label_dict)
    else:
        chain_mask = 1
        alignment_mask = None

    permuted_pred, atom_logs, atom_pred_indices = atom_permutation.run(
        pred_coord=pred_dict["coordinate"],
        true_coord=label_dict["coordinate"],
        true_coord_mask=label_dict["coordinate_mask"] * chain_mask,
        ref_space_uid=input_feature_dict["ref_space_uid"],
        atom_perm_list=input_feature_dict["atom_perm_list"],
        permute_label=False,
        alignment_mask=alignment_mask,
        global_align_wo_symmetric_atom=config.atom_permutation.global_align_wo_symmetric_atom,
    )

    if permute_pred_indices:
        permute_pred_indices = compose_indices(permute_pred_indices, atom_pred_indices)
    elif atom_pred_indices is not None:
        permute_pred_indices = list(atom_pred_indices)

    if config.atom_permutation[stage].diffusion_sample:
        pred_dict.update(permuted_pred)
        log_dict.update(prefix(atom_logs, "sample_perm/Atom-"))
    else:
        log_dict.update(prefix(atom_logs, "sample_perm/Atom.F-"))

    return pred_dict, log_dict, permute_pred_indices, permute_label_indices
```

如果我们要优化 symmetry handling，需要先决定目标是哪一种：

1. 改 mini-rollout label permutation：改变训练 target，是主 diffusion loss 的定义变化。
2. 改 diffusion sample permutation：主要影响 prediction 与 label 的后处理对齐，train 默认不更新 sample。
3. 两者都改：风险最大，因为可能把 label 和 pred 的坐标系同时移动，导致 loss 与 eval 口径不一致。

#### 8.5.6 `ranking_score` 形成路径

模块职责：把 sampled structures 的 confidence summary 转成排序分数。它不是训练 loss，但决定推理 top-1。

```python
def build_ranking_score(pae_logits, coordinates, asym_id, entity_id):
    pae_prob = softmax(pae_logits, dim=-1)

    ptm = compute_ptm_from_pae_prob(pae_prob)
    iptm = compute_interface_ptm_from_pae_prob(
        pae_prob=pae_prob,
        asym_id=asym_id,
        entity_id=entity_id,
    )

    has_clash = detect_clash(coordinates)
    disorder = 0.0

    ranking_score = 0.8 * iptm \
                  + 0.2 * ptm \
                  + 0.5 * disorder \
                  - 100.0 * has_clash

    return ranking_score
```

这段路径解释了一个重要优化矛盾：当前训练主目标在 diffusion coordinate 上，top-1 排序却主要看 PAE 派生的 ipTM/PTM 和 clash。若只提升 oracle-best DockQ，不提升 PAE calibration 或 clash 率，最终 top-1 可能不动。

### 8.6 Chain/Atom permutation 内部算法

上一节描述的是调用顺序。本节补足两个 permutation 子算法本身，因为它们决定了训练 target 的坐标排列。这里尤其需要强调：当前 chain permutation 不是 Hungarian 全局二分匹配，而是 anchor alignment 后的启发式贪心匹配；atom permutation 也不是全局 atom 排列搜索，而是 residue 内对称 atom 枚举。

#### 8.6.1 chain permutation 输入 contract

chain permutation 读取的核心字段来自 atom-level feature：

```text
pred_dict:
  coordinate       [N_crop_atom,3] or [N_sample,N_crop_atom,3]
  pae_rep_atom_mask[N_crop_atom]
  mol_id           [N_crop_atom]
  mol_atom_index   [N_crop_atom]
  entity_mol_id    [N_crop_atom]
  is_ligand        [N_crop_atom]

label_full_dict:
  coordinate       [N_full_atom,3]
  coordinate_mask  [N_full_atom]
  pae_rep_atom_mask[N_full_atom]
  mol_id           [N_full_atom]
  mol_atom_index   [N_full_atom]
  entity_mol_id    [N_full_atom]
```

它先把 atom dict 通过 `pae_rep_atom_mask` 压到 token/representative atom 级别，再按 `mol_id` 分成 per-chain dict。后续匹配时只在同一 `entity_mol_id` 的 asym/molecule 间比较。

```python
def parse_atom_feature_dict(atom_features, rep_atom_mask):
    token_dict = {
        key: value[rep_atom_mask]
        for key, value in atom_features.items()
        if value is not None
    }

    asym_dict = {}
    for mol_id in unique(token_dict["mol_id"]):
        mask = token_dict["mol_id"] == mol_id
        asym_dict[mol_id] = {
            "coordinate": token_dict["coordinate"][mask],
            "coordinate_mask": token_dict.get("coordinate_mask", None)[mask],
            "mol_atom_index": token_dict["mol_atom_index"][mask],
        }

    entity_to_asym, asym_to_entity = build_entity_asym_mapping(
        entity_mol_id=token_dict["entity_mol_id"],
        asym_id=token_dict["mol_id"],
    )
    token_dict["entity_to_asym"] = entity_to_asym
    token_dict["asym_to_entity"] = asym_to_entity
    return token_dict, asym_dict
```

若 full label 中每个 `entity_mol_id` 只对应一个 `mol_id`，则没有 symmetric chain，直接返回 identity/unpermuted mapping。若 chain 数超过 `max_num_chains`，也会跳过搜索，保留原 mapping。

#### 8.6.2 chain permutation anchor 选择

anchor 选择逻辑体现了“先找最不歧义的 entity，再找最长的预测 chain”：

```python
def find_anchor_asym_chain_in_predictions():
    # 预测 chain 至少 4 个 representative tokens。
    valid_pred_asyms = [
        asym_id for asym_id in pred_asym_dict
        if len(pred_asym_dict[asym_id]["coordinate"]) >= 4
    ]

    # GT entity 至少有一个 asym 有 4 个 resolved representative tokens。
    valid_entities = []
    for entity_id, gt_asyms in label_entity_to_asym.items():
        if any(label_asym_dict[a]["coordinate_mask"].sum() >= 4 for a in gt_asyms):
            valid_entities.append(entity_id)

    valid_entity_asym = [
        (entity_id, pred_asym_id)
        for entity_id in valid_entities
        for pred_asym_id in pred_entity_to_asym.get(entity_id, [])
        if pred_asym_id in valid_pred_asyms
    ]

    # 优先 polymer：非 ligand token 超过一半，且 token 数 >= 12。
    polymer_entities = []
    for entity_id in candidate_entities(valid_entity_asym):
        is_ligand = pred_token_dict["is_ligand"][pred_entity_mol_id == entity_id]
        if sum(is_ligand) <= len(is_ligand) / 2 and len(is_ligand) >= 12:
            polymer_entities.append(entity_id)
    if polymer_entities:
        candidate_entities = polymer_entities

    # 选择 GT 中 asym 数最少的 entity，降低歧义。
    min_count = min(len(label_entity_to_asym[e]) for e in candidate_entities)
    candidate_entities = [e for e in candidate_entities if len(label_entity_to_asym[e]) == min_count]

    # 在预测侧选择最长 asym；若并列，random.choice。
    candidate_asyms = [a for e, a in valid_entity_asym if e in candidate_entities]
    max_len = max(len(pred_asym_dict[a]["coordinate"]) for a in candidate_asyms)
    candidate_asyms = [a for a in candidate_asyms if len(pred_asym_dict[a]["coordinate"]) == max_len]
    return random_choice(candidate_asyms)
```

这个实现对抗体复合物有实际含义：如果 antigen chain 很长、antibody chain 较短，anchor 可能优先选择 antigen；如果同 entity 多拷贝但长度相同，最后会有随机性。这个随机性会影响 label permutation，因此建议把 anchor entity/asym 写入训练日志。

#### 8.6.3 chain permutation best match

best match 的核心流程：

```python
def compute_best_match_heuristic():
    pred_anchor = find_anchor_asym_chain_in_predictions()
    if pred_anchor is None:
        return unpermuted_match

    anchor_entity = pred_asym_to_entity[pred_anchor]

    if find_gt_anchor_first:
        gt_anchor = random_choice(label_entity_to_asym[anchor_entity])
        candidate_anchors = pred_entity_to_asym[anchor_entity]
    else:
        candidate_anchors = label_entity_to_asym[anchor_entity]

    best_match = None
    best_rmsd = inf

    for anchor_k in candidate_anchors:
        if find_gt_anchor_first:
            cur_gt_anchor = gt_anchor
            cur_pred_anchor = anchor_k
        else:
            cur_gt_anchor = anchor_k
            cur_pred_anchor = pred_anchor

        # 用预测 anchor 中存在的 mol_atom_index，从 GT anchor 取相同 atom。
        gt_anchor_dict = select_atoms_by_mol_atom_index(
            label_asym_dict[cur_gt_anchor],
            pred_asym_dict[cur_pred_anchor]["mol_atom_index"],
        )

        mask = gt_anchor_dict["coordinate_mask"].bool()
        if not any(mask):
            continue

        rot, trans = get_optimal_transform(
            src_atoms=gt_anchor_dict["coordinate"][mask],
            tgt_atoms=pred_asym_dict[cur_pred_anchor]["coordinate"][mask],
        )

        # 把所有 GT representative atom 都变换到预测坐标系。
        aligned_label_coord = apply_transform(label_token_dict["coordinate"], rot, trans)
        write_aligned_coord_back_to_each_label_asym(aligned_label_coord)

        matched = {cur_pred_anchor: cur_gt_anchor}
        pred_remaining = all_pred_asym_except(cur_pred_anchor)
        gt_remaining = all_gt_asym_except(cur_gt_anchor)

        # 长链优先匹配。
        pred_remaining = sort_by_length_desc(pred_remaining)

        for pred_asym in pred_remaining:
            entity = pred_asym_to_entity[pred_asym]
            gt_candidates = [a for a in label_entity_to_asym[entity] if a in gt_remaining]

            gt_best = argmin_center_distance_after_alignment(
                pred_asym=pred_asym,
                gt_candidates=gt_candidates,
            )
            matched[pred_asym] = gt_best
            gt_remaining.remove(gt_best)

        total_rmsd = mean([
            rmsd_or_center_rmsd(pred_asym, gt_asym)
            for pred_asym, gt_asym in matched.items()
        ])

        if total_rmsd < best_rmsd:
            best_rmsd = total_rmsd
            best_match = matched

    return best_match if best_match is not None else unpermuted_match
```

`match_pred_asym_to_gt_asym()` 对每个候选 GT chain 的比较不是重新 SVD 对齐，而是在 anchor 已经确定的全局对齐坐标系里比较中心距离：

```python
gt_center = mean(label_asym_dict[gt_asym]["aligned_coordinate"][mask])
pred_center = mean(pred_asym_dict[pred_asym]["coordinate"][mask])
error = norm(gt_center - pred_center)
```

最后会构造 `permuted_indices`：

```python
def build_permuted_indices(pred_dict, label_full_dict, best_match):
    indices = zeros([N_pred_atom])
    full_indices = arange(N_label_atom)

    for pred_mol_id, gt_mol_id in best_match.items():
        pred_mask = pred_dict["mol_id"] == pred_mol_id
        mol_atom_index = pred_dict["mol_atom_index"][pred_mask]

        gt_mask = label_full_dict["mol_id"] == gt_mol_id
        gt_indices = select_atoms_by_mol_atom_index(
            {
                "mol_atom_index": label_full_dict["mol_atom_index"][gt_mask],
                "indices": full_indices[gt_mask],
            },
            mol_atom_index,
        )["indices"]

        indices[pred_mask] = gt_indices

    assert len(unique(indices)) == len(indices)
    return indices
```

如果新 permutation 的 aligned RMSD 没有优于 unpermuted RMSD，且 `accept_it_as_it_is=False`，代码会回退到原 mapping。这防止启发式把等价或更差的链排列强行写进 target。

#### 8.6.4 atom permutation 输入筛选

atom permutation 先按 `ref_space_uid` 切 residue，再过滤掉不适合做 permutation 的 residue：

```python
def collect_residues_with_symmetric_atoms(coord, coord_mask, ref_space_uid, atom_perm_list):
    residue_spans = split_contiguous_spans_where_ref_space_uid_changes(ref_space_uid)

    position_list = []
    coord_list = []
    coord_mask_list = []
    perm_list = []

    for start, end in residue_spans:
        res_coord_mask = coord_mask[start:end].bool()

        # 少于 3 个 resolved atoms 不做，因为后续 alignment/RMSD 不稳定。
        if sum(res_coord_mask) < 3:
            continue

        perm = tensor(atom_perm_list[start:end])          # [N_res_atom,N_perm_raw]
        perm = unique(perm, dim=-1)                       # 去重
        assert perm.min() == 0 and perm.max() == N_res_atom - 1

        identity = arange(N_res_atom)[:, None]
        is_sym_atom = perm != identity
        is_valid_perm = any(is_sym_atom & res_coord_mask[:, None], dim=0)

        if not any(is_valid_perm):
            continue

        # identity 永远放第一个，后面用 best_j > 0 判断是否真的 permuted。
        perm = concat([identity, perm[:, is_valid_perm]], dim=-1)
        perm = transpose(perm)                            # [N_perm,N_res_atom]

        position_list.append((start, end))
        coord_list.append(coord[start:end])
        coord_mask_list.append(res_coord_mask)
        perm_list.append(perm)

    return position_list, coord_list, coord_mask_list, perm_list
```

这解释了为什么某些 ligand/residue 即使有 `res_perm` 字段也不会发生 permutation：只要对称 atom 都 unresolved、resolved atom 少于 3、或 permutation 退化成 identity，就会跳过。

#### 8.6.5 atom permutation RMSD 优化

atom permutation 的主算法是：

```python
def atom_permutation(pred_coord, true_coord, true_coord_mask, ref_space_uid, atom_perm_list):
    permutation = identity_indices(N_atom)

    spans, true_res_coords, true_res_masks, per_res_perms = collect_residues_with_symmetric_atoms(...)
    if not per_res_perms:
        return permutation

    no_permute_atom_mask = build_mask_for_atoms_not_participating_in_symmetry(spans, per_res_perms)

    if alignment_mask is None:
        alignment_mask = true_coord_mask
    else:
        alignment_mask = true_coord_mask * alignment_mask

    if global_align_wo_symmetric_atom:
        alignment_mask = alignment_mask * no_permute_atom_mask

    if sum(alignment_mask) < 3:
        return permutation

    # 先把 pred 全局对齐到 true，后续 residue 内比较在这个坐标系中完成。
    _, transformed_pred_coord = self_aligned_rmsd(
        pred_pose=pred_coord,
        true_pose=true_coord,
        atom_mask=alignment_mask,
        allowing_reflection=False,
        reduce=False,
    )

    # 按 residue atom 数分桶，降低 padding 浪费。
    groups = bucket_by_residue_atom_count(
        cutoffs=[15, 30, 50, 100, 100000],
        residues=spans,
    )

    for group in groups:
        best_perm_list = optimize_per_residue_permutation_by_rmsd(
            pred_res_coords=[
                transformed_pred_coord[..., start:end, :]
                for start, end in group.spans
            ],
            true_res_coords=group.true_res_coords,
            true_res_masks=group.true_res_masks,
            per_res_perms=group.per_res_perms,
        )

        for (start, end), best_perm in zip(group.spans, best_perm_list):
            residue_indices = arange(start, end)
            permutation[..., residue_indices] = residue_indices[best_perm]

    return permutation
```

`optimize_per_residue_permutation_by_rmsd()` 会把每个 residue 的所有候选 permutation 展开，计算 masked RMSD，并选最小项：

```python
def optimize_per_residue_permutation_by_rmsd(pred_res_coords, true_res_coords, true_res_masks, per_res_perms):
    N_max_atom = max(num_atoms(res) for res in residues)

    # true 侧按每个候选 permutation 展开并 padding。
    permuted_true_coord, permuted_true_mask = collect_permuted_coords(
        true_res_coords,
        true_res_masks,
        per_res_perms,
        pad_to=N_max_atom,
    )  # [N_total_perm,N_max_atom,3], [N_total_perm,N_max_atom]

    # pred 侧按每个 residue 的候选数 repeat。
    pred_coord = repeat_each_residue_pred_by_num_perms(pred_res_coords)

    per_res_rmsd = rmsd(
        pred_pose=pred_coord,
        true_pose=permuted_true_coord,
        mask=permuted_true_mask,
        reduce=False,
    )

    i = 0
    best_permutation_list = []
    for N_perm, perm in zip(num_perms_per_residue, per_res_perms):
        cur = per_res_rmsd[..., i:i+N_perm]
        best_rmsd, best_j = min(cur, dim=-1)
        best_permutation_list.append(perm[best_j])
        i += N_perm

    return best_permutation_list
```

如果 `permute_label=True`，最后返回 `true_coord[permutation]` 和 `true_coord_mask[permutation]`；如果 `permute_label=False`，返回 `pred_coord[permutation]`。训练 mini-rollout 阶段使用前者，eval/prediction 对齐阶段可能使用后者。

#### 8.6.6 permutation 对 loss 的直接影响

从数学上看，permutation 在训练 target 中引入了一个当前模型依赖的离散选择：

```text
pi_chain, pi_atom = argmin_pi RMSD(pred_mini, pi(label))
label_used_by_loss = pi_atom(pi_chain(label_raw))
```

然后 diffusion loss 实际优化的是：

```text
L_diffusion(D_theta(x_noisy), label_used_by_loss)
```

这不是普通数据增强，而是把 GT 的等价表示选成与当前模型预测最接近的那一个。它能降低对称结构的错误惩罚，但也带来三个优化风险：

1. mini-rollout 质量差时，离散 target 可能被错误选择。
2. chain permutation 的 anchor 随机性可能让同一结构在不同 step 选择不同等价 target。
3. train 阶段默认 diffusion sample permutation 不更新预测，因此主 loss 更依赖 mini-rollout label permutation 的正确性。

建议后续把 `has_sym_chain`、`is_permuted`、`algo:better_rmsd`、`N_res_with_symmetry`、`N_res_permuted`、anchor asym/entity 写进按 dataset split 聚合的训练日志。

## 9. Loss Contract

### 9.1 loss 前置计算

`ProtenixLoss.forward` 先 no-grad 调 `calculate_label`：

```text
distance_mask = coordinate_mask[:,None] * coordinate_mask[None,:]
distance = cdist(label.coordinate,label.coordinate) * distance_mask
lddt_mask = distance < 30A for nucleotide atoms
          or distance < 15A for non-nucleotide atoms
lddt_mask 去掉 diagonal，并乘 distance_mask
```

如果 `loss_metrics_sparse_enable=False` 或 `diffusion_lddt_loss_dense=True`，`label_dict["distance"]` 会保留。当前默认使用 dense smooth lDDT。

### 9.2 当前 loss 权重

sampling wrapper 没有覆盖 `configs.loss.weight.*`。默认权重来自 `configs_base.py:395-455`，聚合方式在 `loss.py:1449-1475`：

| loss | 聚合权重 | 当前是否主导 | 备注 |
|---|---:|---|---|
| `smooth_lddt_loss` | `alpha_diffusion * smooth_lddt = 4 * 1 = 4` | 主导 | 不使用 sigma per-sample scale。 |
| `mse_loss` | `alpha_diffusion = 4` | 主导 | 内部还有 `weight_mse=1/3`；使用 sigma per-sample scale。 |
| `distogram_loss` | `0.03` | 次要 | 代表原子 token pair CE。 |
| `plddt_loss` | `1e-4` | 很小 | confidence，基于 `coordinate_mini` target。 |
| `pde_loss` | `1e-4` | 很小 | confidence，基于 `coordinate_mini` target。 |
| `resolved_loss` | `1e-4` | 很小 | confidence，坐标 mask CE。 |
| `pae_loss` | `alpha_confidence * alpha_pae = 0` | 不参与优化 | 仍可能被计算，但加权为 0 仍有 NaN 风险。 |
| `bond_loss` | `alpha_diffusion * alpha_bond = 0` | 不参与优化 | 仍可能被计算。 |

训练模式下，confidence target 坐标固定用 `pred_dict["coordinate_mini"]`；eval 模式才用 `pred_dict["coordinate"]`。diffusion per-sample scale 是：

```text
scale = (noise_level^2 + sigma_data^2) / (sigma_data * noise_level)^2
```

该 scale 传给 MSE 和 bond，不传给 smooth lDDT。

### 9.3 各 loss 的输入和梯度

| loss | 输入 | label/mask | 梯度主要流向 |
|---|---|---|---|
| smooth lDDT | `pred_dict["coordinate"] [48,N_atom,3]` | `label.coordinate`、`label.lddt_mask` | diffusion denoiser、最后 recycle trunk、distogram 不参与。 |
| MSE | `pred_dict["coordinate"]` | weighted rigid aligned GT；atom weight `1+5*DNA+5*RNA+10*ligand` | diffusion denoiser和 trunk conditioning；GT align no-grad。 |
| bond | `pred_dict["coordinate"]` | `feat_dict["bond_mask"]` 和 `distance_mask` | 当前权重 0；若启用则约束局部键长。 |
| distogram | `pred_dict["distogram"]` | representative atom pair distance bins，64 bins，2.3125-21.6875A | pair trunk `z` 和 DistogramHead。 |
| pLDDT | `pred_dict["plddt"]` | `coordinate_mini` vs label 的 bespoke atom lDDT，50 bins | ConfidenceHead；trunk detach。 |
| PDE | `pred_dict["pde"]` | representative atom pair distance error，0-32A 64 bins | ConfidenceHead；trunk detach。 |
| PAE | `pred_dict["pae"]` | predicted/true frames 下的 alignment error squared bins | 当前权重 0；ranking 依赖其推理输出但训练不直接优化。 |
| resolved | `pred_dict["resolved"]` | `coordinate_mask` 2-class CE | ConfidenceHead；trunk detach。 |

### 9.4 NaN/Inf 处理

`aggregate_losses` 会检测每个 loss 是否 NaN/Inf 并 warning，但代码没有真正 `continue` 或把该 loss 置零，后续仍会：

```python
cum_loss = cum_loss + weight * loss
```

需要注意：在 PyTorch 中 `0 * NaN` 仍是 NaN。因此即使 `pae_loss` 或 `bond_loss` 权重为 0，如果实际计算产生 NaN，也可能污染总 loss。runner 只在总 loss 非有限时跳过本轮反传，并把 loss 替换为 0 dummy loss。下一轮应把“0 权重 loss 是否仍计算”作为训练稳定性检查项。

### 9.5 Loss 聚合伪代码

#### 9.5.1 `ProtenixLoss.__init__`

模块职责：实例化各 loss，并把配置里的权重折成最终聚合权重。

```python
class ProtenixLoss:
    def __init__(configs):
        self.alpha_confidence = configs.loss.weight.alpha_confidence
        self.alpha_pae = configs.loss.weight.alpha_pae
        self.alpha_except_pae = configs.loss.weight.alpha_except_pae
        self.alpha_diffusion = configs.loss.weight.alpha_diffusion
        self.alpha_distogram = configs.loss.weight.alpha_distogram
        self.alpha_bond = configs.loss.weight.alpha_bond
        self.weight_smooth_lddt = configs.loss.weight.smooth_lddt

        self.loss_weight = {
            "plddt_loss": alpha_confidence * alpha_except_pae,
            "pde_loss": alpha_confidence * alpha_except_pae,
            "resolved_loss": alpha_confidence * alpha_except_pae,
            "pae_loss": alpha_confidence * alpha_pae,

            "mse_loss": alpha_diffusion,
            "bond_loss": alpha_diffusion * alpha_bond,
            "smooth_lddt_loss": alpha_diffusion * weight_smooth_lddt,

            "distogram_loss": alpha_distogram,
        }

        self.plddt_loss = PLDDTLoss(...)
        self.pde_loss = PDELoss(...)
        self.resolved_loss = ExperimentallyResolvedLoss(...)
        self.pae_loss = PAELoss(...)
        self.mse_loss = MSELoss(...)
        self.bond_loss = BondLoss(...)
        self.smooth_lddt_loss = SmoothLDDTLoss(...)
        self.distogram_loss = DistogramLoss(...)
```

当前 run 的主要梯度权重是 `smooth_lddt_loss=4`、`mse_loss=4`、`distogram_loss=0.03`。confidence 相关 loss 是 `1e-4` 量级，`pae_loss` 权重为 0。

#### 9.5.2 `calculate_label`

模块职责：在 loss 前预计算 GT atom distance、distance mask 和 lDDT mask。

```python
def calculate_label(feat_dict, label_dict):
    coord = label_dict["coordinate"]                 # [N_atom,3]
    coord_mask = label_dict["coordinate_mask"]       # [N_atom]

    distance_mask = coord_mask[:, None] * coord_mask[None, :]
    distance = cdist(coord, coord) * distance_mask    # [N_atom,N_atom]

    is_nucleotide = feat_dict["is_rna"].bool() + feat_dict["is_dna"].bool()
    lddt_mask = compute_lddt_mask(
        true_distance=distance,
        distance_mask=distance_mask,
        is_nucleotide=is_nucleotide,
        is_nucleotide_threshold=30.0,
        is_not_nucleotide_threshold=15.0,
    )

    label_dict["lddt_mask"] = lddt_mask
    label_dict["distance_mask"] = distance_mask

    if not configs.loss_metrics_sparse_enable or configs.loss.diffusion_lddt_loss_dense:
        label_dict["distance"] = distance

    return label_dict
```

`lddt_mask` 是 smooth lDDT 的邻域定义，核酸半径 30A，非核酸半径 15A，并且去掉不可见坐标和 diagonal。

#### 9.5.3 `ProtenixLoss.forward`

模块职责：loss 总入口。它先构造 label/pred 的派生量，再决定是否按 sample chunk 切分计算。

```python
def protenix_loss_forward(feat_dict, pred_dict, label_dict, mode="train"):
    assert mode in ["train", "eval"]

    with no_grad():
        label_dict = calculate_label(feat_dict, label_dict)

    pred_dict = calculate_prediction(pred_dict)

    diffusion_chunk_size = configs.loss.diffusion_chunk_size_outer

    if mode == "train" or diffusion_chunk_size is None:
        cum_loss, losses = calculate_losses(
            feat_dict=feat_dict,
            pred_dict=pred_dict,
            label_dict=label_dict,
            mode=mode,
        )
        return cum_loss, losses

    # eval 时如果 N_sample 太大，按 sample 维切开算 loss，再加权平均。
    N_sample = pred_dict["coordinate"].shape[-3]
    cum_loss = 0.0
    losses = {}

    for sample_slice in split_sample_dim(N_sample, diffusion_chunk_size):
        pred_i = slice_pred_dict_on_sample_dim(pred_dict, sample_slice)
        pred_i = calculate_prediction(pred_i)

        cur_num = len(sample_slice)
        loss_i, metrics_i = calculate_losses(feat_dict, pred_i, label_dict, mode)

        cum_loss += loss_i * cur_num
        losses = weighted_merge_metrics(losses, metrics_i, cur_num)

    cum_loss = cum_loss / N_sample
    losses = divide_non_per_sample_metrics(losses, N_sample)
    return cum_loss, losses
```

训练模式下当前不走 outer sample chunk 分支，主要 chunk 发生在 diffusion denoise、confidence head 或具体 loss 内部。

#### 9.5.4 `calculate_losses`

模块职责：根据 mode 和 pred_dict 中可用字段，动态组装 loss function 字典。

```python
def calculate_losses(feat_dict, pred_dict, label_dict, mode):
    if mode == "train":
        confidence_coordinate = "coordinate_mini"

        if not configs.train_confidence_only:
            sigma = pred_dict["noise_level"]          # [N_sample]
            diffusion_per_sample_scale = (
                sigma**2 + configs.sigma_data**2
            ) / (configs.sigma_data * sigma)**2
    else:
        confidence_coordinate = "coordinate"
        diffusion_per_sample_scale = None

    loss_fns = {}

    if not (configs.train_confidence_only and mode == "train"):
        # diffusion coordinate losses
        if configs.loss.diffusion_lddt_loss_dense:
            loss_fns["smooth_lddt_loss"] = lambda: smooth_lddt_loss.dense_forward(
                pred_coordinate=pred_dict["coordinate"],
                true_coordinate=label_dict["coordinate"],
                lddt_mask=label_dict["lddt_mask"],
                true_distance=label_dict.get("distance"),
                diffusion_chunk_size=configs.loss.diffusion_lddt_chunk_size,
            )
        elif configs.loss.diffusion_sparse_loss_enable:
            loss_fns["smooth_lddt_loss"] = lambda: smooth_lddt_loss.sparse_forward(
                pred_coordinate=pred_dict["coordinate"],
                true_coordinate=label_dict["coordinate"],
                lddt_mask=label_dict["lddt_mask"],
                diffusion_chunk_size=configs.loss.diffusion_lddt_chunk_size,
            )
        else:
            loss_fns["smooth_lddt_loss"] = lambda: smooth_lddt_loss(
                pred_distance=pred_dict["distance"],
                true_distance=label_dict["distance"],
                distance_mask=label_dict["distance_mask"],
                lddt_mask=label_dict["lddt_mask"],
            )

        loss_fns["bond_loss"] = lambda: bond_loss(
            pred_or_distance=pred_dict["coordinate"] or pred_dict["distance"],
            true_coordinate_or_distance=label_dict["coordinate"] or label_dict["distance"],
            distance_mask=label_dict["distance_mask"],
            bond_mask=feat_dict["bond_mask"],
            per_sample_scale=diffusion_per_sample_scale,
        )

        loss_fns["mse_loss"] = lambda: mse_loss(
            pred_coordinate=pred_dict["coordinate"],
            true_coordinate=label_dict["coordinate"],
            coordinate_mask=label_dict["coordinate_mask"],
            is_rna=feat_dict["is_rna"],
            is_dna=feat_dict["is_dna"],
            is_ligand=feat_dict["is_ligand"],
            per_sample_scale=diffusion_per_sample_scale,
        )

        if "distogram" in pred_dict:
            loss_fns["distogram_loss"] = lambda: distogram_loss(
                logits=pred_dict["distogram"],
                true_coordinate=label_dict["coordinate"],
                coordinate_mask=label_dict["coordinate_mask"],
                rep_atom_mask=feat_dict["distogram_rep_atom_mask"],
            )

    has_valid_resolution = resolution_in_range(
        feat_dict["resolution"],
        configs.loss.resolution.min,
        configs.loss.resolution.max,
    )

    if all(k in pred_dict for k in ["plddt", "pde", "pae", "resolved"]):
        with no_grad():
            coord_mask = label_dict["coordinate_mask"].bool()
            per_atom_lddt, per_atom_weight = calculate_atom_bespoke_lddt(
                pred_coordinate=detach(pred_dict[confidence_coordinate][:, coord_mask, :]),
                true_coordinate=label_dict["coordinate"][coord_mask, :],
                is_nucleotide=(feat_dict["is_rna"] + feat_dict["is_dna"])[coord_mask].bool(),
                is_polymer=1 - feat_dict["is_ligand"][coord_mask],
                rep_atom_mask=feat_dict["plddt_m_rep_atom_mask"][coord_mask].bool(),
            )

        loss_fns["plddt_loss"] = lambda: plddt_loss.forward_given_atom_lddt(
            logits=pred_dict["plddt"][:, coord_mask, :],
            per_atom_lddt=per_atom_lddt,
            per_atom_weight=per_atom_weight,
        )
        loss_fns["pde_loss"] = lambda: pde_loss(
            logits=pred_dict["pde"],
            pred_coordinate=detach(pred_dict[confidence_coordinate]),
            true_coordinate=label_dict["coordinate"],
            coordinate_mask=label_dict["coordinate_mask"],
            rep_atom_mask=feat_dict["distogram_rep_atom_mask"],
        )
        loss_fns["resolved_loss"] = lambda: resolved_loss(
            logits=pred_dict["resolved"],
            coordinate_mask=label_dict["coordinate_mask"],
        )
        loss_fns["pae_loss"] = lambda: pae_loss(
            logits=pred_dict["pae"],
            pred_coordinate=detach(pred_dict[confidence_coordinate]),
            true_coordinate=label_dict["coordinate"],
            coordinate_mask=label_dict["coordinate_mask"],
            frame_atom_index=feat_dict["frame_atom_index"],
            rep_atom_mask=feat_dict["pae_rep_atom_mask"],
            has_frame=feat_dict["has_frame"],
        )

    return aggregate_losses(loss_fns, has_valid_resolution)
```

注意：上面伪代码里 `bond_loss` 用 `pred_or_distance` 简写，真实代码根据 `diffusion_sparse_loss_enable` 选择 `sparse_forward` 或 dense distance 版本。核心结论不变：bond 和 MSE 使用 `diffusion_per_sample_scale`，smooth lDDT 不使用。

#### 9.5.5 `aggregate_losses`

模块职责：逐个执行 loss lambda，记录 metrics，并按权重加到总 loss。

```python
def aggregate_losses(loss_fns, has_valid_resolution):
    cum_loss = 0.0
    all_metrics = {}

    for loss_name, loss_fn in loss_fns.items():
        weight = self.loss_weight[loss_name]
        out = loss_fn()

        if is_tuple(out):
            loss, metrics = out
        else:
            loss, metrics = out, {}

        all_metrics.update(prefix(metrics, f"{loss_name}/"))

        if isnan(loss) or isinf(loss):
            logger.warning(f"{loss_name} loss is NaN. Skipping...")
            # 真实代码没有 continue，也没有把 loss 置零。

        if has_valid_resolution.sum() == 0 and loss_name in confidence_loss_names:
            loss = 0.0 * loss
        else:
            all_metrics[loss_name] = detach_clone(loss)
            all_metrics[f"weighted_{loss_name}"] = weight * detach_clone(loss)

        cum_loss = cum_loss + weight * loss

    all_metrics["loss"] = detach_clone(cum_loss)
    return cum_loss, all_metrics
```

这个实现对优化讨论很关键：0 权重不等于不计算，warning 不等于跳过。若某个 0 权重 loss 出 NaN，`weight * loss` 仍可能污染总 loss。

#### 9.5.6 `MSELoss`

模块职责：把 GT 刚体对齐到每个 predicted sample 后，计算加权 atom MSE。

```python
def mse_loss_forward(
    pred_coordinate,        # [N_sample,N_atom,3]
    true_coordinate,        # [N_atom,3]
    coordinate_mask,
    is_dna,
    is_rna,
    is_ligand,
    per_sample_scale,
):
    with no_grad():
        atom_weight = 1 + 5 * is_dna + 5 * is_rna + 10 * is_ligand
        atom_weight = atom_weight * coordinate_mask

        pred_masked = pred_coordinate * coordinate_mask[None, :, None]
        true_masked = true_coordinate * coordinate_mask[:, None]
        true_expanded = expand_to_sample_dim(true_masked, N_sample)

        true_aligned = weighted_rigid_align(
            x=true_expanded,
            x_target=pred_masked,
            atom_weight=atom_weight,
            stop_gradient=True,
        )

    per_atom_se = sum((pred_coordinate - true_aligned)**2, dim=-1)
    per_sample_mse = sum(atom_weight * per_atom_se, dim=-1) \
                   / (sum(coordinate_mask) + eps)

    if per_sample_scale is not None:
        per_sample_mse = per_sample_mse * per_sample_scale

    loss = weight_mse * mean_over_sample(per_sample_mse)
    return reduce(loss)
```

这说明 MSE 不惩罚全局刚体位置，而惩罚对齐后的内部几何差异。ligand atom 权重最高，核酸次之，普通 protein atom 最低。

#### 9.5.7 `SmoothLDDTLoss`

模块职责：把 predicted atom-atom distance 与 true distance 的误差转成 smooth lDDT 分数，再用 `1 - lDDT` 作为 loss。

```python
def smooth_lddt_chunk(pred_distance, true_distance, lddt_mask):
    dist_diff = abs(pred_distance - true_distance)         # [N_sample,N_atom,N_atom]

    score = 0
    for threshold in [0.5, 1.0, 2.0, 4.0]:
        score = score + 0.25 * sigmoid(threshold - dist_diff)

    lddt = sum(score * lddt_mask[None, :, :], dims=(-1,-2)) \
         / (sum(lddt_mask) + eps)                          # [N_sample]
    return lddt

def smooth_lddt_loss(pred_coordinate_or_distance, true_coordinate, lddt_mask):
    if dense_forward:
        pred_distance = cdist(pred_coordinate, pred_coordinate)
        true_distance = cached_or_compute_true_distance
    elif sparse_forward:
        pred_distance, true_distance = gather_only_pairs_selected_by_lddt_mask(...)
    else:
        pred_distance = pred_dict["distance"]
        true_distance = label_dict["distance"]

    lddt_per_sample = smooth_lddt_chunk(pred_distance, true_distance, lddt_mask)
    lddt = mean_over_sample(lddt_per_sample)
    return 1 - reduce(lddt)
```

这个 loss 对局部相对距离敏感，不需要刚体对齐，也不使用 sigma scale。它与 MSE 共同组成当前 diffusion 主目标：MSE 负责对齐后的 coordinate error，smooth lDDT 负责 pairwise distance quality。

#### 9.5.8 `DistogramLoss`

模块职责：用 GT representative atom pair distance 给 distogram logits 做分类交叉熵。

```python
def distogram_loss(logits, true_coordinate, coordinate_mask, rep_atom_mask):
    with no_grad():
        rep_coord = true_coordinate[rep_atom_mask, :]       # [N_token,3]
        gt_dist = cdist(rep_coord, rep_coord)

        boundaries = linspace(min_bin, max_bin, no_bins - 1)
        true_bin = sum(gt_dist[..., None] > boundaries, dim=-1)
        true_onehot = one_hot(true_bin, no_bins)

        token_mask = coordinate_mask[rep_atom_mask]
        pair_mask = token_mask[:, None] * token_mask[None, :]

    errors = softmax_cross_entropy(logits, true_onehot)
    loss = sum(errors * pair_mask) / (sum(pair_mask) + eps)
    return reduce(loss)
```

DistogramLoss 不看 predicted coordinate，只看 trunk pair logits 和真实 representative atom 距离。因此它给 trunk pair 表示提供一个结构先验辅助信号。

#### 9.5.9 Confidence losses

模块职责：训练 confidence head 的四类 logits。由于 head 默认 stop-gradient，这些 loss 主要更新 confidence head 自身参数。

```python
def plddt_loss(logits, pred_coordinate_for_confidence, label_dict, feat_dict):
    with no_grad():
        per_atom_lddt, per_atom_weight = calculate_atom_bespoke_lddt(
            pred_coordinate=detach(pred_coordinate_for_confidence[:, coord_mask, :]),
            true_coordinate=label_dict["coordinate"][coord_mask],
            is_nucleotide=(feat_dict["is_rna"] + feat_dict["is_dna"])[coord_mask],
            is_polymer=1 - feat_dict["is_ligand"][coord_mask],
            rep_atom_mask=feat_dict["plddt_m_rep_atom_mask"][coord_mask],
        )
        true_bins = bins_from_lddt(per_atom_lddt, per_atom_weight)

    ce = softmax_cross_entropy(logits[:, coord_mask, :], true_bins)
    return mean_over_sample_and_atom(ce)

def pde_loss(logits, pred_coordinate_for_confidence, true_coordinate, rep_atom_mask):
    with no_grad():
        pred_rep = pred_coordinate_for_confidence[:, rep_atom_mask, :]
        true_rep = true_coordinate[rep_atom_mask, :]
        dist_error = abs(cdist(pred_rep, pred_rep) - cdist(true_rep, true_rep)[None])
        true_bins = bin_distance_error(dist_error, min_bin=0, max_bin=32, no_bins=64)
        pair_mask = build_rep_pair_mask(...)

    ce = softmax_cross_entropy(logits, true_bins)
    return masked_mean_over_sample_and_pair(ce, pair_mask)

def pae_loss(logits, pred_coordinate_for_confidence, true_coordinate, frame_atom_index, has_frame):
    with no_grad():
        pred_frames = gather_pred_frames(pred_coordinate_for_confidence, frame_atom_index, has_frame)
        true_frames = gather_true_frames(true_coordinate, frame_atom_index, has_frame)
        squared_pae = compute_alignment_error_squared(
            pred_coordinate=pred_rep_coords,
            true_coordinate=true_rep_coords,
            pred_frames=pred_frames,
            true_frames=true_frames,
        )
        true_bins = bin_squared_pae(squared_pae)
        frame_token_pair_mask = build_frame_token_pair_mask(...)

    ce = softmax_cross_entropy(logits[:, has_frame, :, :], true_bins)
    return masked_mean_over_sample_frame_token(ce, frame_token_pair_mask)

def resolved_loss(logits, coordinate_mask):
    true_resolved = one_hot(coordinate_mask.long(), 2)
    ce = softmax_cross_entropy(logits, true_resolved[None, :, :])
    return mean_over_sample_and_atom(ce)
```

训练模式下 `pred_coordinate_for_confidence = pred_dict["coordinate_mini"]`。这意味着 confidence target 的坐标误差来自 20-step mini-rollout，不是 48 个 single-step denoise sample。eval 模式才使用 `pred_dict["coordinate"]`。

### 9.6 Diffusion loss 的数学实现细节

这一节把当前训练真正反传的 diffusion objective 写成公式。记：

```text
N = N_atom
K = N_sample = diffusion_batch_size，当前 run 通常是 48
x* ∈ R^{N×3}                 permutation 后的 GT coordinate
m ∈ {0,1}^N                  coordinate_mask
sigma_data = 16
sigma_k                      第 k 个 noisy sample 的噪声强度
eps_k ∼ N(0,I)               第 k 个 sample 的 atom-level Gaussian noise
R_k,T_k                      centre_random_augmentation 的随机刚体变换
x_aug,k = R_k x* + T_k       随机增强后的 GT
x_noisy,k = x_aug,k + sigma_k eps_k
x_hat,k = D_theta(x_noisy,k, sigma_k, conditioning)
```

训练不是对完整 sampler 反传。完整 sampler 只在 no-grad mini-rollout 里生成 `coordinate_mini`。主 diffusion loss 的 `x_hat,k` 来自 `sample_diffusion_training()` 的单步 denoise。

#### 9.6.1 训练噪声采样

训练噪声强度是 log-normal：

```text
z_k ∼ N(0,1)
sigma_k = sigma_data * exp(p_mean + p_std * z_k)
p_mean = -1.2
p_std = 1.5
sigma_data = 16
```

对应代码：

```python
sigma = exp(randn(size=[K]) * p_std + p_mean) * sigma_data
noise = randn_like(x_gt_augment) * sigma[:, None, None]
x_noisy = x_gt_augment + noise
```

这里每个 sample 共享同一个 crop 和 trunk embedding，但有独立的 `sigma_k` 和独立的 Gaussian noise。

#### 9.6.2 denoiser 的 EDM 输出公式

`DiffusionModule.forward()` 不是直接输出噪声，也不是直接输出坐标 residual。它先缩放输入，再由网络预测 `r_update`，最后按 EDM 系数组合成去噪坐标。

输入缩放：

```text
r_noisy,k = x_noisy,k / sqrt(sigma_data^2 + sigma_k^2)
```

网络：

```text
r_update,k = F_theta(
    r_noisy,k,
    log(sigma_k / sigma_data) / 4,
    s_inputs,
    s_trunk,
    z_trunk,
    reference atom features,
    local atom pair features
)
```

输出组合：

```text
s_k = sigma_k / sigma_data
c_skip,k = 1 / (1 + s_k^2)
c_out,k = sigma_k / sqrt(1 + s_k^2)
x_hat,k = c_skip,k * x_noisy,k + c_out,k * r_update,k
```

代码写法等价于：

```python
s_ratio = sigma / sigma_data
x_denoised = (
    1 / (1 + s_ratio**2) * x_noisy
    + sigma / sqrt(1 + s_ratio**2) * r_update
)
```

所以后续所有 diffusion coordinate loss 都作用在 `x_hat,k` 上。

#### 9.6.3 per-sample 噪声尺度权重

训练模式下，MSE 和 bond loss 会乘以：

```text
w_sigma,k = (sigma_k^2 + sigma_data^2) / (sigma_data * sigma_k)^2
          = 1 / sigma_data^2 + 1 / sigma_k^2
```

代码：

```python
diffusion_per_sample_scale = (
    pred_dict["noise_level"] ** 2 + configs.sigma_data ** 2
) / (configs.sigma_data * pred_dict["noise_level"]) ** 2
```

注意这个 scale 只传给：

```text
MSELoss.forward(..., per_sample_scale=diffusion_per_sample_scale)
BondLoss.forward/sparse_forward(..., per_sample_scale=diffusion_per_sample_scale)
```

不传给：

```text
SmoothLDDTLoss
DistogramLoss
PLDDT/PDE/PAE/resolved confidence losses
```

这意味着低噪声样本会在 MSE/bond 上获得更大权重，但 smooth lDDT 对所有 sample 的平均权重相同。

#### 9.6.4 MSELoss 数学形式

MSELoss 的第一步是按 atom 类型定义权重：

```text
a_i = 1 + 5 * is_dna_i + 5 * is_rna_i + 10 * is_ligand_i
u_i = a_i * m_i
```

然后对每个 noisy sample，把 GT 刚体对齐到预测坐标。记加权刚体对齐算子为：

```text
Align_u(x*, x_hat,k) = argmin_{R,t} Σ_i u_i || R x*_i + t - x_hat,k,i ||^2
```

代码调用 `weighted_rigid_align(x=true_coordinate, x_target=pred_coordinate, atom_weight=u, stop_gradient=True)`。返回的 aligned GT 会 detach：

```text
y_k = stopgrad(Align_u(x*, x_hat,k))
```

每个 sample 的 coordinate squared error：

```text
E_mse,k = Σ_i u_i || x_hat,k,i - y_k,i ||^2 / (Σ_i m_i + eps)
```

如果是 train：

```text
E_mse,k = w_sigma,k * E_mse,k
```

最终 MSE loss：

```text
L_mse = weight_mse * mean_k(E_mse,k)
weight_mse = 1/3
```

再乘外层聚合权重：

```text
WeightedMSE = alpha_diffusion * L_mse
alpha_diffusion = 4
```

因此当前总 loss 中 MSE 真实贡献是：

```text
4 * (1/3) * mean_k(E_mse,k)
```

其中 `E_mse,k` 已包含 atom 类型权重和可选噪声尺度权重。

两个实现细节很关键：

1. 分母是 `Σ_i coordinate_mask_i`，不是 `Σ_i u_i`。因此 ligand atom 权重会真实放大 loss，而不是只改变归一化权重分布。
2. 对齐结果 `y_k` 是 no-grad/detach，梯度只回到 `x_hat,k` 和 denoiser，不回到对齐 SVD。

#### 9.6.5 SmoothLDDTLoss 数学形式

先构造 GT 距离：

```text
d*_ij = || x*_i - x*_j ||_2
M_ij = m_i m_j
```

lDDT 邻域 mask 是非对称地按 atom i 是否核酸决定半径：

```text
nuc_i = is_rna_i or is_dna_i
C_ij = M_ij * 1[i != j] * (
    1[nuc_i and d*_ij < 30]
    or
    1[not nuc_i and d*_ij < 15]
)
```

对每个 predicted sample：

```text
d_hat,k,ij = || x_hat,k,i - x_hat,k,j ||_2
delta_k,ij = | d_hat,k,ij - d*_ij |
```

代码使用 smooth threshold，而不是 hard threshold：

```text
s(delta) = 1/4 * (
    sigmoid(0.5 - delta)
  + sigmoid(1.0 - delta)
  + sigmoid(2.0 - delta)
  + sigmoid(4.0 - delta)
)
```

每个 sample 的 lDDT：

```text
lddt_k = Σ_ij C_ij s(delta_k,ij) / (Σ_ij C_ij + eps)
```

最终 loss：

```text
L_smooth_lddt = 1 - mean_k(lddt_k)
WeightedSmoothLDDT = alpha_diffusion * smooth_lddt_weight * L_smooth_lddt
```

当前权重：

```text
alpha_diffusion = 4
smooth_lddt_weight = 1
```

实现有 dense、sparse、distance 三个路径：

```text
dense_forward:
  pred_distance = cdist(pred_coordinate, pred_coordinate)
  true_distance = cached label distance or cdist(label coordinate)

sparse_forward:
  indices = nonzero(lddt_mask)
  only gather selected pairs and compute pair distances

forward:
  expects pred_dict["distance"] and label_dict["distance"]
```

当前配置默认使用 dense smooth lDDT。它不使用 `w_sigma,k`，因此 high-noise/low-noise sample 在 smooth lDDT 上被等权平均。

#### 9.6.6 BondLoss 数学形式

BondLoss 的目标是约束 `bond_mask` 指定的 atom pair 距离。先定义：

```text
B_ij = bond_mask_ij * distance_mask_ij
d*_ij = || x*_i - x*_j ||_2
d_hat,k,ij = || x_hat,k,i - x_hat,k,j ||_2
```

dense 版本：

```text
E_bond,k = Σ_ij B_ij (d_hat,k,ij - d*_ij)^2 / (Σ_ij B_ij + eps)
```

sparse 版本只 gather `B_ij=1` 的 pairs，等价于：

```text
E_bond,k = mean_{(i,j):B_ij=1} (d_hat,k,ij - d*_ij)^2
```

训练时：

```text
E_bond,k = w_sigma,k * E_bond,k
L_bond = mean_k(E_bond,k)
WeightedBond = alpha_diffusion * alpha_bond * L_bond
```

当前 `alpha_bond=0`，因此加权贡献为 0。但真实代码仍可能执行这个 loss lambda。若内部出现 NaN，`0 * NaN` 仍可能污染总 loss，这点已在 9.4 提醒。

#### 9.6.7 diffusion 主损失合成

忽略当前权重为 0 的 bond 后，训练主 diffusion objective 可写成：

```text
L_diffusion =
    4 * L_smooth_lddt
  + 4 * L_mse
  + 0 * L_bond
```

展开 MSE 内部 `weight_mse=1/3`：

```text
L_diffusion =
    4 * (1 - mean_k(lddt_k))
  + 4/3 * mean_k( w_sigma,k * Σ_i u_i ||x_hat,k,i - y_k,i||^2 / (Σ_i m_i + eps) )
```

其中：

```text
y_k = stopgrad(weighted_rigid_align(x*, x_hat,k, u))
u_i = (1 + 5*dna_i + 5*rna_i + 10*ligand_i) * m_i
C_ij = bespoke lDDT neighborhood mask
```

这就是当前代码层面的 diffusion loss 数学实现。

从优化角度看，它有几个非常明确的偏置：

1. ligand atom 在 MSE 中权重最高，但 smooth lDDT 的 pair mask 不给 ligand 额外 atom weight。
2. MSE 做刚体对齐，smooth lDDT 不需要对齐；两者对全局 rigid motion 都不敏感。
3. MSE 对低 `sigma` 样本更敏感，smooth lDDT 不区分 `sigma`。
4. label 已经过 mini-rollout driven symmetry permutation，因此 target 是当前预测相关的等价 GT 表示。

#### 9.6.8 DistogramLoss 数学形式

Distogram 不属于 diffusion denoise coordinate loss，但它和 diffusion 共享 trunk pair 表示，是当前非 confidence 辅助监督。

代表原子：

```text
R = {i | distogram_rep_atom_mask_i = 1}
q_a = coordinate of representative atom for token a
```

bin 边界：

```text
b = linspace(2.3125, 21.6875, 63)
num_bins = 64
```

GT bin：

```text
d*_ab = || q_a - q_b ||_2
bin_ab = Σ_t 1[d*_ab > b_t]       # range 0..63
```

pair mask：

```text
P_ab = coordinate_mask_rep_a * coordinate_mask_rep_b
```

loss：

```text
L_distogram = Σ_ab P_ab CE(logits_ab, onehot(bin_ab)) / (Σ_ab P_ab + eps)
WeightedDistogram = alpha_distogram * L_distogram
alpha_distogram = 0.03
```

Distogram logits 来自 `Linear(z) + Linear(z)^T`，所以梯度直接作用于 trunk pair `z` 和 distogram head。

#### 9.6.9 Confidence loss 数学形式和 diffusion 的关系

confidence loss 不直接训练 diffusion coordinate，因为 ConfidenceHead 默认对 trunk embedding detach，且输入坐标训练时来自 no-grad mini-rollout：

```text
x_conf = coordinate_mini = no_grad(sample_diffusion(... N_step_mini_rollout=20 ...))
```

pLDDT target 使用 atom bespoke lDDT。先取 polymer representative atom 作为 m 侧：

```text
A = {j | plddt_m_rep_atom_mask_j = 1 and is_polymer_j = 1}
d_hat,k,i,j = || x_conf,k,i - x_conf,k,j ||_2
d*_i,j = || x*_i - x*_j ||_2
delta_k,i,j = |d_hat,k,i,j - d*_i,j|
```

hard lDDT pair score：

```text
h(delta) = 1/4 * (
    1[delta < 0.5]
  + 1[delta < 1.0]
  + 1[delta < 2.0]
  + 1[delta < 4.0]
)
```

locality mask 同样核酸 30A、非核酸 15A，并去 diagonal。得到：

```text
plddt_target_i = Σ_j pair_mask_i,j h(delta_i,j) / (Σ_j pair_mask_i,j + eps)
```

然后把 `[0,1]` 的 target 分到 50 bins，做 CE。

PDE target 是 representative atom pair distance error：

```text
e_k,ab = | ||x_conf,k,a - x_conf,k,b||_2 - ||x*_a - x*_b||_2 |
boundaries = linspace(0, 32, 65)
bin_k,ab = clamp(Σ_t 1[e_k,ab > boundaries_t], min=1, max=64) - 1
L_pde = masked CE over sample and token pair
```

PAE target 是 frame 下的 alignment error squared。对有 frame 的 token f：

```text
Frame_hat,k,f = frame(x_conf,k, frame_atom_index_f)
Frame*_f = frame(x*, frame_atom_index_f)
q_hat,k,t expressed in Frame_hat,k,f
q*_t expressed in Frame*_f
pae2_k,f,t = || Express(q_hat,k,t, Frame_hat,k,f)
             - Express(q*_t, Frame*_f) ||^2
```

bin 边界先 `linspace(0,32,65)` 再平方：

```text
boundaries2 = linspace(0, 32, 65)^2
bin_k,f,t = clamp(Σ_b 1[pae2_k,f,t > boundaries2_b], min=1, max=64) - 1
```

resolved loss 是：

```text
L_resolved = CE(resolved_logits_i, onehot(coordinate_mask_i, 2))
```

当前聚合权重：

```text
plddt/pde/resolved: 1e-4
pae: 0
```

所以 confidence 主要是弱监督 calibration，且不反向训练 trunk/diffusion 主干。ranking 又主要看 PAE 派生的 PTM/IPTM，这就是当前 top-1 ranking 和主 diffusion objective 之间的结构性错配。

#### 9.6.10 一个 batch 的完整数学链

把训练一步合起来，真实代码的数学链是：

```text
1. 从 dataloader 得到 crop:
   features, label_raw, label_full

2. Pairformer:
   s_inputs, s, z = Trunk(features, N_cycle=random[1,N_cycle])

3. no-grad mini-rollout:
   x_mini = Sampler(D_theta, s_inputs, s, z, N_step=20, N_sample=1)

4. symmetry target selection:
   label = PermuteLabelToMatch(x_mini, label_raw, label_full)
   x* = label.coordinate

5. confidence logits:
   plddt, pae, pde, resolved = ConfidenceHead(x_mini, detach(s_inputs,s,z))

6. single-step diffusion training:
   for k in 1..K:
       sigma_k ~ LogNormal
       x_aug,k = random_rigid_augment(x*)
       x_noisy,k = x_aug,k + sigma_k eps_k
       x_hat,k = D_theta(x_noisy,k, sigma_k, s_inputs, s, z)

7. auxiliary trunk head:
   distogram_logits = DistogramHead(z)

8. total weighted loss:
   L = 4 * L_smooth_lddt(x_hat, x*)
     + 4 * L_mse(x_hat, x*, sigma)
     + 0.03 * L_distogram(z, x*)
     + 1e-4 * (L_plddt + L_pde + L_resolved)
     + 0 * L_pae
     + 0 * L_bond
```

这里 `x_mini` 和 `x_hat` 是两个不同分支：`x_mini` 无梯度，用于 label permutation 和 confidence target；`x_hat` 有梯度，用于主 diffusion loss。

## 10. 训练-推理-评估闭环

### 10.1 训练内 eval 被关闭

`run_sab23h2_sampling_finetune.sh:32-36` 和 `finetune_demo.sh:154-158` 设置：

```text
eval_interval=0
eval_at_end=false
eval_only=false
```

所以 `runner/train.py` 主循环不会周期性跑 eval，也不会在末尾自动 eval。SAb23H2 评估完全由 checkpoint queue、inference、DockQ、top-k 分析脚本完成。这降低训练耦合和资源占用，但训练期间只能看 loss、runtime、dataset window、MSA debug 和 checkpoint 后处理，不能即时看到 DockQ。

### 10.2 checkpoint eval queue

`run_sab23h2_sampling_checkpoint_eval_queue.sh:1-102` 默认查找：

```text
249,499,749,999,1249,1499,1749,1999
```

每个 checkpoint 路径：

```text
${train_run_dir}/checkpoints/${step}_ema_0.999.pt
```

找到后调用 `run_sab23h2_infer_checkpoint.sh` 推理，再用 `eval_sab23h2_dockq_with_extra_runs.py` 追加 run mapping，最后 `analyze_sab23h2_topk.py` 计算 top-k/oracle 指标。

同样要注意路径：eval queue 的 `repo_dir` 也硬编码到 `Protenix_train`，而 `run_sab23h2_infer_checkpoint.sh` 自身默认是 `Protenix_train2`。如果 queue 从 `Protenix_train2` 脚本启动，实际调用的 infer 脚本可能来自 `Protenix_train`。

### 10.3 SAb23H2 inference 配置

`run_sab23h2_infer_checkpoint.sh:47-62`：

```text
model_name=protenix-v2
seeds=31415,31416,31417,31418,31419
model.N_cycle=10
sample_diffusion.N_sample=5
sample_diffusion.N_step=200
triangle_attention=cuequivariance
triangle_multiplicative=cuequivariance
use_msa=true
msa_pair_as_unpair=true
use_template=true
use_rna_msa=false
```

每个 target 产生 5 seeds * 5 samples = 25 个候选。训练和推理的主要错配：

| 维度 | 训练 | 推理/评估 |
|---|---:|---:|
| recycle | 随 step 在 1-4 间采样，最后 cycle 反传 | 固定 10 |
| diffusion sampling steps | mini-rollout 20；training objective 单步 denoise | 200 |
| diffusion candidates | 每 crop 48 noisy samples用于 loss | 每 target 25 final candidates |
| MSA cutoff | train 2048 | test/inference 16384 |
| paired MSA | finetune 脚本相关设置需核对 | inference 显式 `--msa_pair_as_unpair true` |
| template | train 使用 dataset featurizer 输出 | inference `use_template=true` |

错配不一定错误，但必须显式承认。短 finetune 的目标是让 denoiser/trunk 在训练预算较低的情况下改善最终 200-step sampling 的 oracle-best，这需要靠 checkpoint evaluation 验证，而不能仅从训练 loss 推断。

### 10.4 DockQ/top-k 指标

`analyze_sab23h2_topk.py` 的核心规则：

1. 读取一个或多个 `per_model_dockq.csv`。
2. 只保留 `dockq_status == "OK"` 且 `immune_other_dockq`、`ranking_score` 可解析的行。
3. 按 `(ranking_score, immune_other_dockq, -seed, -sample)` 降序排序。
4. 每个 `(run_label,target)` 要求 `expected_models=25`。
5. 对 `k=1,3,5,10,25` 计算：

```text
topk_best_dockq = max immune_other_dockq among ranked[:k]
topk_success = topk_best_dockq > 0.23
oracle_best_dockq = max immune_other_dockq among all 25
oracle_success = oracle_best_dockq > 0.23
gap_to_oracle = oracle_best_dockq - topk_best_dockq
```

Phase A sampling finetune 的主指标应是 `oracle_best_mean_dockq` 和 `oracle_success_rate`，因为目标是提高 25 个候选里“采到好结构”的概率。`top1/top5/gap_to_oracle` 用来判断 ranking 压力是否变大。如果 oracle 提升而 top-1 不提升，不能直接否定 Phase A；但如果 gap 变大，Phase B 需要更强 reranking/confidence 方案。

### 10.5 推理和 eval 内部伪代码

训练和推理的区别不只是 `model.eval()`。推理路径会完整执行 diffusion sampler、confidence summary 和 ranking；eval 模式在有 label 时还会做 prediction permutation。这里把 `main_inference_loop` 写成代码级伪代码，便于讨论 top-k、ranking 和 DockQ 之间的关系。

#### 10.5.1 `Protenix.forward(mode="inference"|"eval")`

```python
def forward(input_feature_dict, label_full_dict, label_dict, mode):
    input_feature_dict = RelativePositionEncoding.generate_relp(input_feature_dict)
    input_feature_dict = update_input_feature_dict(input_feature_dict)

    if mode == "inference":
        pred_dict, log_dict, time_tracker = main_inference_loop(
            input_feature_dict=input_feature_dict,
            label_dict=None,
            N_cycle=self.N_cycle,
            N_model_seed=self.N_model_seed,
            symmetric_permutation=None,
            chunk_size=configs.infer_setting.chunk_size,
        )

    elif mode == "eval":
        # eval 要求 label_dict 和 label_full_dict 是同一个 full-complex 尺寸。
        label_dict.update(label_full_dict)
        pred_dict, log_dict, time_tracker = main_inference_loop(
            input_feature_dict=input_feature_dict,
            label_dict=label_dict,
            N_cycle=self.N_cycle,
            N_model_seed=1,
            symmetric_permutation=symmetric_permutation,
            chunk_size=configs.infer_setting.chunk_size,
        )

    log_dict["time"] = time_tracker
    return pred_dict, label_dict, log_dict
```

`inference` 没有 label，不做 permutation；`eval` 有 label，会在坐标和 heads 生成后对 pred 做 permutation，以便 metric 与 label 对齐。

#### 10.5.2 `main_inference_loop`

```python
def main_inference_loop(input_feature_dict, label_dict, N_cycle, mode):
    pred_dict = {}
    time_tracker = {}

    # 1. trunk。
    s_inputs, s, z = get_pairformer_output(
        input_feature_dict=input_feature_dict,
        N_cycle=N_cycle,
        chunk_size=configs.infer_setting.chunk_size,
        mc_dropout=maybe_enabled,
    )

    # 2. 删除推理后 diffusion/confidence 不再需要的大字段，降低显存。
    for key in input_feature_dict.keys():
        if key.startswith("template_") or key in [
            "msa",
            "has_deletion",
            "deletion_value",
            "profile",
            "deletion_mean",
        ]:
            del input_feature_dict[key]

    # 3. 构造推理 noise schedule。
    N_sample = configs.sample_diffusion["N_sample"]
    N_step = configs.sample_diffusion["N_step"]
    noise_schedule = InferenceNoiseScheduler(
        s_max=160,
        s_min=4e-4,
        rho=7,
        sigma_data=16,
    )(N_step=N_step)

    # 4. 可选缓存 diffusion conditioning 和 atom encoder 中不随 sample 改变的变量。
    if enable_diffusion_shared_vars_cache:
        pair_z = diffusion_conditioning.prepare_cache(input_feature_dict["relp"], z, False)
        p_lm, c_l = atom_attention_encoder.prepare_cache(
            ref_pos=input_feature_dict["ref_pos"],
            ref_charge=input_feature_dict["ref_charge"],
            ref_mask=input_feature_dict["ref_mask"],
            ref_element=input_feature_dict["ref_element"],
            ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            d_lm=input_feature_dict["d_lm"],
            v_lm=input_feature_dict["v_lm"],
            pad_info=input_feature_dict["pad_info"],
            z=pair_z,
        )
        z_trunk_for_diffusion = None
    else:
        pair_z = None
        p_lm, c_l = None, None
        z_trunk_for_diffusion = z

    # 5. diffusion sampler 输出 N_sample 套 atom 坐标。
    pred_dict["coordinate"] = sample_diffusion(
        denoise_net=diffusion_module,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z_trunk_for_diffusion,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        noise_schedule=noise_schedule,
        N_sample=N_sample,
        enable_efficient_fusion=enable_efficient_fusion,
    )

    # 6. distogram 只在推理中转成 contact_probs，减少输出维度。
    distogram_logits = distogram_head(z)
    pred_dict["contact_probs"] = compute_contact_prob(
        distogram_logits,
        min_bin=2.3125,
        max_bin=21.6875,
        no_bins=64,
    )

    # 7. confidence head 对每个 sampled coordinate 生成 logits。
    pred_dict["plddt"], pred_dict["pae"], pred_dict["pde"], pred_dict["resolved"] = run_confidence_head(
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        x_pred_coords=pred_dict["coordinate"],
        chunk_size=configs.infer_setting.chunk_size,
    )

    # 8. eval 模式有 label 时，permutation 会同步重排 coordinate 和 heads。
    if label_dict is not None and symmetric_permutation is not None:
        pred_dict, perm_log = symmetric_permutation.permute_inference_pred_dict(
            input_feature_dict=input_feature_dict,
            pred_dict=pred_dict,
            label_dict=label_dict,
            permute_by_pocket=("pocket_mask" in label_dict and "interested_ligand_mask" in label_dict),
        )
        log_dict.update(perm_log)

    # 9. summary confidence 和 full_data 最后计算，保证使用 permutation 后的坐标和 logits。
    pred_dict["summary_confidence"], pred_dict["full_data"] = compute_full_data_and_summary(
        configs=configs,
        pae_logits=pred_dict["pae"],
        plddt_logits=pred_dict["plddt"],
        pde_logits=pred_dict["pde"],
        contact_probs=pred_dict.get("per_sample_contact_probs", pred_dict["contact_probs"]),
        token_asym_id=input_feature_dict["asym_id"],
        token_has_frame=input_feature_dict["has_frame"],
        atom_coordinate=pred_dict["coordinate"],
        atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
        atom_is_polymer=1 - input_feature_dict["is_ligand"],
        N_recycle=N_cycle,
        interested_atom_mask=label_dict.get("interested_ligand_mask") if label_dict else None,
        return_full_data=True,
        mol_id=input_feature_dict["mol_id"] if mode != "inference" else None,
        elements_one_hot=input_feature_dict["ref_element"] if mode != "inference" else None,
    )

    return pred_dict, log_dict, time_tracker
```

#### 10.5.3 推理 sampler 的数学步骤

推理使用 `sample_diffusion()`，从最高噪声开始走 `N_step` 个 Euler 更新。noise schedule：

```text
t_i = sigma_data * (
    s_max^(1/rho)
  + i/N_step * (s_min^(1/rho) - s_max^(1/rho))
)^rho
t_N = 0
```

默认：

```text
s_max=160
s_min=4e-4
rho=7
sigma_data=16
```

sampler 内部：

```python
def sample_diffusion(noise_schedule, N_sample):
    x = noise_schedule[0] * randn([N_sample, N_atom, 3])

    for step_i, (c_tau_last, c_tau) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
        x = centre_random_augmentation(x, N_sample=1).squeeze(sample_dim)

        gamma = gamma0 if c_tau > gamma_min else 0
        t_hat = c_tau_last * (1 + gamma)

        delta_noise = sqrt(t_hat**2 - c_tau_last**2)
        x_noisy = x + noise_scale_lambda * delta_noise * randn_like(x)

        x_denoised = D_theta(x_noisy, t_hat, conditioning)

        # 当前实现用 x_noisy 而不是 AF3 文档中可能有歧义的 x_l_hat。
        delta = (x_noisy - x_denoised) / t_hat
        dt = c_tau - t_hat
        x = x_noisy + step_scale_eta * dt * delta

    return x
```

默认参数：

```text
gamma0 = 0.8
gamma_min = 1.0
noise_scale_lambda = 1.003
step_scale_eta = 1.5
```

这和训练主 diffusion loss 有本质区别：推理是多步采样，训练主 loss 是单步 denoise。优化时如果发现“训练 loss 下降但 200-step sample 不升”，需要分别检查单步 denoise 质量、sampler 参数、confidence ranking 和 clash。

#### 10.5.4 summary confidence 和 ranking

`compute_full_data_and_summary()` 会把 logits 转成 per-sample summary。关键输入是：

```text
pae_logits        -> ptm / iptm / chain_pair metrics
plddt_logits      -> atom/token/chain pLDDT
pde_logits        -> predicted distance error metrics
contact_probs     -> contact/interface 相关 summary
atom_coordinate   -> clash
atom_to_token_idx -> atom/token/chain 聚合
asym_id           -> chain-level grouping
```

排序分数：

```text
ranking_score = 0.8 * iptm
              + 0.2 * ptm
              + 0.5 * disorder
              - 100 * has_clash
```

当前 `disorder` 为 0，因此主要是：

```text
ranking_score ≈ 0.8 * iptm + 0.2 * ptm - 100 * has_clash
```

这意味着 top-k 评估要拆成两个问题：

1. `N_sample` 里是否至少有一个高 DockQ 结构，也就是 oracle-best。
2. `ranking_score` 是否能把高 DockQ 结构排到前面，也就是 calibration/ranking。

当前 finetune 主损失主要优化 coordinate quality，不直接优化 ranking score。PAE loss 权重为 0，confidence 其他 loss 权重极小，这是 top-1 与 oracle-best 可能脱钩的根本原因之一。

## 11. 当前实现的技术含义

### 11.1 真正改了什么

| 维度 | 改动 |
|---|---|
| 初始化 | 从已有 `999_ema_0.999.pt` 初始化 active model 和 EMA shadow。 |
| 数据分布 | 引入/提高 `sabdab-train`、`distill-sabdab` 在训练 sampling mass 中的比例。 |
| crop 尺寸 | `train_crop_size=768`，给 antibody-antigen interface 留更多 token 容量。 |
| 训练预算 | `max_steps=2000`，`checkpoint_interval=250`，短程密集筛 checkpoint。 |
| 学习率 | 默认 `3e-6`，且 warmup 覆盖整个 2000 steps，实际非常保守。 |
| train-time compute | `N_cycle=4`、train MSA cutoff 2048、mini-rollout N_step 20。 |
| eval 策略 | 训练内不 eval，独立跑 SAb23H2 25 candidates/top-k/DockQ。 |

### 11.2 没改什么

| 维度 | 当前状态 |
|---|---|
| 模型结构 | 仍是 Protenix-v2；没有新增 immune-interface 模块。 |
| 主 loss | 仍由 smooth lDDT 和 aligned MSE 主导。 |
| ranking loss | PAE 权重为 0，confidence 其他项权重 `1e-4`。 |
| interface 专用 loss | 没有 DockQ、interface contact、CDR-antigen distance、chain-pair ipTM direct supervision。 |
| 参数冻结 | 没有只微调 head 或部分 trunk；当前是全模型低学习率。 |
| 数据泄漏检查 | 有 `check_sab23h2_train_overlap.py` 可查 direct PDB ID overlap，但序列/模板/MSA 层面的 gate 仍需明确。 |

### 11.3 为什么它是 sampling finetune

主导梯度来自 48 个 noisy samples 的 denoised coordinate：

```text
weighted_smooth_lddt_loss: 4.0 * (1 - smooth_lddt)
weighted_mse_loss:        4.0 * aligned weighted MSE
weighted_distogram_loss:  0.03 * CE
confidence losses:        1e-4 scale
PAE/bond:                 0
```

这会主要改变模型在给定 trunk conditioning 下的坐标 denoise 分布，以及 trunk 表示对 denoising 的支撑。ranking score 依赖 PAE/iptm/ptm/clash，但 PAE 没有 loss 权重，confidence loss 很小且 stop-gradient。因此当前设计自然更适合提升 oracle-best，而不是直接提升 top-1。

## 12. 下一步优化的质量门槛

### 12.1 Reproducibility gate

| 风险 | 证据 | 通过条件 |
|---|---|---|
| 训练/推理 repo 不一致 | wrapper 和 queue 硬编码 `Protenix_train`，infer 默认 `Protenix_train2` | 每个 run 保存 dry-run 命令、`repo_dir`、git status、实际 `runner/train.py` 路径。 |
| checkpoint 语义误解 | `load_params_only=True`，同一 EMA path 同时初始化 active/EMA | run manifest 明确写“从 EMA 参数重新 step0 finetune”，不要写“恢复 999 step 续训”。 |
| checkpoint 编号误读 | 保存第 250 个 update 为 `249_ema_0.999.pt` | eval queue steps 与保存语义一致，并在报告中说明 zero-based 文件名。 |

### 12.2 Data distribution gate

| 问题 | 需要记录 |
|---|---|
| 名义 `TRAIN_SAMPLE_WEIGHTS` 是否变成真实 batch mass？ | replacement 后窗口级 dataset 分布。 |
| SAbDab 样本是否真的把 crop 放在 immune interface？ | crop method、interface token 数、两侧 chain 覆盖、reference chain。 |
| 失败重采样是否改变免疫样本占比？ | 每 dataset 失败率、替换率、异常类型。 |
| 训练集是否和 SAb23H2 泄漏？ | direct PDB ID、chain/entity ID、序列相似性、template 命中、MSA 资源时间界限。 |

### 12.3 Model/loss gate

如果下一轮仍是 Phase A sampling：

| knob | 当前值 | 可调方向 | 验证指标 |
|---|---:|---|---|
| `TRAIN_SAMPLE_WEIGHTS` | 外部必传 | conservative vs immune-heavy vs distill-heavy | oracle-best DockQ、oracle success、target-level delta。 |
| `train_crop_size` | 768 | 保持或增加；同时记录有效 token/atom | interface coverage、OOM、oracle-best。 |
| crop method weights | 0.2/0.4/0.4 | 提高 SpatialInterfaceCropping，但要防止过拟合局部接口 | interface token coverage、global fold regression、oracle-best。 |
| train MSA cutoff | 2048 | 对齐推理更高 cutoff 或做 ablation | MSA-depth bucket vs DockQ。 |
| `diffusion_batch_size` | 48 | sigma 分布分桶后再决定是否调 | 按 sigma bucket 的 MSE/smooth lDDT。 |
| loss 主项 | smooth lDDT/MSE | 可考虑 interface-weighted coordinate loss，但需定义 mask | immune-other DockQ、非免疫结构退化。 |

如果下一轮转 Phase B ranking：

| 当前障碍 | 需要改的实现 |
|---|---|
| PAE loss 权重为 0 | 打开或重设 `alpha_pae`，并确认 PAE target 与 interface ranking 相关。 |
| confidence loss 权重只有 `1e-4` | 单独 confidence-focused finetune 或 reranker；避免破坏 denoiser。 |
| ConfidenceHead stop-gradient | 若要让 ranking 反推 trunk，需要显式改变 `stop_gradient` 并评估稳定性。 |
| ranking 依赖全局 ipTM/ptm/clash | 对 immune-other interface 应考虑 chain-pair or interface-aware ranking feature。 |

### 12.4 Debug instrumentation gate

下一轮至少应在训练日志或 sidecar JSON 中加入：

```text
resolved_config.json
dry_run_command.txt
dataset_window_distribution.csv
sample_failure_window.csv
crop_window_stats.csv
msa_template_window_stats.csv
loss_by_dataset_window.csv
noise_level_histogram.csv
minirollout_permutation_stats.csv
checkpoint_manifest.jsonl
```

这些都能从现有代码轻量插桩得到，不需要改变模型。没有这些观测时，partner 讨论只能看到 DockQ 结果，却无法判断收益来自数据比例、crop、MSA/template、训练不稳定、还是 checkpoint 选择噪声。

### 12.5 还应补充的细节

除了用户已经指出的数据规模和卡数，我建议后续再补这些内容。它们不是背景铺垫，而是会直接影响 chief engineer / chief algorithm researcher 对实验可信度的判断。

| 需要补充 | 为什么重要 | 建议记录方式 |
|---|---|---|
| 每个 run 的硬件拓扑 | 32 卡是总卡数，但还要区分 `4 nodes * 8 GPUs`、`1 node * 32 GPUs`、GPU 型号、互联和 NCCL timeout；这些会影响吞吐和失败模式。 | run manifest 写 `WORLD_SIZE/LOCAL_WORLD_SIZE/NNODES/NPROC_PER_NODE/GPU name/CUDA/NCCL`。 |
| 实际 optimizer update 吞吐 | 同样 32 卡，如果 data loading 或 crop 很慢，实际样本数/小时差异很大。 | 每 50 steps 记录 `samples/sec`、`tokens/sec`、forward/loss/backward/optimizer 秒数。 |
| replacement 后唯一样本数 | 名义 64,000 cropped sample slots 可能重复抽到同一 row；有效覆盖度决定是否过拟合。 | 每个 run 记录 sampled `(dataset,row_id)` unique count、top repeated rows。 |
| 每 dataset 的有效 batch mass | `TRAIN_SAMPLE_WEIGHTS` 是名义权重，失败重采样和 DDP 切片会改变有效分布。 | `dataset_window_distribution.csv` 按 rank/step 汇总。 |
| crop 后有效 interface coverage | SAbDab row 是 interface 不等于 crop 中包含 antibody-antigen interface。 | 记录 crop method、reference chain、两侧 chain token 数、interface token pair count。 |
| token/atom/MSA/template 分布 | OOM、训练速度、模型信号强弱都由 crop 后 shape 决定。 | p50/p90/p95/p99 的 `N_token/N_atom/N_msa/N_template`。 |
| 数据泄漏和时间界限 | SAb23H2 是评估集，必须说明 PDB/date/template/MSA 是否可能泄漏。 | PDB ID overlap、sequence similarity、template hit date、MSA DB version。 |
| 原始训练 vs 当前 finetune 的 scale ratio | partner 会问“这个 finetune 和原训练相比到底多小”。 | 表中同时放 256 A100/约 24.8M samples vs 32 卡/64k crops/3.072M noisy samples。 |
| loss by dataset | 如果 weightedPDB loss 下降、SAbDab loss 不降，说明数据比例不是唯一问题。 | loss metrics 按 `dataset_name` 分桶。 |
| sigma bucket loss | diffusion 训练对不同 sigma 的行为不同，oracle 失败可能来自高噪声粗采样或低噪声局部 refinement。 | `noise_level_histogram.csv` + MSE/smooth lDDT by sigma bucket。 |
| mini-rollout permutation 成功率 | 对称置换如果频繁改变 label，训练 target 本身在移动。 | chain/atom permutation log 汇总：是否更新、RMSD 改善、失败数。 |
| checkpoint 选择成本 | 当前每 250 steps 存一次，评估 25 candidates；需要说明总推理成本。 | 每 checkpoint inference wall time、DockQ wall time、GPU/CPU worker 数。 |
| ranking gap 分析 | Phase A 目标是 oracle，Phase B 目标是 top-1；两者不能混为一个指标。 | top1/top5/oracle/gap_to_oracle per target。 |

如果只选最优先的三项，我会先补：

1. `run_manifest.json`：硬件、git、resolved config、数据 index checksum、checkpoint 来源。
2. `dataset_window_distribution.csv`：真实采样分布、失败重采样、crop/interface coverage。
3. `checkpoint_eval_summary.csv`：每个 checkpoint 的 oracle/top-k、ranking gap、按 target delta。

这三项能把“训练是否按预期发生”和“评估收益来自哪里”连接起来。没有它们，即使 DockQ 有提升，也很难判断下一步应该调数据、调 loss、调 ranking，还是只是保留某个 lucky checkpoint。

## 13. 当前实现审查结论

当前 SAb23H2 sampling finetune 是一个合理的 Phase A 实验框架：它用低学习率 continuation 和免疫相关数据混合，尝试提高 25 candidates 中至少一个结构命中 immune-other interface 的概率。实现上最核心的事实是，主梯度来自 diffusion coordinate loss，而不是 ranking/confidence；训练预算被脚本显式降到 4 recycle、20-step mini-rollout 和 2048 train MSA cutoff；评估又回到 10 recycle、200-step diffusion、5 seeds * 5 samples。这种设计可以成立，但必须通过 checkpoint-level oracle/top-k DockQ 来选择，而不能只看训练 loss 或 top-1。

下一步最优先的不是继续扩写文档，而是补齐三个 gate：第一，修正或记录训练和推理 repo 路径，确保结果可复现；第二，记录真实数据和 crop 分布，确认免疫接口样本真的进入梯度；第三，把 sampling 和 ranking 目标拆开评估，oracle-best 提升归 Phase A，top-1 gap 留给 Phase B 的 confidence/reranking 方案。
