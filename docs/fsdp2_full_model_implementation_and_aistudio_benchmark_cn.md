# train2 FSDP2 全模型训练实现与 AIStudio benchmark

更新日期：2026-07-14

## 1. 代码边界

本轮开发只在下面的新副本中进行：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
```

来源目录 `Protenix_train2` 保持不变。新副本建立时已核对文件数、目录数、
符号链接、Git HEAD 和排除 `.git` 后的内容差异。

本轮比较的语义是：

| variant | 训练参数范围 | 分布式策略 | optimizer state |
| --- | --- | --- | --- |
| `ddp_full` | 完整模型 | DDP | 每个 rank 完整 Adam state |
| `fsdp2_full` | 完整模型 | composable FSDP2 `fully_shard` | 随参数分片 |

两条路线都不是 post-Pairformer 冻结训练。汇总器固定检查总参数
`464,442,431`、可训练参数 `464,441,841` 和四组官方固定常量共 590 个元素，
防止把冻结范围差异误算成 FSDP 收益。

## 2. 初始化 checkpoint

matched benchmark 的 raw model 和 EMA 都从官方 Protenix v2 checkpoint 初始化：

```text
/mnt/project_abd/public_databases/protenix.20260101/checkpoint/protenix-v2.pt
```

benchmark 固定 `load_params_only=true`。因此官方 checkpoint 只负责模型初始化，
optimizer、scheduler、step 和 GradScaler 从新任务状态开始。raw 与 EMA checkpoint
路径都会写入 `run_manifest.json`、`benchmark_result.json` 和最终 summary。
入口还会在 node 0 校验该文件的 SHA256：
`8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599`。

## 3. 核心实现

### 3.1 FSDP2 wrap policy 与 DeviceMesh

实现文件：`protenix/utils/fsdp.py`。

bottom-up wrap 顺序为：

1. 完整 `msa_module` 一个 unit；
2. trunk `pairformer_stack.blocks` 的 48 个 block；
3. diffusion transformer 的 24 个 block；
4. confidence Pairformer 的 4 个 block；
5. 最后 wrap root model。

总计 77 个 child units。MSA module 整体 wrap 是刻意设计：它存在依赖输入的
early return，整体 wrap 可保证各 rank 在分支之前进入一致的 FSDP collective。

mesh 语义：

- `global`：在整个 world size 上 full shard；
- `hybrid`：二维 `(replicate, shard)` mesh，在节点内 shard、节点间 replicate；
- `auto`：单节点选择 `global`，多节点选择 `hybrid`。

例如 4 节点 × 8 GPU 时，`auto` 解析为 `(4, 8)` hybrid mesh。正式结果必须看
`benchmark_result.json` 中的 `resolved_fsdp_mesh`，不能只根据传入的 `auto` 猜测。

### 3.2 runner 初始化顺序

`runner/train.py` 保持 train2 原有数据诊断、JSONL、run status、eval 和显存日志，
只在必要接缝接入 FSDP2：

```text
创建 FP32 raw model 并放到 CUDA
→ 校验 ddp/fsdp2 与 optimizer_sharding 组合
→ FSDP2 时同步 rank-0 初始化参数和 buffer
→ 创建 DeviceMesh
→ child bottom-up fully_shard，root 最后 fully_shard
→ 创建 EMA wrapper
→ 创建 optimizer
→ 创建生命周期级 GradScaler 和 scheduler
→ 加载 checkpoint
→ 注册 EMA shadow
```

optimizer 必须在 `fully_shard` 后创建，否则 optimizer 会引用 sharding 前的参数。
默认策略仍是 `ddp`，FSDP2 必须显式开启且要求 `world_size > 1`。
`fsdp2 + zero1` 会被拒绝，因为 FSDP2 已经分片参数、梯度和 optimizer state。

### 3.3 精度

首轮验收固定：

```text
training dtype=bf16
FSDP param_dtype=fp32
FSDP reduce_dtype=fp32
reshard_after_forward=true
```

`param_dtype=fp32` 在 FSDP mixed-precision policy 中映射为 `None`，保留 FP32
参数，并继续由 Protenix 现有的 operation-level BF16 autocast 控制计算精度。
FSDP BF16 parameter mode 已提供入口，但没有作为首轮数值等价配置。

### 3.4 DTensor rank-local CPU EMA

实现文件：`runner/ema.py`。

FSDP2 参数是 DTensor。EMA 在每个 rank 只保存该 rank 的 local shard，并可放在
CPU；register/update/apply/restore 前会递归 reshard，避免 eval 后参数仍处于
unsharded 状态。DTensor apply/restore 使用 local tensor `copy_`，不会替换
Parameter 或 DTensor 对象。CPU D2H 更新采用同步 copy，避免 EMA 计算读取尚未
完成的异步传输。

DDP 与 FSDP2 benchmark 都固定 `ema_shadow_device=cpu`、decay `0.999`，从而保持
EMA 数值语义和 GPU placement 一致。DDP 每个 rank 保存完整 CPU shadow；FSDP2
每个 rank 保存 local shard，这是分片策略本身的 host-memory 差异。

### 3.5 non-finite loss/gradient

任一 rank 出现非有限 loss 时，所有 rank 通过 all-reduce 得到一致标志。坏 step
不会把 loss 换成脱离计算图的新 scalar，而是对原 loss 使用 `torch.nan_to_num`
并完成 backward，确保所有 FSDP backward hook 和 reduce-scatter 都执行。

若一个 gradient accumulation window 中任一 microbatch 的 loss 或 gradient
非有限，则整个 window：

- 完成所有必要 collective；
- 丢弃 gradient；
- 不执行 optimizer；
- 不推进 scheduler；
- 不更新 EMA。

FP16 GradScaler 是 trainer 生命周期对象。全局 skip 时各 rank 使用相同 backoff
完成 scaler 状态机，因此只有部分 shard 非有限时也不会产生 scale 分叉。
当前正式 benchmark 使用 BF16，但 FP16 连续坏窗口/好窗口已有 CPU 回归测试。

### 3.6 checkpoint

FSDP2 full model/optimizer state 由所有 rank 共同进入 PyTorch distributed
state-dict API；只有 rank 0 写现有单文件 checkpoint。主文件先写同目录临时文件，
再用 `os.replace` 原子提交；保存成功或普通 I/O 失败都会广播给所有 rank，避免一部分
rank 继续运行、另一部分 rank 已抛错。checkpoint 索引和运行状态属于审计元数据，
它们写失败只记录 warning，不会把已经原子提交的主文件误报为损坏。模型 key 保持
canonical Protenix FQN。旧 DDP checkpoint 的 `module.` 前缀会在 FSDP2
params-only 加载时删除。

DDP 对照组也在“文件加载/预校验”和“状态应用”两个阶段汇总所有 rank 的普通异常；
任一 rank 失败时所有 rank 一起退出，不会让其余 rank 继续到下一轮训练 collective。

FSDP2 optimizer 完整续训要求 checkpoint optimizer key 为 canonical FQN。
旧 DDP integer-key optimizer 不能直接恢复到 FSDP2，应使用 params-only 或
`skip_load_optimizer=true`。新 checkpoint 带 `format_version=2` 和明确的
`checkpoint_kind=raw|ema`；EMA 文件只允许 params-only 初始化，禁止误作 optimizer
完整续训。EMA 文件只聚合/写 model state，不重复保存 raw checkpoint 已包含的巨大
optimizer state、scheduler 和 GradScaler。加载前会先校验 model、optimizer、
scheduler、step 和 checkpoint 类型，
校验通过后才改写 live model。checkpoint 还保存 scheduler、step 和 GradScaler。
没有类型标签的旧文件按 `legacy` 兼容；其是否为历史 EMA 文件无法自动判定，完整续训
前需要人工确认。对文件名含 `_ema` 的 legacy 文件会直接拒绝完整续训；未知
`format_version`、未知 checkpoint 类型，以及缺少类型的 v2 文件也会在应用状态前拒绝。

当前格式会在 rank 0 CPU 聚合完整 model 和 optimizer state。benchmark 默认跳过
最终 checkpoint，避免这部分 CPU 内存和 I/O 污染测量。长期训练若 checkpoint
成为瓶颈，后续应单独实现 sharded on-disk DCP，而不是改变本轮 benchmark 口径。

当前完整 checkpoint 仍未保存 Python/NumPy/Torch/CUDA RNG、sampler epoch 和 epoch
内数据 offset，因此它能恢复参数、optimizer、scheduler、step 和 GradScaler，但不是
trajectory-exact resume。长期训练恢复时可能重新经过 epoch 前段样本；在补齐这些状态前，
不能把恢复后的数据/随机序列称为与中断前逐步连续。

### 3.7 从 fast-training 定向移植的显存设置

本轮只移植与目标口径直接相关的实现，没有整份覆盖 fast-training 分支：

- 主干 Pairformer 的 `blocks_per_ckpt=2`，48 层组成 24 个 activation-checkpoint
  region；其他 MSA、diffusion、confidence、template 模块仍保留各自原设置；
- 主干 Pairformer 的 pair transition，以及 diffusion conditioning 的
  `transition_z1/z2`，沿 pair row 维按 128 分块；crop 768 时为 6 个 row chunks；
- 每个 transition chunk 单独 activation checkpoint；
- `diffusion_batch_size=24`，现有 `diffusion_chunk_size=4`，因此为 6 个 diffusion
  sample chunks。

这些开关只改变 activation 的保存/重计算和运行批量，不引入参数，也不改变 state-dict
key。官方 `protenix-v2.pt` 仍可 strict params-only 初始化；模型总参数必须保持
`464,442,431`，其中可训练参数 `464,441,841`、固定常量 590。

### 3.8 CUDA 扩展与 benchmark 证据链

FusedLayerNorm 不再从可写源码目录加载或原地重编译。`layer_norm.py` 始终使用 PyTorch
extension cache；AIStudio launcher 将 `TORCH_EXTENSIONS_DIR` 固定到每节点独立的
`/tmp/protenix_torch_extensions/<BENCHMARK_ID>/node_<NODE_RANK>`。`finetune_demo.sh`
在启动 `torchrun` 前以单进程完成扩展 import、CUDA forward/backward 和 finite 检查，
随后本节点各 rank 只复用已验证的 `.so`。这避免旧 CUDA/PyTorch ABI 的仓库内 `.so`
被误加载，也避免多 rank 并发写 `build.ninja/.o/.so`。

测量窗口为每个 rank 记录有序 workload SHA256，以及逐批 token 数的 min/max。summary
要求每个实测 batch 都正好 768 token，不能再由 767/769 的聚合值相互抵消。每节点还会
记录完整训练源码树、官方 checkpoint、官方源索引和生成索引的 SHA256；summary 要求
所有节点一致，并要求 rank-0 manifest 的关键文件哈希与节点源码树一致。GPU compute PID
不仅要属于 benchmark 进程树，还必须覆盖 measured window 中的全部 GPU UUID。

## 4. 测试

### 4.1 CPU/静态测试

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2

CU126_PY=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
test -x "${CU126_PY}"
"${CU126_PY}" -m pytest -q \
  tests/test_fsdp2.py \
  tests/test_train_fsdp_safety.py \
  tests/test_finetune_fsdp2_args.py \
  tests/test_aistudio_fsdp2_benchmark.py \
  tests/test_aistudio_benchmark_summary.py \
  tests/test_training_optimizations.py \
  tests/test_transition_train_chunk.py \
  tests/test_transition_train_config.py \
  tests/test_pairformer_checkpoint_grouping.py \
  tests/test_layer_norm_extension_config.py \
  tests/test_long_protein_benchmark_builder.py \
  tests/test_paired_aistudio_benchmark.py

bash -n benchmark.sh finetune_demo.sh finetune_submit.sh \
  scripts/benchmark/run_aistudio_benchmark.sh
```

### 4.2 两卡 FSDP2 生命周期 smoke

必须在两张空闲 GPU 上执行：

```bash
CU126_PY=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
TORCH_LIB=/mnt/project_abd/klp/miniconda3/envs/protenix/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd)"
export TMPDIR=/tmp/ptx_f2_smoke
mkdir -p "${TMPDIR}"

CUDA_VISIBLE_DEVICES=0,1 "${CU126_PY}" -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/distributed_fsdp2_smoke.py
```

另跑一次 `reshard_after_forward=false`：

```bash
FSDP2_SMOKE_RESHARD_AFTER_FORWARD=false \
CUDA_VISIBLE_DEVICES=0,1 "${CU126_PY}" -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/distributed_fsdp2_smoke.py
```

smoke 覆盖 forward/backward、global grad clip、Adam、CPU EMA、live/EMA eval、
apply/restore、raw/EMA full state round-trip、optimizer restore 和恢复后继续训练。
它是 toy model 的 global-mesh lifecycle gate，不代替真实 Protenix benchmark。
2026-07-14 已使用系统 PyTorch `2.7.0a0`、CUDA runtime 12.9 两卡通过默认
`reshard_after_forward=true` smoke。该脚本只依赖 PyTorch；系统 Python 缺少完整
Protenix 数据/训练依赖，因此这个结果不能替代真实 464M 模型 one-step。

### 4.3 本轮已完成的验证

- 上述 12 个测试文件统一回归：92 passed（现有完整依赖 cu126 环境，仅用于代码回归）；
- 其中包含 FSDP2/EMA/checkpoint/训练安全、Transition/diffusion conditioning、
  Pairformer group-2 CUDA 数值/梯度，以及 benchmark builder/summary/paired 负例；
- 官方长蛋白索引实建/复核：128 rows，原始 token 范围 1530–1536；
- 系统 PyTorch CUDA 12.9 两卡 FSDP2 lifecycle smoke：passed；
- 系统 PyTorch CUDA 12.9/H100 node-local FusedLayerNorm sm90 编译及 CUDA
  forward/backward finite smoke：passed；
- 目标入口已切换为真实的 `protenix` cu126 runtime；CUDA 12.9 driver 负责向后兼容。
- 正式 benchmark 后使用同一 cu126 Python 复验核心 unittest：37 tests，全部通过；
- 同一 cu126 PyTorch `2.7.1+cu126` 两卡 lifecycle smoke 在
  `reshard_after_forward=true/false` 两种模式下均通过。

## 5. benchmark 固定口径

`benchmark.sh` 对两种 variant 固定以下项目：

```text
model_name=protenix-v2
full model training
official raw checkpoint = official EMA checkpoint
load_params_only=true
seed=42
dtype=bf16
PyTorch CUDA runtime=12.6（运行于 CUDA 12.9 driver）
c_z=256
MSA c_m=128
exact crop=768
diffusion_batch_size=24
diffusion_chunk_size=4
Pairformer blocks_per_ckpt=2
Pairformer + diffusion-conditioning pair Transition row chunk=128
Transition chunk activation checkpoint=true
N_cycle=4
N_step=20
lr=1e-5
EMA=0.999 on CPU
optimizer_sharding=none
train/test set=benchmark_long_protein_context_768
source rows have num_tokens>=769; measured samples must all equal 768
index rows=128, source/index SHA256 and per-run ready marker validated
DataLoader workers=4/rank
epoch_size=10000（30-step benchmark 不跨 epoch、不重建 workers）
torch multiprocessing sharing strategy=file_system
sample weight=1.0
MSA max=16384, min=16, sample cutoff=32
pair MSA as unpaired=true, shuffle=true
evaluation disabled
periodic/final checkpoint disabled
empty_cache_interval=0
```

DDP/FSDP2 对照只改变 `distributed_strategy`。FSDP 的 mesh/dtype/reshard 字段会
同时写入两条命令，但 DDP 不使用这些字段。汇总器会拒绝非 768-token 样本、错误
模型宽度/参数量、错误 checkpoint 分组、未启用 Transition chunk、非 CPU EMA、错误
初始化 checkpoint、节点代码/数据哈希漂移、测量窗口 GPU 进程覆盖不全，或 PyTorch
CUDA runtime 不是 12.6 的结果。

注意：`nvidia-smi`/driver 显示 12.9，而当前 PyTorch wheel 的
`torch.version.cuda` 是 12.6；这是受支持的“新驱动运行旧 runtime”组合。正式口径按
真实 wheel 记录为 cu126，入口在模型初始化前要求 `torch.version.cuda == "12.6"`，
并执行以下 preflight：

```bash
CU126_PY=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python
test -x "${CU126_PY}"
"${CU126_PY}" - <<'PY'
import biotite, cuequivariance, ml_collections, pandas, rdkit, torch, wandb
assert torch.cuda.is_available()
assert torch.version.cuda == "12.6", torch.version.cuda
print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())
PY
```

benchmark 入口还会校验实际导入的 `configs`/`protenix` 必须来自本代码副本，并在
torchrun 前单进程编译/加载 FusedLayerNorm，完成一次 CUDA 前后向 finite smoke。

## 6. AIStudio 执行顺序

结果入口：

```text
scripts/benchmark/run_aistudio_benchmark.sh
```

该脚本为每个节点单独保存 launcher log、NVML 显存、compute process、host
memory/load、topology、完整代码树 hash、artifact hash 和退出状态。相同
`BENCHMARK_ID` 的已有节点目录
不会被覆盖。脚本默认把 Triton cache 放到节点本地
`/tmp/protenix_triton_cache/<BENCHMARK_ID>/node_<NODE_RANK>/rank_<RANK>`，避免多个
rank 在 AIStudio 共享 home cache 并发编译时触发共享文件系统 `EBUSY`；可用
`BENCHMARK_TRITON_CACHE_ROOT` 覆盖根目录。Python `TMPDIR/TEMP/TMP` 也默认切到
节点本地短路径 `/tmp/ptx_<BENCHMARK_ID的12位hash>_<NODE_RANK>`。这里不能直接拼接
长 benchmark ID：PyTorch 还会追加 `/torch-shm-dir-XXXXXX/manager.sock`，完整路径
超过 Linux AF_UNIX socket 上限时会出现 `torch_shm_manager ... Invalid argument`。
入口拒绝解析后超过 60 字符的 runtime tmp 路径；可用 `BENCHMARK_RUNTIME_TMPDIR`
覆盖，但覆盖值也必须足够短。

AIStudio 登录环境可能把系统 PyTorch 库放在 `LD_LIBRARY_PATH` 前面。入口在激活 conda
后重新把当前 wheel 的 `torch/lib` 前置，并用 `ldd` 要求 `torch_shm_manager` 的
`libshm.so`、`libc10.so`、`libtorch_cpu.so` 都来自同一个 cu126 wheel；随后实际执行
`file_system` 共享内存和 `DataLoader(num_workers=1)` queue round-trip。CUDA 12.9
toolkit 编译扩展时固定使用 GCC/G++ 13，避免 conda GCC 15 超出 nvcc 支持范围。
FusedLayerNorm 的 node-local cache 默认为
`/tmp/protenix_torch_extensions/<BENCHMARK_ID>/node_<NODE_RANK>`，可用
`BENCHMARK_TORCH_EXTENSIONS_DIR` 覆盖。

### 6.1 dry-run

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
BENCHMARK_ID=20260713_train2_fsdp2_dry \
BENCHMARK_VARIANT=fsdp2_full \
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29700 \
NPROC_PER_NODE=8 DRY_RUN=1 SKIP_PATH_CHECK=1 \
bash benchmark.sh
```

检查最终命令中至少存在：

```text
--distributed_strategy fsdp2
--optimizer_sharding none
--ema_shadow_device cpu
--fsdp.mesh auto
--fsdp.param_dtype fp32
--fsdp.reduce_dtype fp32
--fsdp.reshard_after_forward true
--c_z 256
--model.msa_module.c_m 128
--train_crop_size 768
--diffusion_batch_size 24
--diffusion_chunk_size 4
--model.pairformer.blocks_per_ckpt 2
--model.pairformer.train_pair_transition_chunk_size 128
--model.pairformer.train_pair_transition_checkpoint_chunks true
```

### 6.2 单节点 8 卡 one-step smoke

先 DDP，再 FSDP2；使用不同 ID 和端口：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
BENCHMARK_ID=20260713_train2_1n8g_ddp_full_smoke_r1 \
BENCHMARK_VARIANT=ddp_full \
BENCHMARK_WARMUP_STEPS=0 BENCHMARK_MEASURE_STEPS=1 \
BENCHMARK_FSDP_MESH=global \
BENCHMARK_OUTPUT_ROOT=/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2 \
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29701 \
NPROC_PER_NODE=8 \
CONDA_ENV=protenix \
bash scripts/benchmark/run_aistudio_benchmark.sh
```

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
BENCHMARK_ID=20260713_train2_1n8g_fsdp2_full_smoke_r1 \
BENCHMARK_VARIANT=fsdp2_full \
BENCHMARK_WARMUP_STEPS=0 BENCHMARK_MEASURE_STEPS=1 \
BENCHMARK_FSDP_MESH=global \
BENCHMARK_OUTPUT_ROOT=/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2 \
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29702 \
NPROC_PER_NODE=8 \
CONDA_ENV=protenix \
bash scripts/benchmark/run_aistudio_benchmark.sh
```

one-step 只验证真实 Protenix lifecycle、checkpoint 初始化、optimizer/EMA 和
collective，不用于排名速度。

### 6.3 单节点 8 卡 10+20 matched benchmark

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
BENCHMARK_ID=20260713_train2_1n8g_ddp_full_10w20m_r1 \
BENCHMARK_VARIANT=ddp_full \
BENCHMARK_WARMUP_STEPS=10 BENCHMARK_MEASURE_STEPS=20 \
BENCHMARK_FSDP_MESH=global \
BENCHMARK_OUTPUT_ROOT=/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2 \
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29711 \
NPROC_PER_NODE=8 \
CONDA_ENV=protenix \
bash scripts/benchmark/run_aistudio_benchmark.sh
```

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
BENCHMARK_ID=20260713_train2_1n8g_fsdp2_full_10w20m_r1 \
BENCHMARK_VARIANT=fsdp2_full \
BENCHMARK_WARMUP_STEPS=10 BENCHMARK_MEASURE_STEPS=20 \
BENCHMARK_FSDP_MESH=global \
BENCHMARK_OUTPUT_ROOT=/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2 \
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29712 \
NPROC_PER_NODE=8 \
CONDA_ENV=protenix \
bash scripts/benchmark/run_aistudio_benchmark.sh
```

### 6.4 AIStudio 4 节点 × 8 卡

AIStudio Pytorch-DDP 外层通常提供：

```text
WORLD_SIZE=节点数
RANK=节点序号
MASTER_ADDR/MASTER_PORT=节点 rendezvous
```

不要把外层 `WORLD_SIZE` 手工改成 GPU 总数。4×8 时外层是 4，torchrun 子进程
才会形成 world size 32。

DDP job（四个节点执行同一入口）：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
: "${WORLD_SIZE:?}" "${RANK:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}"

BENCHMARK_ID=20260713_train2_4n8g_ddp_full_10w20m_r1 \
BENCHMARK_VARIANT=ddp_full \
BENCHMARK_WARMUP_STEPS=10 BENCHMARK_MEASURE_STEPS=20 \
BENCHMARK_OUTPUT_ROOT=/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2 \
NPROC_PER_NODE=8 \
CONDA_ENV=protenix \
bash scripts/benchmark/run_aistudio_benchmark.sh
```

FSDP2 hybrid job：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
: "${WORLD_SIZE:?}" "${RANK:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}"

BENCHMARK_ID=20260713_train2_4n8g_fsdp2_full_auto_10w20m_r1 \
BENCHMARK_VARIANT=fsdp2_full \
BENCHMARK_WARMUP_STEPS=10 BENCHMARK_MEASURE_STEPS=20 \
BENCHMARK_FSDP_MESH=auto \
BENCHMARK_OUTPUT_ROOT=/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2 \
NPROC_PER_NODE=8 \
CONDA_ENV=protenix \
bash scripts/benchmark/run_aistudio_benchmark.sh
```

建议再把 `BENCHMARK_FSDP_MESH=global` 作为第三个独立实验。单节点历史只验证过
global；4×8 上 global 与 hybrid 的显存/通信赢家必须实测，不能预设。

## 7. 输出与汇总

```text
${BENCHMARK_OUTPUT_ROOT}/${BENCHMARK_ID}/
  nodes/node_${NODE_RANK}_${HOSTNAME}/
    launcher.log
    nvml.csv
    compute_apps.csv
    artifact_hashes.sha256
    code_tree.sha256
    system.csv
    topology.txt
    node_status.json
  train/${RUN_NAME}_${timestamp}/
    config.yaml
    manifest/run_manifest.json
    manifest/run_status.json
    logs/*.jsonl
    benchmark/benchmark_result.json
  summary.json
  summary.md
```

单节点成功后自动汇总；多节点任务结束后手工执行：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
python scripts/benchmark/summarize_aistudio_benchmark.py \
  --profile-root /mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2/<BENCHMARK_ID>
```

两个单项都通过后，必须再做严格配对校验；只有 `paired_status=passed` 才计算相对速度与
显存收益：

```bash
python scripts/benchmark/compare_aistudio_benchmarks.py \
  --ddp-profile /mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2/<DDP_ID> \
  --fsdp2-profile /mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2/<FSDP2_ID>
```

配对器会从原始 profile 重新构建两个 summary，而不是信任已有 summary 自报的
`passed`。它会核对 resolved manifest、逐节点完整代码树、数据/checkpoint SHA256、
逐 rank 有序 workload SHA256、runtime、GPU UUID/型号/driver/显存容量；策略、输出路径、
耗时及显存峰值是允许不同的字段。

汇总同时区分：

- 同步 measured window 的 runner seconds/step、samples/s、tokens/s、atoms/s；
- measured window 重置后的 PyTorch max allocated/reserved；
- NVML measured-window 和 whole-job `memory.used`；
- job wall；
- rank imbalance；
- loss/gradient non-finite 与 optimizer skip。

NVML 是设备级采样。launcher 会按 PID 祖先关系标记 benchmark 自身的 CUDA 进程；
measured window 出现其他 GPU compute process 时，汇总器会直接判失败。正式结果仍应
使用独占 GPU。

## 8. 验收标准

只有满足以下条件的 DDP/FSDP2 结果才进入 matched 表：

1. 所有节点 `node_status.json` exit code 为 0；
2. `run_status.json` 为 `completed`；
3. world size、node count、local world size 和 rank records 完整；
4. variant 与 `distributed_strategy` 一致；
5. optimizer sharding 为 `none`；
6. EMA 都是 CPU、decay 相同；
7. 除模型源码刻意固定的 Fourier `w/b` 和 confidence bin 上下界四个常量外，
   全部参数可训练，且 DDP/FSDP2 的 trainable/fixed/total 计数一致；
8. official raw/EMA checkpoint、数据集、权重、MSA、crop、diffusion batch 相同；
9. 每个 rank 的 measured sample/token/atom、逐批 768-token min/max 和有序 workload
   SHA256 完全可比；
10. non-finite loss/gradient 和 optimizer skip 均为 0；
11. NVML 覆盖全部 GPU 和完整 measured window；
12. 无其他进程占用参与 benchmark 的 GPU，且 benchmark PID 覆盖所有实测 GPU；
13. 所有节点的完整训练源码树与 artifact SHA256 一致，DDP/FSDP2 两次运行也一致；
14. GPU UUID、型号、driver 和显存容量一致。

建议同一拓扑连续提交两种 variant，并各重复 3 次，最终报告中位数。主要报告：

| ID | strategy/mesh | topology | seconds/step | tokens/s | PyTorch peak | NVML hottest | rank imbalance | wall | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DDP | `ddp` | 记录值 | 记录值 | 记录值 | 记录值 | 记录值 | 记录值 | 记录值 | pass/fail |
| FSDP2 | `fsdp2 global/hybrid` | 记录值 | 记录值 | 记录值 | 记录值 | 记录值 | 记录值 | 记录值 | pass/fail |

## 9. 2026-07-14 cu126 单节点 8×H100 实测

### 9.1 环境与运行口径

本次实测使用单节点 8 张 `NVIDIA H100 80GB HBM3`，每卡 NVML 可见显存
81,559 MiB。软件环境为 Python 3.11.15、PyTorch `2.7.1+cu126`、PyTorch CUDA
runtime 12.6、driver `535.183.06`；`nvidia-smi` 显示的最高 CUDA 兼容版本为 12.9。
CUDA 扩展使用 GCC/G++ 13。正式运行前 8 张卡均无 compute process，两个 profile
结束后也都回到 0 MiB。

输出根目录：

```text
/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_cu126_20260714
```

除 §5 的固定口径外，本次还锁定每 rank 4 个 DataLoader workers、
`epoch_size=10000` 和 `file_system` sharing strategy。FSDP2 使用 8 卡 global
DeviceMesh，运行日志确认 77 个 bottom-up child units、456.12M child parameters、
464.44M 全模型参数和 `optimizer_sharding=none`。DDP/FSDP2 都从 §2 的 official raw
和 EMA checkpoint strict params-only 初始化；checkpoint、长蛋白源索引、生成索引及
完整训练代码树的 SHA256 都由 profile 保存并进入严格配对校验。

### 9.2 one-step lifecycle smoke

| strategy | benchmark ID | s/step | token/s | allocated peak | summary |
| --- | --- | ---: | ---: | ---: | --- |
| DDP | `20260714_train2_1n8g_ddp_full_1step_cu126_r2` | 32.2209 | 190.6836 | 36.74 GiB | 训练完成；旧版 AIStudio PID namespace 证据分类导致 summary fail |
| FSDP2 global | `20260714_train2_1n8g_fsdp2_full_1step_cu126_r1` | 44.0037 | 139.6246 | 32.88 GiB | passed |

DDP 的模型 step、有限 loss、反向、optimizer 和 CPU EMA 都正常完成；失败项仅是修复前
汇总器无法把 NVML host PID 映射到容器 `/proc`，不是训练失败。正式 10+20 profile 已
使用修复后的进程证据逻辑并严格通过。

`0 warm-up + 1 measured` 会把首次 optimizer state materialization、allocator、kernel
和 collective 冷启动计入唯一一步，因此上表只作为真实 464M 模型 lifecycle gate。
不能用 32.22 与 44.00 秒做速度排名，也不能据此计算 FSDP2 overhead。

### 9.3 10 warm-up + 20 measured matched 结果

| metric | DDP full | FSDP2 full/global |
| --- | ---: | ---: |
| benchmark ID | `20260714_train2_1n8g_ddp_full_10w20m_cu126_r1` | `20260714_train2_1n8g_fsdp2_full_10w20m_cu126_r1` |
| seconds/step | 23.334393 | 24.313794 |
| tokens/s | 263.302327 | 252.696063 |
| atoms/s | 1936.129209 | 1858.138640 |
| PyTorch max allocated | 41.45 GiB / 44,511,714,816 B | 34.57 GiB / 37,122,442,752 B |
| PyTorch max reserved | 44.76 GiB / 48,064,626,688 B | 37.58 GiB / 40,355,495,936 B |
| NVML hottest GPU | 48,031 MiB | 40,423 MiB |
| rank elapsed imbalance | 1.000938 | 1.000134 |
| launcher wall | 864 s | 907 s |
| individual summary | passed | passed |

严格配对后的相对结果：

- FSDP2 PyTorch allocated peak 减少 **6.8818 GiB / 16.6007%**；
- FSDP2 NVML hottest GPU 减少 7,608 MiB（约 7.43 GiB / 15.84%）；
- FSDP2 step time overhead 为 **4.1972%**；
- FSDP2 token throughput 是 DDP 的 **95.9718%**。

两个 measured window 都恰好包含 160 个 rank-samples、122,880 tokens 和 903,568
atoms。8 个 rank 的 sample/token/atom 范围及有序 workload SHA256 逐 rank 相同；
loss non-finite、gradient non-finite 和 optimizer skip 的每-rank 最大值都为 0。
最终累计训练 loss 为 DDP `1.1370636`、FSDP2 `1.1369445`，相差约 0.0105%。

### 9.4 严格配对产物

实际执行命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2
ROOT=/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_cu126_20260714
CU126_PY=/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python

"${CU126_PY}" scripts/benchmark/compare_aistudio_benchmarks.py \
  --ddp-profile "${ROOT}/20260714_train2_1n8g_ddp_full_10w20m_cu126_r1" \
  --fsdp2-profile "${ROOT}/20260714_train2_1n8g_fsdp2_full_10w20m_cu126_r1" \
  --output-json "${ROOT}/20260714_train2_1n8g_ddp_vs_fsdp2_10w20m_cu126_r1.json" \
  --output-md "${ROOT}/20260714_train2_1n8g_ddp_vs_fsdp2_10w20m_cu126_r1.md"
```

结果为 `paired_status=passed`、`mismatches=[]`，comparison signature：

```text
d132d72ffb556cc1adb963db65043c6a1c8884c702160b6e559856c84993c05a
```

配对器从两个 raw profile 重建 summary，而不是信任已有 summary；它核对 manifest、
完整代码和 artifact SHA256、checkpoint/数据 SHA256、逐 rank workload、runtime、
compiler、torch library、sharing strategy、GPU UUID/driver/容量和训练健康状态。

### 9.5 已知限制：atom-permutation fallback 与诊断 I/O

两个正式 launcher log 各有 208 条 `atom_permutation.py:92` warning，分布也相同：
warm-up 64 条、measured window 144 条。`atom_permutation.run` 对异常做 broad catch 后
返回空字典，因此不会丢弃 batch、loss、backward 或 optimizer step；但该次 label
只保留 chain 对齐结果，跳过 atom-level symmetric correction。本次保存的 case 可复现到
`collect_residues_with_symmetric_atoms` 的 `perm.min()==0` 断言。

异常路径还会同步调用 `save_permutation_error` 写 debug tensor。每个 profile 的共享
error directory 最多保留 50 个文件；DDP 最终为 49,129,330 B，FSDP2 为
47,123,826 B。由于 8 个 rank 共用目录、文件名只精确到秒，且采用非原子的
`exists/listdir + torch.save`，最终文件数和大小不等于真实异常次数或总写入量。

两边 warning 次数、出现阶段和 workload 相同，说明这组结果可作为当前 fallback 路径的
matched end-to-end 对照；但严格 comparator 目前不检查 permutation warning，现有 workload
SHA 也不包含 crop residue indices、`atom_perm_list` 或完整 feature tensor。因此本结果不能
称为“无 permutation 语义异常”或“纯通信/纯计算 benchmark”。在最终发表或据此选择长期
训练策略前，应先修复 atom permutation 根因，并禁用 measured window 的同步 debug save
或改为 rank-local、窗口外诊断，再按交叉顺序重复至少 3 组。

正式 10+20 为避免 checkpoint 聚合/I/O 污染而关闭最终 checkpoint；随后已经使用同一
cu126 runtime 的两卡 toy lifecycle smoke，分别在 `reshard_after_forward=true/false`
完成 raw/EMA model state、optimizer restore 和恢复后继续训练。它补充验证 checkpoint
实现，但不等价于 464M 模型 checkpoint 的性能 benchmark。

## 10. 历史预期与当前结论边界

参考 fast-training 仓库的历史 8×A100、`c_z=384`、crop 768、diffusion batch 8、
CPU EMA、30-step 结果，FSDP2 相比 DDP 的最热卡峰值低约 10.7%，稳态约慢 1.4%。
这只是历史预期方向，不能直接套用到 train2：train2 official v2 是不同宽度，本次
benchmark 使用 diffusion batch 24，GPU 和软件环境也不同。当前单节点实测结论见 §9，
必须以保存的 `summary.json` 和 paired comparison 为准。

当前正式运行环境确定为完整依赖的 PyTorch cu126，并由 CUDA 12.9 driver 向后兼容。
系统 cu129 还单独验证过 FusedLayerNorm sm90 编译和前后向。one-step DDP、one-step
FSDP2、DDP 10+20、FSDP2 10+20 已按顺序在独占资源上完成；当前结论只取严格校验通过的
10+20 summary/paired comparison，不使用 one-step 冷启动时间排名。
