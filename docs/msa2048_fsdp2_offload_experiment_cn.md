# Crop1024 + MSA2048 + FSDP2 activation offload 实验

## 目标

本代码库从 `Protenix_train2_fsdp2` 原样复制，专门验证以下正式训练负载：

- exact crop：1024 tokens
- `c_z=256`、`c_m=128`
- 有效 MSA：每次 MSA module 调用取 `min(input_depth, 2048)`
- MSA 候选池上限：16384
- MSA 计算 chunk：128 行
- diffusion batch/chunk：24/4
- Pairformer：48 blocks，每 2 blocks 一个 activation-checkpoint region
- Pairformer checkpoint 输入：每 2 个 region offload 一次
- MSA block checkpoint 输入：每个 region 都 offload
- FSDP2 full shard、CPU EMA、8 张 H100 80GB

MSA32 仅作为早期工程基线，不是正式训练设置。

## 为什么同时设置 min 和 cutoff

只设置 `sample_cutoff.train=2048` 不会固定采样 2048 行。现有采样器会先随机选择行数，再执行 cutoff。正式 profile 同时设置：

```text
data.msa.min_size.train=2048
data.msa.sample_cutoff.train=2048
data.msa.max_size.train=16384
```

因此实际采样行数严格为 `min(input_depth, 2048)`。如果输入 MSA 少于 2048，内部零 padding 不算真实 MSA；正式结果必须通过运行时 depth coverage 区分两者。

## 为什么 MSA chunk 使用 128

`msa_chunk_size` 只控制 MSA weighted averaging 和 transition 每次处理多少 MSA 行，不改变实际采样深度。2048 行、1024 tokens、`c_m=128` 时，完整 MSA embedding 的 BF16 大小约 512 MiB。若 `msa_chunk_size=2048`，transition 等中间激活会一次性展开，容易直接 OOM。

`msa_chunk_size=128` 将 2048 行拆成 16 个计算 chunk，保留全部 MSA 信息，同时降低临时激活峰值。

## activation checkpoint 输入 offload

普通 activation checkpoint 不保留 region 内部中间激活，但仍需保留每个 region 的输入供反向重算。offload 使用：

```python
torch.autograd.graph.save_on_cpu(pin_memory=True)
```

只把被选中 region 的 checkpoint 输入从 GPU 保存到 pinned CPU memory。参数、梯度和 optimizer state 不会被移动；它们仍由 FSDP2 管理。

Pairformer 的主要边界 tensor 是 `z=[1024,1024,256]`，BF16 下约 512 MiB。24 个 checkpoint region 中 offload 12 个，理论上可减少约 6 GiB/卡的常驻 GPU 激活。MSA 模块的 `(m,z)` checkpoint 边界也会全部 offload，以抵消 MSA2048 新增的常驻激活。

## 已知失败基线

原 `MSA min/cutoff=16/32` 的 800 条、100 步 FSDP2 运行在 step 39 发生真实 CUDA OOM：

- 失败位置：`smooth_lddt_loss`
- 额外申请：982 MiB
- GPU1 NVML 峰值：80067 MiB
- PyTorch allocated：76.03 GiB
- 已完成：step 0--38

新方案并不直接减少 smooth-LDDT 的瞬时工作量，而是先把仍驻留的 trunk/MSA checkpoint 边界移出 GPU，为 loss 腾出空间。

## 验收顺序

1. CPU 单元测试：配置、采样语义、offload cadence、梯度等价。
2. 单卡/单步：确认实际 MSA=2048、MSA chunk=128 和 offload 生效。
3. 8 卡 FSDP2 one-step：确认 DTensor/FSDP2 生命周期兼容。
4. 重放原 step39 重样本或等价重负载样本。
5. 10 warmup + 20 measured matched benchmark。
6. 800 条无放回、10 warmup + 90 measured，共 100 步。

若仍然 OOM，调整顺序为：进一步减小 MSA chunk、优化 smooth-LDDT loss chunk、再评估 diffusion batch；不降低 MSA 2048 目标。

## 旧 800 条索引的 MSA 容量审计

对 `benchmark_long_protein_context_1024_800` 的 800 个 PDB 做完整源 A3M 审计后得到：

- protein entities：1347，其中 1047 个达到 2048
- index 引用的 protein entities：912，其中 771 个达到 2048
- 至少一个 protein entity 达到 2048 的 PDB：706/800
- 所有 protein entities 都达到 2048 的 PDB：637/800
- missing mapping/A3M/invalid A3M：0/0/0

因此旧索引不能作为严格 MSA2048 数据集。新索引必须继续扫描官方长蛋白候选池，只接受所有 protein entities 的 pairing/non-pairing 合并去重深度均不小于 2048 的样本，同时维持 800 个 PDB 和 800 个 cluster 唯一。

## 严格 MSA2048 数据集

新索引位于：

```text
data/benchmark_long_protein_context_1024_800_msa2048/
```

构建器审计了 1006 个去重候选，接受 800 个、因 MSA 深度不足拒绝 206 个。最终索引包含 800 个唯一 PDB、800 个唯一 cluster 和 1094 个 protein entities；所有 entity 的有效 MSA 深度均不小于 2048，最小值 2222、平均值 10066.92。锁定哈希为：

```text
source index: e9fc812d2250c8d19a995dbd38991ac05204d2ace5e1e4878d4760e708ad2bc1
MSA mapping:  cb5fb9aba9372dab1bbb0ed4322c3420fa3e606b597b4857c6bb577f94687ea9
output index: dce9db1758566a67f626959ba3e1cb5b56f8f693d91f2c37a44cec99ca2c54f1
```

benchmark 启动前会用这三个哈希和 ready marker 重新验证 artifacts；不会把旧的 800 条索引误当成严格数据集。

## AIStudio 启动命令

先执行 1-step smoke：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload

BENCHMARK_ID=msa2048_smoke_$(date +%Y%m%d_%H%M%S) \
MASTER_PORT=29531 \
bash run_msa2048_fsdp2_benchmark.sh
```

smoke 通过后执行 10 warmup + 20 measured：

```bash
BENCHMARK_ID=msa2048_matched_$(date +%Y%m%d_%H%M%S) \
MSA2048_BENCHMARK_PHASE=matched \
MASTER_PORT=29532 \
bash run_msa2048_fsdp2_benchmark.sh
```

最后执行总计 100 步（10 warmup + 90 measured）：

```bash
BENCHMARK_ID=msa2048_full_$(date +%Y%m%d_%H%M%S) \
MSA2048_BENCHMARK_PHASE=full \
MASTER_PORT=29533 \
bash run_msa2048_fsdp2_benchmark.sh
```

### Crop960 严格 800 条、100 步 profile

Crop960 使用独立 workload/dataset 名称，但不会复制或重新筛选数据。它会重新校验并
复用已经审计通过的
`data/benchmark_long_protein_context_1024_800_msa2048/` artifacts；该数据集按
`min_n_token=1025` 构建，因此也严格满足 exact crop960。启动命令为：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload

BENCHMARK_ID=msa2048_crop960_full_$(date +%Y%m%d_%H%M%S) \
MASTER_PORT=29534 \
bash run_msa2048_crop960_fsdp2_benchmark.sh
```

该入口固定为 1 节点、8 GPU、FSDP2 full shard、exact crop960、800 条无放回
数据覆盖和总计 100 步（10 warmup + 90 measured）。其余 MSA、checkpoint、CPU
offload、模型宽度和 diffusion 设置与 crop1024 full profile 完全一致。

三种 profile 都固定为 1 节点、8 GPU、FSDP2 full shard、crop1024、MSA2048、MSA chunk128、Pairformer offload cadence=2 和 MSA offload cadence=1。每次必须使用新的 `BENCHMARK_ID`；结果目录不会覆盖旧实验。

## 2026-07-15 one-step 实测

运行：

```text
20260715_train2_fsdp2_crop1024_msa2048_offload_smoke_r1
```

状态为 `passed`，完整执行了 device transfer、forward、全部 loss、backward、optimizer 和 CPU EMA，随后因 `max_steps=1` 正常退出。关键结果：

- 8/8 rank 的 token 数均严格为 1024。
- 8 次 MSA module 调用的真实输入深度为 2484--13829；真实采样深度全部为 2048，below-target 为 0。
- 总参数 464,442,431；可训练参数 464,441,841。
- 单步 76.12 秒；forward 约 29.06 秒、backward 约 29.89 秒、optimizer 约 13.85 秒。
- PyTorch 最大 allocated/reserved：51.58/54.18 GiB。
- NVML 最热 GPU 峰值：56.09 GiB，距离 80GB H100 上限约 23.5 GiB。
- loss NaN、nonfinite gradient、optimizer skipped 均为 0。

结果文件：

```text
/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_offload_crop1024/20260715_train2_fsdp2_crop1024_msa2048_offload_smoke_r1/summary.md
```

这证明当前实现至少在 one-step 生命周期中能够用完整 MSA2048 负载稳定训练；它还不能替代 10+20 或 100-step 对累积峰值、重样本和持续稳定性的验证。
