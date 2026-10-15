# max_nuc_tokens 128 → 256 实验（2026-07-16）

## 结论

在 8×H100 80GB、FSDP2、crop 1024、MSA 2048、diffusion batch 32、
Pairformer/MSA checkpoint CPU offload 2/1 的 31-step benchmark 中，将
`max_nuc_tokens` 从 128 提高到 256 后训练成功完成：

- 31/31 steps，状态 `passed`；
- 无 CUDA OOM、NaN、non-finite gradient 或 optimizer skip；
- PyTorch max allocated/reserved 与 nuc128 完全相同；
- measurement-window NVML 峰值与 nuc128 完全相同，均为 60,817 MiB
  （59.3916 GiB）；
- `7d1a` 实际变为 393 protein + 256 RNA = 649 tokens、8,685 atoms。

这说明 nuc256 在当前 benchmark 上可安全使用，而且没有抬高整轮显存峰值。
但 248 个样本中只有 4 个受 cap 改变影响，因此不能把结果直接外推到高比例、
长核酸训练集；若正式训练的 NA-rich 样本明显更多，仍需专门构造压力数据集。

## 单变量设置

两轮保持相同的模型、数据集、样本身份顺序和训练配置：

- FSDP2 full shard，BF16，`c_z=256`，`c_m=128`；
- crop 1024，diffusion batch/chunk 32/4；
- Pairformer checkpoint/offload：每 2 blocks / 每 2 regions；
- MSA depth/chunk/offload：2048 / 128 / 每 1 region；
- CPU EMA，global seed 42，train sampler seed 77；
- 10 warmup + 21 measured steps；
- 官方 `protenix-v2.pt` raw/EMA 初始化。

唯一配置变化是：

```text
baseline:  max_nuc_tokens=128
candidate: max_nuc_tokens=256
```

## 实际 workload 变化

样本 PDB/rank/step 身份顺序完全相同，只有以下 4/248 条 composition 变化：

| rank / step | PDB | nuc128 | nuc256 |
|---|---|---:|---:|
| 0 / 17 | 1asz | 909 tokens，128 RNA | 1024 tokens，243 RNA |
| 3 / 9 | 4kr2 | 1014 tokens，128 RNA | 1024 tokens，138 RNA |
| 4 / 11 | 6me0 | 555 tokens，128 RNA | 683 tokens，244 RNA + 12 DNA |
| 5 / 30 | 7d1a | 521 tokens，128 RNA | 649 tokens，256 RNA |

聚合变化：

- full-run tokens：252,855 → 253,236，增加 381（0.151%）；
- measured tokens：170,945 → 171,316，增加 371（0.217%）；
- measured atoms：1,262,150 → 1,268,513，增加 6,363（0.504%）；
- full-run nucleic tokens：2,108 → 2,489；
- 30 个含核酸样本数量不变，其余 244 条 composition 完全不变；
- MSA 仍为 664 次调用，每次实际采样 2048。

Measured workload hash 只在受影响的 rank 0、4、5 上改变；rank 1、2、3、6、7
保持不变。`4kr2` 位于 warmup step 9，因此不会改变 rank 3 的 measured hash。

## 显存与速度

| 指标 | nuc128 | nuc256 | 差值 |
|---|---:|---:|---:|
| PyTorch max allocated | 54.9189 GiB | 54.9189 GiB | 0 |
| PyTorch max reserved | 57.4805 GiB | 57.4805 GiB | 0 |
| NVML peak | 60,817 MiB | 60,817 MiB | 0 |
| seconds/step | 53.5639 | 48.8088 | -8.88%（不可归因） |

nuc256 轮次的表观速度更快，但几乎所有完全相同的样本也同步变快，说明差异主要
来自共享文件页缓存或机器运行状态，而不是提高核酸 cap。并且两轮 measured
workload 的 token/atom 分母不同，所以不能把 tokens/s 变化解释为实现加速。
若需要精确测量 cap 的时间成本，应在相同热缓存状态下进行 back-to-back control，
或构造 NA-rich paired benchmark。

## 复现命令

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload

BENCHMARK_ID=<new-unique-id> \
bash run_msa2048_crop1024_nuc256_db32_31step_7d1a_fsdp2_benchmark.sh
```

## 结果位置

- nuc128 baseline：
  `/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_offload_crop1024_nuc128_db32_31step_7d1a/20260716_train2_fsdp2_crop1024_msa2048_nuc128_db32_31step_7d1a_r1/summary.md`
- nuc256 candidate：
  `/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_offload_crop1024_nuc256_db32_31step_7d1a/20260716_train2_fsdp2_crop1024_msa2048_nuc256_db32_31step_7d1a_r1/summary.md`
