# MSA 2048 行单块实验（2026-07-16）

## 语义

当前代码不能把 `msa_chunk_size` 设为空或 `None` 来关闭分块；训练路径要求正整数。
由于本 profile 的 MSA 实际采样深度和运行时 padding 上限均为2048，所以设置：

```text
msa_chunk_size=2048
```

等价于将所有 MSA 行放在一个计算块中。运行日志已直接确认：

```text
sampled_msa_shape=[2048, 1024]
msa_chunk_size=2048
msa_max_size=2048
```

若真实 MSA 少于2048，训练态会先零 padding 到2048，完成计算后再截回真实行数；
padding 不计入运行时 `sampled_depth`，也不增加有效 MSA 信息。本次严格数据集的
所有调用都实际采到了2048行，没有使用不足2048的输入。

## 单变量设置

与 chunk256 十步实验保持相同：8×H100 80GB、FSDP2、crop1024、MSA2048、
nuc256、diffusion batch32、Pairformer/MSA checkpoint CPU offload 2/1、CPU EMA、
sampler seed77，2 warm-up + 8 measured。唯一模型配置变化为：

```text
msa_chunk_size: 256 -> 2048
```

两轮80个样本的身份、rank、step和token composition完全相同，归一化SHA256均为：

```text
0d39a281e5b288030c0a9833c7ba1cd1f6b6013dd6cf5e4babe91c298f08e823
```

## 结果

| 指标 | chunk256 | chunk2048 单块 | 变化 |
|---|---:|---:|---:|
| seconds/step | 51.7310 | 50.4068 | -2.56% |
| tokens/s | 158.358 | 162.518 | +2.63% |
| PyTorch max allocated | 52.14 GiB | 52.14 GiB | +约6 MiB |
| PyTorch max reserved | 56.41 GiB | 57.27 GiB | +880 MiB |
| NVML peak | 59,715 MiB | 60,595 MiB | +880 MiB |

单块轮次完成10/10 step，状态 `passed`；224次MSA module调用均实际采样2048，
无CUDA OOM、loss NaN、non-finite gradient或optimizer skip。

诊断均值中，forward由24.465秒降至22.810秒，backward由25.739秒降至
25.294秒。总步速提升有限，是因为MSA行分块只影响整步中的一部分计算。

## 结论

在这组固定负载上，MSA单块训练可用，代价约为0.86 GiB额外峰值显存，换来约
2.6%的吞吐提升。它仍只覆盖前10步；正式长跑可能遇到后续更重的diffusion/loss
样本，因此将其设为默认前，建议至少补一次31步或100步稳定性验证。

## 复现

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload

BENCHMARK_ID=<new-unique-id> \
bash run_msa2048_crop1024_nuc256_db32_chunk2048_10step_fsdp2_benchmark.sh
```

结果：

```text
/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_offload_crop1024_nuc256_db32_chunk2048_10step/20260716_train2_fsdp2_crop1024_msa2048_nuc256_db32_chunk2048_10step_r1/summary.md
```
