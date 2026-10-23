# crop1024 / nuc128 / db40 / paired-MSA 与原始 immune-heavy 的实现差异

更新时间：2026-07-20

## 1. 文档范围

本文比较以下两个训练实现。

### 原始 immune-heavy

代码库：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train
```

入口：

```bash
EXPERIMENT_NAME=sampling_immune_heavy \
TRAIN_SAMPLE_WEIGHTS=0.25,0.5,0.25 \
bash run_sab23h2_sampling_finetune.sh
```

已完成的实际 run：

```text
/mnt/project_abd/klp/proteinx_output/
  protenix_sampling_sampling_immune_heavy_from20260531_999ema_lr0.000003_
  20260603_004205_20260603_004232
```

该 run 已完成 2,000 step，共使用 4 节点、32 张 GPU，最终 checkpoint 为
`1999.pt` 和 `1999_ema_0.999.pt`。

### 当前 crop1024 / nuc128 / db40 / paired-MSA

代码库：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/
  Protenix_train2_fsdp2_crop1024_nuc128_db40
```

入口：

```text
run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh
```

当前入口已经通过 dry-run 和配置回归测试，但截至本文更新时间，尚未启动新的正式
2,000-step 训练。因此本文能够确定的是配置、数据路径和代码行为差异；对最终模型
效果的影响属于有依据的优先级判断，仍需 matched 训练与评估验证。

## 2. 一句话结论

当前实现已经恢复了原始 immune-heavy 的：

- 相同起始 checkpoint；
- 相同三数据集及 `0.25 / 0.50 / 0.25` 权重；
- 相同 `max_nuc_tokens=128`；
- 相同正常 species-paired MSA，不再把 paired MSA 当作 unpaired，也不再 shuffle；
- 相同全模型训练范围、loss、optimizer、学习率和 2,000-step schedule。

当前仍未与原始训练对齐的主要训练内容是：

1. 每个 recycle 的真实 MSA 行数：原始最多 16,384，当前最多 2,048；
2. crop 上限：原始 768，当前 1,024；
3. 每个 target 的 diffusion noisy sample 数：原始 48，当前 40。

此外，当前使用 FSDP2、CPU EMA 和 activation offload，而原始训练使用 DDP 和
GPU EMA。这些主要是分布式、显存和 checkpoint 实现差异，预期影响低于前三项
训练信号差异。

## 3. 最终有效配置总表

| 项目 | 原始 immune-heavy | 当前 crop1024/nuc128/db40/paired-MSA | 是否对齐 |
|---|---:|---:|---|
| 起始 raw 参数 | 上游 `999_ema_0.999.pt` | 同一个文件 | 是 |
| 起始 EMA 参数 | 同上 | 同上 | 是 |
| checkpoint 加载 | params-only | params-only | 是 |
| optimizer/scheduler/step | 从头开始 | 从头开始 | 是 |
| 训练数据集 | weightedPDB、SAbDab、distill-SAbDab | 相同 | 是 |
| 数据集权重 | `0.25 / 0.50 / 0.25` | 相同 | 是 |
| sampler | replacement、seed 42 | 相同有效序列 | 是 |
| world size | 32 ranks | 由AIStudio任务决定 | 32 ranks时对齐 |
| global target batch | 32 | `WORLD_SIZE × 1` | 32 ranks时对齐 |
| trainable scope | 全模型 | 全模型 | 是 |
| 可训练参数 | 464,441,841 | 464,441,841 | 是 |
| `c_z / c_m` | `256 / 128` | `256 / 128` | 是 |
| dtype | BF16 autocast，参数FP32 | 相同；FSDP参数/归约FP32 | 基本对齐 |
| crop 上限 | 768 | 1,024 | **否** |
| `max_nuc_tokens` | 128 | 128 | 是 |
| paired MSA | 正常 species pairing | 正常 species pairing | 是 |
| paired-hit shuffle | 无 | 无 | 是 |
| MSA featurizer pool | 实际最多 16,384 | 最多 16,384 | 是 |
| 每 recycle 真实 MSA 行数 | `1..min(actual,16384)` | `1..min(actual,2048)` | **否** |
| MSA 计算/pad高度 | 16,384 | 2,048 | **否** |
| MSA chunk | 2,048，最多8个计算块 | 2,048，单个计算块 | **否** |
| recycle 数 | 每step确定性采样1至4轮 | 相同 | 是 |
| diffusion batch | 48 | 40 | **否** |
| diffusion chunk | 4 | 4 | 是 |
| LR | `3e-6` | `3e-6` | 是 |
| warmup / max step | `2000 / 2000` | `2000 / 2000` | 是 |
| checkpoint interval | 250 | 250 | 是 |
| EMA decay | 0.999 | 0.999 | 是 |
| 分布式 | DDP | FSDP2 hybrid/full-shard | **实现不同** |
| EMA shadow | GPU全模型 | rank-local CPU | **实现不同** |
| Pairformer checkpoint | 每1层 | 每2层 | **实现不同** |
| Pairformer activation offload | 无 | 每2个region一次CPU offload | **实现不同** |
| MSA activation offload | 无 | 每个checkpoint region | **实现不同** |
| Pair Transition | 原始整块计算 | chunk 128并checkpoint | **实现不同** |

## 4. 已经对齐的部分

### 4.1 起始 checkpoint 完全相同

两者的 raw 模型和 EMA shadow 都从以下文件做 params-only 初始化：

```text
/mnt/project_abd/klp/proteinx_output/
  protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/
  checkpoints/999_ema_0.999.pt
```

SHA256：

```text
9e0df397d9deb18ea46921c678a62afc4a587241aacd01acd27887729e3a0c3e
```

该 checkpoint 的来源链是：

```text
官方 protenix-v2.pt
  -> weightedPDB/SAbDab = 0.7/0.3，训练约1,000步
  -> step999 EMA
  -> 作为 immune-heavy 和当前训练的共同起点
```

两边都不继承上游 optimizer、scheduler 和 step，所以不是 resume 上游训练状态，而是
用同一组模型参数重新建立 Adam、学习率调度器并从 step 0 开始。

### 4.2 数据身份和采样权重相同

两边都使用：

```text
weightedPDB_before210930_v20260101
sabdab-train
distill-sabdab
```

权重均为：

```text
0.25,0.5,0.25
```

三套 index 文件、结构目录和 MSA/template 数据路径保持一致。当前 sampler 虽修正了
`__len__` 的元数据计算，但实际 multinomial 抽样及 rank slice 逻辑没有改变。

原始 run 的完整 64,000 个 target 统计为：

```text
weightedPDB:     16,121
sabdab-train:    31,887
distill-sabdab:  15,992
```

此前完成的 crop1024 FSDP2 run 在相同 seed 下得到完全相同的三项计数；每50步记录的
1,280个 dataset/PDB identity 也与原始 run 全部相同。因此顶层数据混采不是此前效果
差异的来源。

原始 run 是32 ranks、每rank一个 target、gradient accumulation为1，所以 global
target batch为32，2,000步共处理64,000个target。当前入口仍是每rank一个target且
accumulation为1；只有AIStudio任务也使用32 ranks时，global target batch和总target
暴露量才与原始训练一致。如果改变节点/GPU数量，这会成为额外且重要的训练差异。

### 4.3 正常 paired MSA 已恢复

当前专用 wrapper 明确锁定：

```text
PROTENIX_PAIR_MSA_PROFILE=species_paired
```

最终 resolved config 为：

```text
data.msa.pair_msa_as_unpaired_only=false
data.msa.shuffle_pair_msa_when_unpaired_only=false
data.sabdab-train.msa.pair_msa_as_unpaired_only=false
data.distill-sabdab.msa.pair_msa_as_unpaired_only=false
```

因此在包含多个不同蛋白序列的复合物中，当前实现会和原始 immune-heavy 一样调用
`MSAPairingEngine.pair_chains_by_species(...)`。

这意味着此前 nuc256/db32 run 中存在的以下行为已经消失：

- 不再把 paired A3M 拼接到每条链自己的 unpaired A3M；
- 不再清空 paired MSA；
- 不再随机打乱 paired hits；
- 不再因为 paired-hit shuffle 提前消耗 NumPy RNG，从而间接改变模板候选排列。

### 4.4 核酸上限已经对齐

原始和当前都使用：

```text
max_nuc_tokens=128
```

该参数表示一个 crop 中最多保留128个核酸 token，不会将不足128的样本 padding 到
128，也不会把总 crop 长度改成128。

### 4.5 模型、loss、optimizer 和 schedule 对齐

两者均为完整 Protenix-v2：

- `c_z=256`；
- `c_m=128`；
- Pairformer 48层；
- 全模型 464,441,841 个可训练参数；
- Adam，`beta1=0.9`、`beta2=0.95`；
- 非 AdamW，coupled weight decay `1e-8`；
- gradient clipping 10；
- AF3 linear warmup；
- LR `3e-6`；
- warmup 2,000 step；
- 总训练 2,000 step；
- 每250 step保存 raw 和 EMA；
- EMA decay 0.999。

diffusion noise law、sigma 分布、loss 权重、`N_step=20` 和
`diffusion_chunk_size=4` 也保持相同。

### 4.6 关键科学逻辑源码核对

逐文件校验后，下列当前源码与原始 immune-heavy 代码完全相同：

```text
protenix/model/protenix.py
protenix/model/generator.py
protenix/model/loss.py
protenix/data/pipeline/data_pipeline.py
protenix/data/msa/msa_utils.py
protenix/utils/cropping.py
protenix/utils/permutation/chain_permutation/heuristic.py
protenix/data/template/ 下4个Python文件
```

因此 recycle 主流程、diffusion noise生成、loss公式、crop算法、chain permutation 和
template直接实现都没有改变。

当前 `protenix/model/modules/diffusion.py` 的差异是将 Transition chunk/checkpoint
配置传入 diffusion conditioning；没有改变 noise law 或 loss。

当前 dataloader 增加了 sampler 参数校验、显式读取 replacement/seed，并修正
`__len__` 元数据；在本配置的 `replacement=true`、有效 seed 42 下，实际 target
抽样序列不变。

## 5. 仍然存在的科学训练内容差异

### 5.1 每个 recycle 的真实 MSA 深度

这是当前与原始训练之间最重要的剩余差异。

#### 原始训练的实际行为

原始 resolved config 中虽然写有：

```text
data.msa.max_size.train=4096
```

但旧版 `get_msa_featurizer()` 没有把该值传给 `MSAFeaturizer`。实际代码调用了无参数
的 `FeatureAssemblyLine()`，所以 featurizer pool 使用默认值16,384。

旧模型随后执行：

```text
K ~ Uniform[1, actual_depth]
K = min(K, 16384)
```

即真实采样行数最多16,384，并将 MSA 计算高度 pad 到16,384。

#### 当前训练的实际行为

当前 featurizer pool 仍是16,384，但模型采样直接受2,048上限约束：

```text
K ~ Uniform[1, min(actual_depth, 2048)]
```

计算高度也只 pad 到2,048。

#### 对训练信号的意义

对于 MSA 深度足够大的样本：

```text
原始平均真实行数约为: (1 + 16384) / 2 = 8192.5
当前平均真实行数约为: (1 + 2048) / 2  = 1024.5
```

平均真实 MSA 行数约相差8倍。

每个 recycle 都会独立重新采样 K 和具体行；不是1至4轮 recycle 合计只使用2,048行。
两边都只有最后一轮 recycle 开启梯度，但前面的 recycle 状态会进入最后一轮，因此
所有轮次看到的 MSA 分布都可能影响最终更新。

需要区分：

- 真实采样行数从最多16,384降到2,048，是科学训练信号变化；
- pad高度和chunk数量变化主要是计算/显存变化。

### 5.2 crop 上限从768增加到1,024

两边的 crop method 权重和算法相同：

```text
Contiguous / Spatial / SpatialInterface = 0.2 / 0.4 / 0.4
```

`complete_ligand=true`、`drop_last=true`、`remove_metal=true` 的有效行为也相同。
差别只是 crop 上限：

```text
原始: train_crop_size=768
当前: train_crop_size=1024
```

crop size 是上限，不是将每个样本强制 padding 到该长度。

此前已完成的 crop1024 FSDP2 run 和原始 run 都包含64,000个相同数据分布的 target。
其暴露量为：

| 指标 | 原始 crop768 | 已完成的 crop1024 前序run | 变化 |
|---|---:|---:|---:|
| 总 token | 23,994,451 | 25,671,301 | `+6.99%` |
| 总 atom | 185,754,649 | 200,225,494 | `+7.79%` |
| weightedPDB 平均 token | 548.25 | 651.36 | `+18.81%` |
| weightedPDB 平均 atom | 4,229.33 | 5,120.02 | `+21.06%` |
| SAbDab 平均 token | 245.58 | 245.77 | `+0.08%` |
| distill-SAbDab 平均 token | 458.06 | 458.59 | `+0.12%` |

注意：表中的 crop1024 数据来自此前完成的
`nuc256/db32/unpaired-MSA` 前序run，不是当前尚未启动的
`nuc128/db40/paired-MSA` run。因为 target identity 和 crop 上限一致，该表可以
用于估计 crop1024 对长度/原子暴露量的影响，但不能用于判断当前配置的最终效果。

现有证据表明，crop1024 对两个抗体数据集的平均长度几乎没有影响，主要增加了
weightedPDB 的长上下文。因此它更可能通过通用复杂体系、接口上下文和正则化分布
间接影响抗体效果，而不是直接让大部分 SAbDab crop 变长。

### 5.3 diffusion batch 从48降低到40

这里的 diffusion batch 不是一次读取多少个 target，而是每个 target/crop 内生成的
独立 noisy diffusion training sample 数。

```text
原始: 48 samples / target，chunk4，共12个chunk
当前: 40 samples / target，chunk4，共10个chunk
```

noise分布和loss平均方式不变，因此期望目标基本相同；主要变化是每个 target 内
Monte Carlo 梯度估计的方差。

在独立同分布近似下，当前相对原始的标准误差比例约为：

```text
sqrt(48 / 40) = 1.095
```

即约增加9.5%。这可能在只有2,000步的短程微调中产生可见轨迹差异，但明显小于此前
`48 -> 32` 时约22.5%的标准误差增幅。

## 6. 分布式、显存和 checkpoint 实现差异

### 6.1 DDP 与 FSDP2

原始训练使用32卡 DDP；当前使用 FSDP2：

```text
distributed_strategy=fsdp2
optimizer_sharding=none
fsdp.mesh=auto
fsdp.param_dtype=fp32
fsdp.reduce_dtype=fp32
fsdp.reshard_after_forward=true
```

在当前多机配置中，`mesh=auto` 是节点内 shard、节点间 replicate 的 hybrid mesh，
不是简单地将整个模型跨32卡做单一全局 shard。

两者目标数学上相同，但 all-reduce 与 reduce-scatter/all-gather 的运算顺序不同，
因此不会 bitwise 一致。BF16 forward、FP32 gradient reduction 和2,000次 optimizer
update 会放大很小的舍入差异，但一般不应比 MSA 行数或 crop 内容变化更重要。

### 6.2 activation checkpoint 与 CPU offload

当前额外启用：

```text
Pairformer blocks_per_ckpt=2
Pairformer checkpoint CPU offload every 2 regions
MSA checkpoint CPU offload every 1 region
Pair Transition chunk size=128
Pair Transition chunk checkpoint=true
```

Pairformer/MSA checkpoint 和 `save_on_cpu` 的目标是不保存或不驻留部分反向所需
activation，反向时再搬回并重算。它们不移动模型参数、梯度或 optimizer state，
理论上不改变训练目标。

Transition chunk 128 会改变矩阵乘法分块方式，因此可能产生小的浮点舍入差异；
该设置同时用于 Pairformer 和 diffusion conditioning，是比单纯 activation offload
更值得保留的低优先级数值因素。

当前 LayerNorm CUDA extension 还调整了编译/缓存目录和ABI失败后的重建处理，以避免
多rank同时写共享源码目录。LayerNorm计算公式和CUDA kernel本身未改变，这属于运行
稳定性修改，不是预期的科学训练差异。

### 6.3 EMA

原始训练：

```text
GPU full-model EMA
```

当前训练：

```text
rank-local DTensor CPU EMA
```

EMA 不参与 raw 参数的下一步 forward/backward，只在 optimizer step 后更新 shadow。
因此 EMA 实现不会改变 raw 模型的训练轨迹；它只可能造成最终 EMA checkpoint 的
微小浮点差异。

需要注意 checkpoint 包装格式：

- 原始 EMA checkpoint 是旧格式，模型 key 常带 `module.`，并包含 optimizer/scheduler；
- 当前 EMA checkpoint 使用新格式，模型状态完整，但不重复保存无用 optimizer state。

仓库自带 inference loader 同时支持两种格式。若使用自定义评估脚本并以
`strict=false` 静默加载，则必须确认没有因为 key prefix 或新格式而漏加载参数。

### 6.4 NaN/Inf 处理

当前 FSDP2 runner 会在所有 rank 之间同步 non-finite loss/gradient 状态，并让整步
一致跳过 optimizer、scheduler 和 EMA update；旧 runner 的处理更偏 rank-local。

健康训练步上两者没有差别。只有实际出现 NaN/Inf 时，optimizer step 和 scheduler
轨迹才可能不同。原始完整 run 和此前完整 crop1024/db32 run 的40个聚合日志点中，
`loss_nan=true` 与 non-finite grad 均为0，因此目前没有证据表明该实现差异参与了
此前的效果差异。

## 7. 效果差异的可能来源优先级

下列优先级针对“当前 crop1024/nuc128/db40/paired-MSA 与原始 immune-heavy 最终
效果不同”这一问题。

### P0：先排除运行规模或评估口径不一致

这些不是本文锁定配置本身的差异，但最容易制造额外差异，且应在任何归因之前检查。

必须保证：

- 两边都评估同一步数的 EMA，例如都使用 `1999_ema_0.999.pt`；
- 不是旧 EMA 对比当前 raw；
- 当前正式训练也使用32 ranks；否则 global target batch 和2,000步总样本数会变化；
- 使用同一个 inference 代码版本；
- 使用相同 MSA、template、seed、diffusion `N_step`、sample数和 ranking 逻辑；
- 加载日志确认全部4,174个模型 tensor 都成功加载；
- 不因旧 `module.` prefix 或新 checkpoint format 静默漏权重。

### 优先级1：每 recycle 的真实 MSA 深度

判断：**最高优先级，最可能的剩余科学训练来源。**

原因：

- 对所有有深 MSA 的蛋白样本都可能生效，而不仅是少量核酸样本；
- 深池样本平均真实行数约从8,192降至1,024，相差约8倍；
- 直接进入 MSA module 并更新 pair representation；
- 每轮 recycle 都重新采样，因此影响所有 recycle 状态；
- 原始配置表面的“4,096”其实没有生效，容易低估真实差距。

针对抗体-抗原效果，正常 species pairing 已恢复，但 paired row 和 unpaired row
进入模型后的总深度仍被限制到2,048，所以 paired 语义对齐并不等于 MSA 训练信号
已经完全对齐。

### 优先级2：crop 768变为1,024

判断：**中高优先级。**

原因：

- 已有完整日志显示总 token/atom 暴露量增加约7%至8%；
- weightedPDB 平均 token 增加约18.8%，atom 增加约21.1%；
- 会改变长链、多链和大界面体系的上下文与损失贡献；
- 但 SAbDab 和 distill-SAbDab 平均 token 几乎不变，所以对抗体样本的直接影响
  小于对通用数据分布的影响。

### 优先级3：diffusion batch 48变为40

判断：**中等优先级。**

原因：

- 不改变期望loss，但减少每个 target 的噪声样本；
- Monte Carlo 标准误差理论上约增加9.5%；
- 2,000步短程微调中可能形成不同参数轨迹；
- 差距明显小于此前 db32 与原始 db48 的差距。

### 优先级4：Transition chunk 与 FSDP2 数值顺序

判断：**低至中等优先级。**

原因：

- FSDP2和DDP归约顺序不同，不可能 bitwise 一致；
- Transition chunk改变 GEMM 分块；
- 小的舍入差异可随训练累积；
- 但数学目标、FP32参数/归约、optimizer和loss都相同。

只有在 MSA、crop 和 diffusion batch 都 matched 后，这一项才适合成为主要对照变量。

### 优先级5：CPU EMA 与 checkpoint 格式

判断：**训练轨迹低优先级，评估加载问题则是P0。**

原因：

- EMA不反哺raw模型训练；
- CPU/GPU EMA可能有微小舍入差异；
- 官方loader支持两种格式；
- 但自定义loader若静默漏权重，会造成非常大的表面效果差异。

## 8. 当前不应再归因的项目

以下项目在当前代码中已经与原始 immune-heavy 对齐，不应作为首要训练差异来源：

- 起始 checkpoint；
- raw/EMA 初始参数；
- optimizer是否继承：两边都不继承；
- 三数据集身份和路径；
- `0.25 / 0.50 / 0.25` 数据权重；
- 全模型训练范围；
- 模型参数规模；
- loss定义和权重；
- Adam配置；
- LR、warmup、总步数；
- `max_nuc_tokens=128`；
- 正常 species-paired MSA；
- paired-hit shuffle：两边都关闭；
- EMA decay 0.999。

## 9. 已有训练日志能说明什么

此前完成的原始 run 与 `crop1024/nuc256/db32/unpaired-MSA` run 的平均训练 loss 为：

```text
原始: 1.047979
前序crop1024 FSDP2: 1.050297
```

最后50步：

```text
原始: 1.040374
前序crop1024 FSDP2: 1.042640
```

这说明前序 FSDP2 训练没有明显爆炸或整体失效。其最终 EMA 相对共同起点的全局参数
更新向量与原始 run 的余弦相似度约为0.974，diffusion部分约0.983；差异更集中在
MSA/Pairformer。

但前序 run 同时使用 `nuc256/db32/unpaired-MSA`，不能把这些数字直接当作当前
`nuc128/db40/paired-MSA` 的预期结果。它们只能说明：

- FSDP2/CPU EMA checkpoint 没有发现缺失模型参数；
- 顶层 sampler 和训练 lifecycle 正常；
- MSA/Pairformer 的训练内容变化确实比 diffusion 部分更值得优先排查。

## 10. 若要做到“只换DDP为FSDP2”的严格 matched 配置

若实验目标是仅验证 DDP 与 FSDP2 的差别，应将当前配置进一步改为：

```text
train_crop_size=768
max_nuc_tokens=128
diffusion_batch_size=48
pair_msa_as_unpaired_only=false
shuffle_pair_msa_when_unpaired_only=false
MSA featurizer pool=16384
MSA min_size.train=1
MSA sample_cutoff.train=16384
model.msa_module.msa_max_size=16384
MSA chunk_size=2048
```

同时保留：

```text
FSDP2
CPU EMA
Pairformer/MSA activation checkpoint
CPU activation offload
```

为了进一步隔离纯FSDP数值差异，Transition chunk也应暂时关闭；但这会增加显存，
需要重新 benchmark。

推荐消融顺序：

1. 当前配置先完成 `paired-MSA + db40 + nuc128` 训练；
2. 只将 MSA cutoff/计算高度恢复到16,384；
3. 再将 crop恢复到768；
4. 再将 diffusion batch恢复到48；
5. 最终比较 matched DDP 与 FSDP2。

如果显存不允许一步恢复 MSA 16,384，可以依次测试4,096、8,192、16,384，但这些
属于新的深度消融，不应称为原始 immune-heavy 的严格复现。

## 11. 当前专用入口的 dry-run 验证

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/
cd Protenix_train2_fsdp2_crop1024_nuc128_db40

RUN_NAME=immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_fsdp2_dryrun \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
DRY_RUN=1 \
NPROC_PER_NODE=1 \
bash run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh
```

关键输出必须包含：

```text
content_profile=immune_heavy_old_content
crop=1024 max_nuc_tokens=128 diffusion_batch/chunk=40/4
MSA min/cutoff/pool/chunk=1/2048/16384/2048
pair_msa_profile=species_paired
pair_as_unpaired=false
shuffle_paired_hits=false
```

展开后的 train CLI 还必须包含：

```text
--data.msa.pair_msa_as_unpaired_only false
--data.msa.shuffle_pair_msa_when_unpaired_only false
--data.sabdab-train.msa.pair_msa_as_unpaired_only false
--data.distill-sabdab.msa.pair_msa_as_unpaired_only false
```

## 12. 证据位置

当前代码：

```text
run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh
run_msa2048_crop1024_nuc256_db32_chunk2048_fsdp2_finetune.sh
protenix/data/msa/msa_featurizer.py
protenix/data/pipeline/dataset.py
protenix/model/utils.py
protenix/model/modules/pairformer.py
runner/train.py
runner/ema.py
tests/test_immune_heavy_diffusion_batch_profiles.py
```

原始代码：

```text
../Protenix_train/run_sab23h2_sampling_finetune.sh
../Protenix_train/finetune_demo.sh
../Protenix_train/protenix/data/msa/msa_featurizer.py
../Protenix_train/protenix/data/pipeline/dataset.py
../Protenix_train/protenix/model/utils.py
../Protenix_train/protenix/model/modules/pairformer.py
../Protenix_train/runner/train.py
../Protenix_train/runner/ema.py
```

原始实际运行产物：

```text
/mnt/project_abd/klp/proteinx_output/
  protenix_sampling_sampling_immune_heavy_from20260531_999ema_lr0.000003_
  20260603_004205_20260603_004232/
    config.yaml
    logs/train_metrics.jsonl
    logs/data_samples_rank*.jsonl
    checkpoints/1999.pt
    checkpoints/1999_ema_0.999.pt
```

前序 crop1024 FSDP2 运行产物，仅作为定量旁证：

```text
/mnt/project_abd/klp/proteinx_output/
  immune_heavy_old_content_crop1024_nuc256_db32_fsdp2_20260716_111332/
    config.yaml
    logs/train_metrics.jsonl
    logs/data_samples_rank*.jsonl
    checkpoints/1999.pt
    checkpoints/1999_ema_0.999.pt
```
