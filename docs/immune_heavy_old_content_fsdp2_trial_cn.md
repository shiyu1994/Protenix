# immune-heavy 旧训练内容 + 当前 FSDP2 引擎试验

> 状态：最终配置已固化，并通过 dry-run、聚焦测试和修正后的8卡1-step smoke；正式2,000-step尚未启动  
> 建档日期：2026-07-16  
> 代码目录：`Protenix_train2_fsdp2_msa2048_offload`  
> 唯一公开入口：`run_immune_heavy_old_content_fsdp2_finetune.sh`

本文描述的是一个**混合试验**：恢复 2026-06-03 `sampling_immune_heavy`
中指定的训练内容和短程 schedule，同时保留当前已经 benchmark 过的
crop1024、FSDP2、CPU EMA 与 activation offload 执行引擎。它不是旧 DDP run 的逐项
复现，也不会覆盖当前从官方 PX v2 开始的通用训练入口。

## 1. 一句话定义

从上游训练第 1,000 次 update 后保存的 `999_ema_0.999.pt` 同时初始化 active/raw
模型和新的 EMA shadow，丢弃该文件中的 optimizer、scheduler 与 step，以
`0.25 weightedPDB + 0.50 sabdab-train + 0.25 distill-sabdab` 进行 2,000 次新的
全模型 optimizer update；LR 为 `3e-6`，全部 2,000 步处于 2,000-step warmup，
每 250 次 update 保存 raw 与 EMA。执行侧继续使用 crop1024、核酸上限256、
diffusion batch32、FSDP2 full-shard、rank-local CPU EMA 和 checkpoint-input CPU
offload。MSA继续使用当前16,384行featurizer pool；MSAStack中的rowwise weighted
averaging和transition路径补到2,048行，并以单个2,048-row chunk计算，但把每个实际
执行的recycle的真实采样数改为在 `1..min(实际深度, 2048)` 内均匀随机。

## 2. 相对当前通用 profile 的精确变更

两个 profile 使用同一底层 launcher。默认通用 profile 是 `general_msa2048`；本文入口
显式选择 `immune_heavy_old_content`。只有下表内容发生变化：

| 配置 | 当前通用 profile | 本试验 |
| --- | ---: | ---: |
| raw 初始化 | 官方 `protenix-v2.pt` | 上游 `999_ema_0.999.pt` |
| EMA 初始化 | 官方 `protenix-v2.pt` | 同一个 `999_ema_0.999.pt` |
| 数据集权重 | `0.50 / 0.30 / 0.20` | `0.25 / 0.50 / 0.25` |
| `data.msa.max_size.train` | 16,384 | 16,384（保留当前） |
| `data.msa.min_size.train` | 2,048 | 1 |
| `data.msa.sample_cutoff.train` | 2,048 | 2,048（保留当前） |
| MSA真实深度策略 | 池足够时固定2,048 | 每个recycle从 `1..min(实际深度,2048)` 均匀随机 |
| MSAStack rowwise runtime padding高度 | 2,048 | 2,048（保留当前） |
| `model.msa_module.msa_chunk_size` | 2,048（1块） | 2,048（1块，保留当前） |
| LR | `1e-5` | `3e-6` |
| warmup | 2,000 | 2,000 |
| 总 optimizer updates | 20,000 | 2,000 |
| checkpoint interval | 1,000 | 250 |

三个数据集的顺序固定为：

```text
0: weightedPDB_before210930_v20260101
1: sabdab-train
2: distill-sabdab
```

因此 `0.25,0.50,0.25` 的第二项明确对应 `sabdab-train`。后两个抗体相关数据集的
名义总权重为75%，但这不等于“只训练免疫重链”：没有新增 heavy-chain、CDR 或抗原
界面过滤器，也没有专门的抗体 loss；`immune_heavy` 是采样分布名称。

### 2.1 明确保留的当前设置

下列设置不回退到历史 DDP run，而是继续使用当前 profile：

| 类别 | 本试验保留值 |
| --- | --- |
| crop | `train_crop_size=1024`；正式随机 crop，并非每条严格1024 token |
| 核酸 | 三个数据集均为 `max_nuc_tokens=256` |
| 模型宽度 | `c_z=256`、MSA `c_m=128` |
| recycle / diffusion | `N_cycle=4`是训练上限；每步实际为1..4轮；diffusion batch/chunk=`32/4`、`N_step=20` |
| MSA pool / compute | featurizer最多16,384行；MSAStack rowwise路径padding 2,048、chunk 2,048（1块）；OuterProductMean使用真实 `k` |
| MSA配对处理 | paired MSA 当 unpaired 使用，并开启 shuffle；三个数据集都生效 |
| 数值 | BF16 autocast，FSDP parameter/reduce dtype为FP32，TF32关闭 |
| 分布式 | FSDP2一维 DeviceMesh、bottom-up fully-shard、forward后reshard |
| EMA | decay `0.999`，每个 rank 保存 rank-local CPU shard shadow |
| Pairformer checkpoint | 每2层一个region；每2个region offload一次checkpoint输入 |
| MSA checkpoint | 每1层一个region；每个region offload checkpoint输入 |
| transition | Pairformer和diffusion均为chunk128并对chunk做checkpoint |
| optimizer | Adam，beta=`0.9/0.95`，非AdamW，weight decay `1e-8` |
| 稳定性 | grad clip norm10、跨rank non-finite loss/gradient同步跳步 |
| eval | periodic与final eval均关闭；WandB关闭 |

这意味着本文试验与历史 `sampling_immune_heavy` 至少还有这些有意差异：历史 run 使用
crop768、核酸上限128、diffusion batch48、DDP、GPU完整EMA、Pairformer每1层
checkpoint、无当前CPU offload，并使用4,096行featurizer pool、16,384行runtime MSA
padding和标准species pairing；本文不恢复这些旧执行项。
所以结果应称为“old-content on current FSDP2 engine”，不能称为旧 run 的严格复现。

## 3. 起始 checkpoint 的完整来源链

```text
官方 PX v2 checkpoint
/mnt/project_abd/public_databases/protenix.20260101/checkpoint/protenix-v2.pt
    │
    │ params-only；raw和EMA均由官方权重初始化
    ▼
上游 run
protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805
    │  weightedPDB / sabdab-train = 0.7 / 0.3
    │  LR=1e-5，warmup=2000，crop=768，32 GPU
    │
    │ 第1000次optimizer update后保存，zero-based step=999
    ▼
999_ema_0.999.pt
    │
    │ 本试验再次params-only warm start；训练step重新从0开始
    ▼
immune_heavy_old_content + 当前FSDP2引擎
```

### 3.1 provenance 根

```text
path:
/mnt/project_abd/public_databases/protenix.20260101/checkpoint/protenix-v2.pt

size:
1,859,785,497 bytes

sha256:
8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599
```

上游 run 以该文件同时初始化 raw 和 EMA，`load_params_only=true`，然后使用
`weightedPDB_before210930_v20260101,sabdab-train` 和权重 `0.7,0.3` 训练。上游
总预算为20,000步，但本文有意选择它完成前1,000次 update 后的早期 EMA，而不是后续
checkpoint。文件名中的 `999` 是 zero-based step；它代表第1,000次 update 已完成。

### 3.2 本试验直接加载的文件

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

该文件的 `model` 字段保存的是上游 EMA 权重，同时还包含上游 optimizer、scheduler和
`step=999`。launcher 将同一路径同时传给：

```text
load_ema_checkpoint_path
load_checkpoint_path
```

并固定 `load_params_only=true`、`load_strict=true`。非 dry-run 启动会先计算完整文件的
SHA256；路径缺失或哈希不一致会在创建训练进程前失败。

## 4. FSDP2 下的加载语义

本试验是 warm start，不是从上游 step 1000 做 full resume。FSDP2 trainer 的有效顺序
如下：

```text
构建模型并bottom-up fully_shard
    ↓
构建新的Adam、AF3 scheduler和CPU EMA wrapper
    ↓
rank 0把999_ema文件加载到CPU并校验checkpoint header
    ↓
以canonical full model state加载EMA来源权重
    ↓
EMA wrapper把各rank本地DTensor shard注册为CPU shadow
    ↓
再次从同一文件加载active/raw模型权重
    ↓
从新step=0开始optimizer update
```

如 checkpoint key 带有旧 DDP 的 `module.` 前缀，FSDP2 loader 会在加载前去除；完整
state 通过 FSDP2 full-state loader 分发到当前 sharded model。因为只加载参数：

- 不加载上游 optimizer state；
- 不加载上游 scheduler state；
- 不继承 `step=999`；
- 新训练从 step 0 使用 `3e-6` / 2,000-step warmup；
- active/raw 和 EMA shadow 的初始数值相同，均来自旧 EMA 权重。

这一设置也避免了尝试把历史 DDP optimizer state直接恢复为FSDP2 optimizer shard。
checkpoint读取和 full-state分发在初始化阶段可能使用明显的CPU内存，但参数不会因此
长期复制到GPU；训练中的参数生命周期仍由FSDP2管理。

## 5. MSA 16,384 pool → 每轮随机至多2,048 → MSAStack rowwise补到2,048

最终确认的四层语义如下：

| 阶段 | 数值 | 含义 |
| --- | ---: | --- |
| data featurizer | 16,384 | 一个训练样本最多组装/保留16,384条MSA行 |
| model lower bound | 1 | 每个recycle独立随机真实采样的下界 |
| model cutoff | 2,048 | 随机采样数上限，同时是训练时MSAStack rowwise padding高度 |
| row chunk | 2,048 | MSAStack padding后的2,048行只形成1个计算块 |

对于featurizer实际交给模型的深度 `n`：

```text
1 <= n <= 16384
        ↓
u = min(n, 2048)
        ↓
k ~ UniformInteger(1, u)
        ↓
从全部n行中无放回随机选择k行
        ↓
OuterProductMean直接使用真实k行
        ↓
非末MSABlock内的MSAStack：
weighted averaging / transition把m补零到2048行
        ↓
msa_chunk_size=2048：rowwise路径为1个chunk
        ↓
MSAStack输出截回真实深度k
```

这里“均匀随机”是严格对最终样本数 `k` 定义的。原工具函数先执行
`k0 ~ UniformInteger(1,n)` 再做 `min(k0,2048)`；当 `n>2048` 时会把大量概率堆在
2,048，和 `UniformInteger(1,2048)` 不等价。本试验把2,048作为 `sample_indices` 的
上界，在生成 `k` 前先计算 `min(n,2048)`，同时仍从全部 `n` 行中随机选行。因此：

- 深池样本的1到2,048每个采样数等概率；
- 浅池样本的1到 `n` 每个采样数等概率；
- 同一次采样对 `msa/has_deletion/deletion_value` 使用完全相同的无重复行索引；
- 不会只从pool的前2,048行选取。

每个recycle到达MSA module时都会重新生成 `k` 和行索引。若某一步实际执行4个
recycle，就有4次相互独立的采样；不是四轮合计共享2,048行。模型配置的 `N_cycle=4`
是训练上限，具体step会按 `current_step` 确定性选择1到4轮。

“补到2,048”不适用于MSA module中的所有算子。每个MSABlock先执行的
OuterProductMean以真实采样深度 `k` 计算；只有非末MSABlock里的MSAStack rowwise
weighted averaging和transition为了固定训练图形状而补到2,048。第4个（末）MSABlock
没有MSAStack。因此随机到较小的 `k` 会减少OuterProductMean等真实行路径的工作量，
但上述rowwise路径仍保持一个完整2,048-row chunk。

`pair_msa_as_unpaired_only=true` 与 shuffle 保留当前行为。它们影响进入16,384行pool的
内容及顺序；补到2,048的零行只用于稳定训练计算图，不代表新增生物序列，也不扩大MSA
来源数据库。

## 6. 全模型训练范围

本文不是 Pairformer-only，也不是“只训练 Pairformer output之后”。运行时预期参数计数
为：

| 项目 | 数量 |
| --- | ---: |
| 总参数 | 464,442,431 |
| 可训练参数 | 464,441,841 |
| 固定参数 | 590 |

`train_confidence_only=false`，所有可学习模型权重进入 optimizer。固定的590个参数是模型
定义中的数值常量：

```text
confidence_head.lower_bins                                      39
confidence_head.upper_bins                                      39
diffusion_module.diffusion_conditioning.fourier_embedding.b    256
diffusion_module.diffusion_conditioning.fourier_embedding.w    256
```

这些量设计上不求梯度；不应为了字面上的“全模型”强行改为 trainable。这里“全模型训练”
准确指除上述固定常量外的全部464,441,841个可学习参数。

## 7. 显存和速度风险

当前通用 profile 的 matched crop1024 / MSA2048 / chunk2048 10-step benchmark曾观测：

```text
PyTorch max allocated: 52.14 GiB
PyTorch max reserved : 57.27 GiB
NVML peak            : 60,595 MiB（约59.17 GiB）
```

本试验不再引入16,384行runtime padding。它和当前通用profile使用相同的16,384行
featurizer pool；MSAStack rowwise路径使用2,048行padding高度和单个2,048-row chunk。
每轮真实采样数最多2,048，因此MSA module不会比当前固定2,048行策略看到更多真实行；
随机到更浅的 `k` 时，OuterProductMean等真实行路径还可能少做一部分工作，而MSAStack
rowwise padding路径保持固定。

所以此前担心的“8个MSA chunk和约4GiB完整m”不属于最终配置。主要显存风险重新来自
crop1024的token组成、原子数、diffusion batch32和全模型训练，而不是MSA高度扩大。
CPU offload仍然只搬运activation-checkpoint保存输入，不会offload参数、梯度或
optimizer state。既有59.17GiB结果是重要基线，但数据混合与随机batch不同，最终profile
仍需独立smoke，不能直接把旧峰值当作保证。

因此执行顺序固定建议为：

```text
dry-run解析配置
    ↓
1-step smoke：必须完成checkpoint load + forward + loss + backward + optimizer + EMA
    ↓
检查所有rank无OOM、NaN、non-finite、skip
    ↓
确认GPU/CPU峰值和首次编译行为
    ↓
再启动正式2000步
```

1-step smoke只证明所抽到的一个global batch能走完完整训练生命周期。训练step 0按当前
`N_cycle`选择逻辑只执行1个recycle，因此该smoke本身不能证明同一步中的多recycle重采；
它也不覆盖4-recycle、高核酸占比或正式数据最坏样本。正式训练仍需持续监控。

### 7.1 已废弃的16,384-height smoke（仅保留审计证据）

在用户修正最终MSA语义之前，曾对错误的
`pool=4096/sample_cutoff=16384/runtime_height=16384/chunk=2048×8` profile执行一次
8卡1-step smoke。该run成功，但它不是最终配置的验证结果，不能用于决定正式训练。
保留路径只是为了避免丢失实验 provenance：

```text
smoke id:
20260716_train2_immune_heavy_old_content_fsdp2_smoke1_r1

trainer run dir:
/mnt/project_abd/klp/proteinx_output/
20260716_train2_immune_heavy_old_content_fsdp2_smoke1_r1_20260716_172405

完整stdout与NVML采样:
/mnt/project_abd/klp/proteinx_benchmark/train2_immune_heavy_old_content_fsdp2/
20260716_train2_immune_heavy_old_content_fsdp2_smoke1_r1/
```

错误profile的最高NVML为58,139MiB、聚合loss为1.1582898，且完成了checkpoint strict
load、optimizer和EMA update。这里只说明历史错误配置也未OOM；最终结果必须记录在新的
独立run id下。

### 7.2 最终2,048-height smoke

最终配置已于2026-07-16在单机8张H100 80GB上完成1-step smoke，进程退出码为0：

```text
smoke id:
20260716_train2_immune_heavy_old_content_msa_random_cap2048_fsdp2_smoke1_r1

trainer run dir:
/mnt/project_abd/klp/proteinx_output/
20260716_train2_immune_heavy_old_content_msa_random_cap2048_fsdp2_smoke1_r1_20260716_175841

完整stdout与0.5秒NVML采样:
/mnt/project_abd/klp/proteinx_benchmark/train2_immune_heavy_old_content_fsdp2/
20260716_train2_immune_heavy_old_content_msa_random_cap2048_fsdp2_smoke1_r1/
```

启动日志解析出的最终配置为
`min/cutoff/pool/chunk=1/2048/16384/2048`、runtime `msa_max_size=2048`。
旧 `999_ema_0.999.pt` 以 `strict=True` 分别成功加载为active/raw模型和CPU EMA
shadow；FSDP2 mesh为8卡，模型计数为464,441,841个可训练参数和590个固定常量。

step 0按确定性cycle调度只执行1个recycle。该recycle在8个rank上的结构化记录如下；
`input MSA`是featurizer提供的真实pool深度，`sampled MSA`是进入本次MSA module的随机
深度：

| rank | token | 样本 | input MSA | sampled MSA | NVML峰值MiB |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 296 | weightedPDB `5mr2` | 93 | 14 | 10,577 |
| 1 | 1,024 | weightedPDB `6fml`（28 DNA token） | 6,309 | 722 | 56,073 |
| 2 | 1,024 | weightedPDB `7bvy` | 11,218 | 1,697 | 55,955 |
| 3 | 237 | weightedPDB `1bdr` | 5,659 | 578 | 9,095 |
| 4 | 231 | sabdab-train `7jwp_e_f_s` | 16,384 | 1,186 | 9,157 |
| 5 | 240 | sabdab-train `6bqb_h_l_p` | 16,384 | 1,703 | 9,475 |
| 6 | 451 | distill-sabdab | 16,384 | 914 | 15,575 |
| 7 | 565 | distill-sabdab | 16,384 | 1,914 | 20,497 |

debug日志进一步直接记录了深池调用使用
`randint_low=1, randint_high_exclusive=2049`；浅池93行则使用
`randint_low=1, randint_high_exclusive=94`。因此本次真实运行同时验证了：

- featurizer能够保留到16,384行，而不是回退为4,096行；
- 随机深度上限在抽样前应用，深池从闭区间1..2,048均匀抽取；
- 浅池从1..实际深度抽取；
- 两条样本确实达到精确1,024 token，其中一条包含核酸；
- 每个rank均有 `did_update=true`、`loss_nan=false`、`nonfinite_grad=false`、
  `optimizer_skipped=false`。

聚合 `train/loss.avg=1.1185643524`，全局grad norm为0.232917；最慢rank的
step wall time为198.77秒。run manifest最终记录
`status=completed, exit_reason=max_steps_reached, last_global_step=1`，且exception为空。
结构化显存最高值出现在rank 1：max allocated 51,237MiB、max reserved 54,120MiB；
0.5秒NVML观测到的全局最高峰值为56,073MiB。错误目录为空，日志中没有OOM、NaN、
non-finite gradient或optimizer skip。

这一步已经证明最终profile能够完成checkpoint load、forward、loss、backward、optimizer
update和CPU EMA update；它仍然只覆盖step 0的一轮recycle和一个global batch，不能替代
正式训练期间对4-recycle或更坏token/atom组成的持续显存监控。

## 8. AIStudio 执行命令

所有命令均在当前新代码库中执行：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/
Protenix_train2_fsdp2_msa2048_offload
```

AIStudio提供的 `WORLD_SIZE`、`RANK`、worker host/port会由 `finetune_submit.sh` 映射为
`torchrun` 的 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。不要手工在每个
worker上设置不同的科学参数；所有节点必须使用同一命令和同一个 `RUN_NAME`。

### 8.1 只展开最终配置：dry-run

dry-run不读取5.6 GB checkpoint、不初始化GPU、不启动训练，也不验证一次forward；它只
用于检查最终 `torchrun` 命令及 resolved 参数链。

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload

RUN_NAME=immune_heavy_old_content_fsdp2_dryrun \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
DRY_RUN=1 \
bash run_immune_heavy_old_content_fsdp2_finetune.sh \
  > immune_heavy_old_content_fsdp2_dryrun.txt 2>&1
```

输出开头应包含：

```text
content_profile=immune_heavy_old_content
max_steps=2000
weights=0.25,0.5,0.25
MSA min/cutoff/pool/chunk=1/2048/16384/2048
lr/warmup=0.000003/2000
checkpoint_interval=250
```

展开的命令还应同时出现旧 checkpoint 的 raw/EMA路径、crop1024、三个
`max_nuc_tokens=256`、diffusion batch32、FSDP2和CPU EMA参数。dry-run成功不代表显存
可容纳。

### 8.2 完整训练生命周期：1-step smoke

`PROTENIX_SMOKE_TEST=1` 是专用安全模式：强制 `max_steps=1`、
`checkpoint_interval=-1`、`skip_final_checkpoint=true`。它会真实校验checkpoint SHA，
并执行一次完整 optimizer update和一次EMA更新，但不会写出约5.6 GB的模型checkpoint。

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload

RUN_NAME=immune_heavy_old_content_fsdp2_smoke1 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
PROTENIX_SMOKE_TEST=1 \
bash run_immune_heavy_old_content_fsdp2_finetune.sh \
  > immune_heavy_old_content_fsdp2_smoke1.log 2>&1
```

需要确认日志至少满足：

- checkpoint SHA256匹配且 strict load成功；
- raw与EMA均从同一个 `999_ema_0.999.pt` 初始化；
- 参数计数为 `464441841 trainable / 590 fixed`；
- 所有rank完成forward、loss、backward、optimizer与EMA update；
- 没有 `CUDA out of memory`、NaN、non-finite gradient或optimizer skipped；
- `max_steps_reached` 正常退出，而不是数据或分布式异常退出。

首次运行可能包含CUDA extension/Triton编译，因此1-step耗时不能当作稳态吞吐。smoke
不保存模型checkpoint是有意行为；它不是可续训的短程实验。

### 8.3 正式2,000-step试验

确认1-step smoke通过并检查GPU/CPU峰值后，再启动正式 run：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload

RUN_NAME=immune_heavy_old_content_crop1024_nuc256_db32_fsdp2 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
MAX_STEPS=2000 \
PROTENIX_SMOKE_TEST=0 \
bash run_immune_heavy_old_content_fsdp2_finetune.sh \
  > immune_heavy_old_content_crop1024_nuc256_db32_fsdp2.log 2>&1
```

该 profile只接受 `MAX_STEPS=2000`；若设置为其他值会在启动前失败。也可以不设置
`MAX_STEPS`，默认仍严格为2,000。不要把 `PROTENIX_SMOKE_TEST=1` 遗留在正式命令中。

训练产物写入：

```text
${BASE_DIR}/${RUN_NAME}_YYYYMMDD_HHMMSS/
```

每250次 update同时保存raw与EMA，zero-based文件名预期为：

```text
249.pt       249_ema_0.999.pt
499.pt       499_ema_0.999.pt
749.pt       749_ema_0.999.pt
999.pt       999_ema_0.999.pt
1249.pt      1249_ema_0.999.pt
1499.pt      1499_ema_0.999.pt
1749.pt      1749_ema_0.999.pt
1999.pt      1999_ema_0.999.pt
```

第2,000次 update 的周期保存和final保存同时命中，trainer应只保留一组最终
`1999/1999_ema`。训练内eval全部关闭；最佳抗体checkpoint仍需通过独立推理与DockQ
评估选择，不能只按训练loss决定。

## 9. 正式启动后的最小监控清单

正式 run 至少检查以下内容：

1. 启动摘要中的 profile、checkpoint SHA、三数据集权重、MSA四个数值无漂移；
2. 每个rank都进入同一步，FSDP2 collective没有超时或rank失联；
3. GPU allocated/reserved/NVML峰值和host/pinned内存没有持续增长；
4. MSA debug抽样中 `input_depth<=16384`、
   `sampled_depth<=min(input_depth,2048)`；MSAStack rowwise padding高度和chunk均为
   2,048（1块），OuterProductMean使用真实 `sampled_depth`；
5. loss、gradient、optimizer skip计数无异常；
6. update 250后同时出现 `249.pt` 与 `249_ema_0.999.pt`，并核对checkpoint index；
7. 最终状态为 `max_steps_reached`，最后一组文件为 `1999/1999_ema`。

若显存不足，任何降低 crop、diffusion batch、MSA cutoff、模型宽度或关闭模块的处理都会
改变本文科学 profile，应作为新的对照实验命名并重新记录，不能静默修改后继续沿用本
试验名称。
