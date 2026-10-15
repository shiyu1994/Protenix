# crop384 / DB48 数据集与 loss 决策记录

## 1. 历史 baseline：weightedPDB-only

本节保留 2026-07-24 weightedPDB-only 阶段的锁定记录。该历史 scratch 与
strict-resume 入口使用：

- 数据集：
  `weightedPDB_before210930_v20260101`；
- 外层采样 mass：`1.0`；
- official initial-training loss：
  `alpha_diffusion=4`、`smooth_lddt=1`、`alpha_distogram=0.03`、
  `alpha_confidence=1e-4`、`alpha_pae=0`、`alpha_bond=0`。

对应的不可变命名 profile 是 `weighted_pdb_only_v1`，checkpoint 合同为
`crop384_db48_ddp_scratch_50k_ws128_acc4_data_weightedpdb_only_v1_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_q20_u30_p50_v8`。
该历史入口不启用 `sabdab-train`、`distill-sabdab` 或其他蒸馏集；这些数据以后只能
通过新的命名 profile 和新的训练合同加入，不能沿用本阶段 checkpoint 合同。

2026-07-25 的当前正式入口是
`weighted_pdb_afdb_mgnify_short_long_disordered_v3`。此前的
`weighted_pdb_afdb_mgnify_disordered_v2` 是历史中间阶段；v1、v2、v3 三者都是
不同且互不兼容的命名 contract。上面的“当前”只指历史 baseline 当时的状态，已有
profile、launcher 和 checkpoint identity 均不原地修改。

## 2. 数据集与采样合同

### 2.1 历史 weightedPDB-only 数据规模和过滤

| 数据集 | index 原始行 | 过滤后行 | 唯一 PDB | 历史 mass |
| --- | ---: | ---: | ---: | ---: |
| weighted PDB | 1,417,835 | 1,040,707 | 164,831 | 1.00 |

该集合使用 2021-09-30 之前、分辨率上限 4.5 Å 的版本化 index；排除 ion row、
移除纯核酸 chain 与 nuc–nuc interface，并允许数据构造失败后在同一集合内随机
换 row。

### 2.2 历史 weightedPDB-only 两级权重

第一层在每个数据集内部重算 row weight：

- interface 的 `beta=1`，chain 的 `beta=0.5`；
- protein/nucleic acid/ligand 的 `alpha=3/3/1`；
- 再按 cluster size 和同一 PDB entity-pair 的重复数降权。

第二层先把每个集合内部归一化，再乘外层 dataset mass。本阶段只有 weightedPDB，
所以 dataset mass 为 1.0；这不等于 index 内所有 row 均匀采样，第一层 row
weight 仍然生效。采样使用 replacement，sampler seed 继承全局 seed 42。

### 2.3 历史 weightedPDB-only 的 5 万次 optimizer update 暴露量

该历史阶段锁定 16×8=128 卡 DDP，每 rank 每个 micro-step 处理 1 个 target，梯度累计
4 次，因此每个正常 optimizer update 消耗 512 个 target。5 万个累计窗口共有
200,000 个 micro-step/rank、约 25,600,000 次全局 target draw。全部 draw 都来自
weightedPDB：

| 数据集 | 128 卡总 draw | 相对过滤后行数的平均 draw/row |
| --- | ---: | ---: |
| weighted PDB | 25,600,000 | 24.60 |

实际 row 权重不均匀，因此高权重 row 的重复次数会更高；24.60 只是总 draw 除以
过滤后行数的算术平均，不是每行保证出现的次数。

当前 v3 仍先在每个 dataset 内归一化，再按以下 outer mass 选择 dataset：

| 数据集 | dataset key | raw mass | 归一化概率 | 50k update 期望 draw |
|---|---|---:|---:|---:|
| PDB | `weightedPDB_before210930_v20260101` | 0.500 | 49.0196% | 12,549,020 |
| Short protein distill | `afdb_mgnify_short_distillation_v20260725` | 0.005 | 0.4902% | 125,490 |
| Long protein distill | `afdb_mgnify_long_distillation_v20260725` | 0.495 | 48.5294% | 12,423,529 |
| OpenFold3 disordered | `openfold3_disordered_v20260725` | 0.020 | 1.9608% | 501,961 |

Short + Long raw mass 合计仍为 0.50。它们沿用当前 AFQ/MGN sequence
`clusters.csv.gz`，以最终 `representative_id` canonical length 对整簇分区：

| 分区 | 判定 | cluster | row |
|---|---|---:|---:|
| Short | representative length `<=200` | 14,327,216 | 15,425,587 |
| Long | representative length `>200` | 16,163,340 | 18,073,064 |
| 合计 | 精确并集 | 30,490,556 | 33,498,651 |

Foldseek 产物只提供已审计的 canonical-length path set，不是新的 cluster authority。
两边共享 pair ID
`9c87d52dd446f622be689b7a09611f302d3791ac2448ecd86f8aee2488fb8734`，
pair receipt SHA-256 为
`1b47dd0a7f15840b4a3f27ad67e1d821c954b263ee58c4c110c6c45a880a5476`。
Short/Long manifest SHA-256 分别为
`cab1953754d9c45302d8ab359006b21352d332628d767584471778433431ff1c` /
`b935c0d623d1fc2982a1449fab7ffed50b259653e11fb4c41c7bda131f535d9a`。

200,000 次确定性 sampler smoke 按表中顺序得到
`97,914 / 948 / 97,136 / 4,002`，与归一化概率一致。replacement sampling 不保证每个
epoch 或任意有限窗口恰好等于理论比例。

### 2.4 crop、MSA 与 recycle

- crop 384 是上限，不是 exact 384；
- crop method 为 contiguous/spatial/spatial-interface=`0.2/0.4/0.4`；
- `max_nuc_tokens=128`，完整 ligand、drop-last 和 remove-metal 均开启；
- 每个训练 sampled target occurrence 使用一个互斥三分类 gate：10% 为
  query-only、30% 为 all-protein-row unpaired-shuffle、60% 为正常
  species-paired。10% 是 occurrence 级**外层 mode gate**，不是每条链各自独立
  抽 10%；同一 occurrence 的 mode 对全部 recycle 一致；
- 当前 `weighted_pdb_afdb_mgnify_short_long_disordered_v3` 在 10% query-only mode 内先完成
  crop，再按 crop 中实际出现的 physical protein `asym_id` 计算候选数 `N`。homomer
  的不同 physical copy 具有不同 `asym_id`，因此是相互独立的候选；RNA、DNA 和
  ligand 不进入候选集合；
- `N=1` 时保留 legacy global query-only：跳过该 occurrence 的 protein/RNA homolog
  I/O，把 crop 后全部 target sequence 合成唯一一条全局 query row，并按该行重算
  row feature、`profile`、`deletion_mean` 和 alignment counts。当前 contract 对罕见
  `N=0` crop 也声明使用同一 legacy global fallback；
- `N>=2` 时先均匀采样
  `k ~ UniformInteger[1, min(2, N-1)]`，再从按 `asym_id` 排序的候选中无放回选择
  `k` 条 protein chain。这样每个 multi-protein crop 至多两条 chain 被置为
  query-only，并且至少一条 protein chain 不会被该增广强制删除 homolog；
- 被选 protein chain 在 MSA 读取前跳过 homolog I/O，仅从 authoritative canonical
  sequence 构造 query。最终 assembled MSA 再强制这些 chain 的 row 0 保持 query、
  row `>0` 为 gap，deletion-derived row feature 为 0，`profile` 为 query one-hot；
  未选 protein chain 以及 multi-protein crop 中的 RNA 继续走正常 Protenix MSA
  流程，DNA/ligand feature 不变；
- 如果 query-only 选择后只剩一条 protein chain 保留 MSA，该链的 paired-source hits
  会并入 unpaired pool；否则 species pairing 会因没有第二条可配 protein chain 而把
  这些 hits 静默丢掉。这里的“保留正常链”只保证代码不强制 query-only，不能凭空保证
  源 alignment 一定含 homolog；
- 30% 分支使用 `pair_msa_as_unpaired_only=true` 与
  `shuffle_pair_msa_when_unpaired_only=true`，关闭 species pairing，把全部 protein
  paired/unpaired homolog rows 合入 unpaired pool，并在保留 query row 的前提下 shuffle；
- 该分支先按 crop-visible physical protein `asym_id` 计数 `N`：`N=0` 为 no-op，
  `N=1` 总是请求 segment；`N>=2` 用独立 occurrence seed 以 50%/50% 请求 segment 或
  whole-physical-chain。segment 按 canonical `res_id` gap 24、最短 9 residue 建组；
  whole variant 每个 physical protein asym 一个 group，homomer copy 不合并，非蛋白不参与；
- 50/50 是 requested variant 比例；crop 无断点时 segment grouping 会退化成每条 physical chain
  一个 group。全 crop 最终少于两个 group，或 homolog 不足时可以不实际应用；
  requested、eligible、applied 分别记录；
- 60% normal 分支显式使用 `false/false`，沿用正常 species pairing；
- 以下 assembled-pool 与 K 抽样语义描述两个 homolog 分支：
- protein MSA 原始文件按 sequence 从官方 `mmcif_msa_template` 定位；raw A3M
  会先解析，再经 dedup、species pairing、头部截断和多链合并形成最终 pool，
  **16,384 是最终 assembled pool 上限，不是 raw A3M 读取上限，也不会 padding
  到 16,384**；
- crop 中出现两个以上不同 protein sequence 时才做 species pairing；paired
  配额最多 8,192 行，剩余容量由 unpaired 行填充。pool 构造阶段保留的是有序
  头部，不是从全部 raw hit 均匀抽 16,384 行；
- 设最终 pool 实际深度为 `D`，每个实际 recycle 独立采
  `K ~ UniformInteger[1, min(D, 2048)]`，再执行
  `randperm(D)[:K]`，即从完整 assembled pool 中无放回抽 K 行；
- sampled MSA one-hot、embedding 和 OuterProductMean 的真实高度是 K。4 个
  MSA block 中前 3 个 block 的 MSAStack 仅在 pair-weighted averaging 与
  Transition 内部把 hidden `m` 临时补零到 2,048，计算后再切回 K；末 block
  没有这类 MSAStack padding；
- `msa_chunk_size=2048` 只让上述临时 2,048 高度运算形成一个 chunk，不改变
  pool、K、抽样内容或 OPM 高度；
- `N_cycle=4` 是上限，每步实际执行 1–4 轮，只有最后一轮反传；
- 同一 optimizer update 的四个 accumulation micro-batch 使用相同 recycle
  轮数，但 torch RNG 继续前进，因此 MSA 深度和行子集通常不同；
- full-MSA 的随机 MSA row 子采样不会强制保留 row 0/query。原始 A3M 的 row 0 身份本身
  依赖数据库约定，代码只验证对齐长度；因此 `K=1` 也不等价于“保证 query-only”。
- `profile` 与 `deletion_mean` 在 recycle 抽 K 行之前，已经由每链完整 unpaired
  MSA 计算，并由 InputFeatureEmbedder 使用。因此即便某轮随机得到 `K=1`，模型
  仍看到了全 unpaired MSA 的统计摘要，不等价于真正的 single-sequence/no-MSA
  训练。

例如 `D=73` 时，K 在 1–73 间离散均匀，期望为 37；`D=4096` 或 16,384
时，K 都在 1–2,048 间离散均匀，期望为 1,024.5，但行分别从完整 4,096 或
16,384 行 pool 中抽取，而不是只看前 2,048 行。

### 2.5 Template 与可恢复的 occurrence gate

Template 使用决定与 MSA 决定相互独立：

- 20% occurrence 请求真实 template；若过滤后有候选，实际采 1–4 个；
- 80% occurrence 强制使用官方固定 4-slot dummy-template；
- 请求真实 template 但无有效候选时，也只能得到 dummy，因此实际有效 template
  比例不保证达到 20%；
- 旧 `template_dropout_rate` 在正式入口中锁为 0，避免两层随机逻辑叠加。

MSA categorical gate 与 template Bernoulli gate 都由 sampler 的
`(seed, epoch, global_position)` 和独立 stream name 经 SHA256 计算，不消耗
worker RNG。名义联合比例为 query-only/template 2%、query-only/no-template
8%、unpaired-shuffle/template 6%、unpaired-shuffle/no-template 24%、
species-paired/template 12%、species-paired/no-template 48%。这些概率、分类
顺序、算法版本、true/true 语义和旧 dropout rate 都进入 canonical data
contract；strict resume 改任一值都会在加载权重前失败。

当前 multi-protein query-only 的 chain 选择使用独立
`partial_query_only_chain_selection` stream。它不消费 Python、NumPy 或 torch
全局 RNG；但候选集合由实际 crop 决定，因此确定性范围准确表述为
`occurrence_seeded_given_crop`：给定同一个 occurrence 和同一组 crop-visible
protein `asym_id`，`k` 与被选 chain 可复现；这不额外承诺由 worker RNG 产生的
crop 本身在任意 worker topology 下逐位相同。

运行时 `basic` telemetry 保留外层 `msa_mode=query_only` /
`query_only_msa=true`，并额外记录
`N_msa_crop_protein_chain`、`N_msa_query_only_protein_chain`、
`N_msa_retained_protein_chain`、`msa_partial_query_only_applied` 和
`msa_global_query_only_applied`。因此 10% outer-gate 命中率、单蛋白 global
实现率与多蛋白 partial 实现率可以分别审计；partial occurrence 的 `N_msa`
通常大于 1，不应再用 `N_msa==1` 反推 query-only gate。

模式 gate 可逐 occurrence 严格恢复。历史 profile 的旧 `true/true` 内
paired-hit shuffle 仍使用 DataLoader worker NumPy RNG，因此多 worker 中断前后的
具体 hit 顺序不承诺逐位相同。当前 v3 的 30% 分支同时使用 occurrence-seeded row shuffle 和独立
`multichain_msa_shuffle_variant` stream；给定同一 crop，segment/whole 请求和排列可复现。
四源统一生效。运行时额外记录 `N_msa_shuffle_group`、
`msa_shuffle_requested_variant_code`、`msa_shuffle_variant_eligible`、
`msa_shuffle_variant_applied` 和 `msa_shuffle_whole_physical_chain_applied`，避免把请求
50/50 错当成实际 split 50/50。详细算法与非标准残基映射约束见
`docs/openfold3_disordered_distillation_cn.md` 第 8.3 节。该 gate 不接入 inference，推理继续
使用独立现有路径。

### 2.6 多机 epoch 边界

旧 `epoch_size=10000` 在 world size 128 下会让 rank 0–15 每轮得到 79 条，
rank 16–127 每轮只有 78 条，导致 sampler epoch/seed 边界错位。当前入口已锁为
`epoch_size=10240`：

- `10240 / 128 = 80` 个 micro-step/rank/epoch；
- 累计 4 次后是 20 个 optimizer update/epoch；
- 50,000 个累计窗口恰好是 2,500 个 sampler epoch；
- 10,240 是 sampler 每轮的总抽样次数，replacement=true 时各数据集只按 mass
  给出期望数量，不保证每轮精确比例。

`epoch_size` 会进入 strict-resume sampler signature；正式 scratch 开始后不能把
已有 10,240 checkpoint 严格恢复到其他 epoch size。

### 2.7 crop 与最小有效监督 token

当前四源 profile `weighted_pdb_afdb_mgnify_short_long_disordered_v3` 的实验 PDB、
Short、Long 和 OpenFold3 disordered 都通过 profile override 固定
`min_diffusion_supervised_tokens=4`。历史 `weighted_pdb_only_v1`、
`weighted_pdb_afdb_mgnify_distill_v1` 和
`weighted_pdb_afdb_mgnify_disordered_v2` 没有被原地改写。这里的“有效 token”不是只看中心原子：

- 普通 PDB、Short/Long AFDB/MGnify：token 内至少一个 `is_resolved` atom；
- disordered protein-only diffusion：token 内至少一个
  `is_resolved & is_protein` atom。

结构先生成最终 crop indices，再执行该条件的修复/校验，之后才读取 crop 涉及链的 MSA。
Short/Long AFDB/MGnify 与 disordered 都会在 crop 层补入最近的 eligible protein token 直到 4 个；
普通 PDB 的不合格极端 crop 由既有 fail-and-resample 处理；所有数据源都会在物理 crop 后
重新计数并硬校验。运行时
还会根据最终 feature/label 的 `coordinate_mask` 与 `atom_to_token_idx` 计数；数量记录为
`basic_info.N_diffusion_supervised_token`。模型内 mini-rollout 的 chain/atom permutation 可能
改写 label，所以阈值也作为 per-sample feature 传入模型：低于 4 的 permutation 候选会被拒绝并
保留 permutation 前的合法 label，permutation 完成后再做一次硬断言。这一层才是实际 diffusion
loss 之前的最后防线。

MSA lookup、query-row 校验、full-row dedup 和 species pairing 仍必须使用完整 canonical
sequence；最终才按 crop residue 映射 gather MSA 列。不能把 spatial crop 的非连续子序列直接
当成新的 A3M query。

该规则保证 diffusion MSE 至少有 4 个带坐标 label 的 token，但不额外承诺
smooth-LDDT 有效距离 pair、distogram representative 或 bond edge 数量。最小值与 token 定义已
进入 canonical data contract，profile tag 含 `resolvedproteincrop_minloss4_v2`。当前 profile
使用 schema 7，MSA tag 为
`allmsashuffle_multichain_seg50_whole50_gap24_min9_v2`；crop-aware query-only 使
MSA mode policy 为
`categorical_crop_aware_partial_query_only_all_protein_multichain_segment_or_whole_shuffle_species_paired_v4`，
并使 profile contract tag 包含 `partialquery_max2_keep1_v1`。schema 7 绑定
partial-query policy、post-crop physical-asym 候选轴、uniform `k`、最大 2 条、
至少保留 1 条、selected/retained chain 处理、query 固定语义和
`partial_query_only_chain_selection` seed stream，以及 multichain variant/row-shuffle
stream；它还绑定四个 dataset 的完整配置、全局与 dataset override 合并后实际生效的
MSA/template 配置、`train_ref_pos_augment`、`limits`、解析后的有效 sampler seed、
Short/Long runtime manifest、partition roles、相同 pair ID/pair receipt，以及 disordered
`indices.csv.gz`、`_SUCCESS.json`、manifest 的实际 SHA256；任一 identity 不一致时启动直接失败。
当前正式 schema-7 training-data contract SHA-256 为
`b76b782fee58b6353992d2ebf8c4138173bf013f5dd7d05344abeaf739a8f55b`。
历史 PDB-only 仍生成原
schema 3 payload，已用
2026-07-24 run manifest 验证 hash 保持不变。

源数据下界也在 2026-07-25 做了只读审计：

- AFDB+MGnify 的 33,498,651 个 fast-path PKL 全部转换成功；全库 canonical/resolved-protein
  token 最小值为 16。最短样例实测为 16 token、119/119 resolved atom；
- disordered 的 28,567 个 bioassembly 全量逐 PKL 计数成功，失败 0；resolved-protein token
  最小值为 53，`<4` 数量为 0。

所以 Short、Long 和 disordered 都不需要额外构建 `<4` eligibility bitset；运行时条件用于防止
crop 退化，并为
普通实验 PDB 提供同一返回契约。

真实读取 smoke 还覆盖了两个分区和两种 shuffle variant：Short compact runtime 最后一行生成
27-token crop、`N_msa=3`；Long 第 0 行生成 384-token crop、`N_msa=64`；PDB `6coi` 与
disordered `1a52` 都分别跑通 segment 和 whole-physical-chain 路径。

## 3. 当前 loss 的真实含义

### 3.1 有效系数

| loss | 外层有效系数 | 当前梯度 |
| --- | ---: | --- |
| smooth lDDT | `4 × 1 = 4` | diffusion + trunk/upstream |
| diffusion MSE | `4`，内部还有 `weight_mse=1/3` | diffusion + trunk/upstream |
| distogram | `0.03` | distogram head + pair trunk/upstream |
| pLDDT | `1e-4` | confidence head only |
| PDE | `1e-4` | confidence head only |
| experimentally resolved | `1e-4` | confidence head only |
| PAE | `1e-4 × 0 = 0` | 无 |
| bond | `4 × 0 = 0` | 无 |

MSE 的配置权重是 DNA/RNA/ligand=`5/5/10`，但实现公式为
`1 + 5*DNA + 5*RNA + 10*ligand`，所以实际 atom weight 是
protein/other=`1`、DNA=`6`、RNA=`6`、ligand=`11`。再乘外层 4 和内部
1/3 后，单原子的基础相对系数约为 `1.333 / 8 / 8 / 14.667`。

confidence loss 只在实验分辨率处于 `[0.1, 4.0]` 时生效，而且 ConfidenceHead
显式 detach trunk，因此它们不会反传 trunk/diffusion。

### 3.2 与官方阶段的关系

当前配方逐项等于仓库官方文档中的 Initial Training：

| 阶段 | crop / DB | smooth lDDT | bond | PAE | diffusion | distogram | confidence only |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Initial | 384 / 48 | 1 | 0 | 0 | 4 | 0.03 | false |
| Fine-tune 1 | 640 / 32 | 0 | 1 | 0 | 4 | 0.03 | false |
| Fine-tune 2 | 768 / 32 | 0 | 1 | 0 | 4 | 0.03 | false |
| Fine-tune 3 | 768 / 32 | 0 | 0 | 1 | 0 | 0 | true |

所以当前 loss 适合作为从随机初始化开始的 initial stage，但不是完整的最终课程。
若 5 万次更新后直接把模型当最终模型，PAE 专属 projection 会一直没有梯度，bond
也没有直接监督。必须提前约定后续 fine-tune 1/2 和 confidence-only stage 3。

### 3.3 zero-weight loss 的工程边界

当前实现即使 PAE/bond 的权重为 0，仍会完整计算它们，再做 `0 × loss`。
如果内部出现 NaN，`0 × NaN` 仍会使总 loss 非有限；trainer 会跨 rank 跳过该次
optimizer update，但会浪费整步。是否对 zero-weight loss 做真正短路，应作为单独
等价优化实现和测试，不与数据比例决策混在一起。

## 4. 多蒸馏数据集扩展规则

profile 的定义位于 `configs/crop384_db48_data_profiles.sh`。底层
`WeightedMultiDataset` 和 `finetune_demo.sh` 支持任意长度的 `train_sets` 与
`train_sample_weights`。历史 `weighted_pdb_only_v1` profile 锁定单集合，历史
`weighted_pdb_afdb_mgnify_disordered_v2` 锁定三集合；当前
`weighted_pdb_afdb_mgnify_short_long_disordered_v3` 按同一接口接入 PDB、Short、Long
和 disordered 四个集合。

以后新增或替换蒸馏数据时，继续遵守以下规则：

1. 在 `configs/configs_data.py` 为每个版本化 catalog 注册独立 dataset key，
   明确结构/index、MSA/template、cropping、sampler 以及
   `is_distillation` 语义；不要只靠 dataset 名称猜测 distillation。
2. 在 profile registry 新增一个**新名字和新 contract tag**，列出有序的
   dataset 名称、正的 outer weights、每数据集必要的 CLI override 和额外
   required paths。任何已经产出 checkpoint 的 profile 都不得原地改变语义。
3. 为新 profile 新建阶段入口和 strict-resume 合同；历史 PDB-only 入口继续锁定
   `weighted_pdb_only_v1`，历史三源入口继续锁定
   `weighted_pdb_afdb_mgnify_disordered_v2`，当前 v3 入口锁定
   `weighted_pdb_afdb_mgnify_short_long_disordered_v3`，均不接受环境变量切换成另一 profile。
4. 新增 CPU/dry-run 测试，核对 dataset 顺序、weights、path preflight、manifest、
   25.6M draw 的暴露量以及 checkpoint 合同不兼容性，再做真实多卡 smoke。

通用 `finetune_demo.sh` 的路径覆盖按实际选中的已知 dataset 注入，不会因为
weightedPDB-only 而强制依赖 SAbDab 路径；新增 profile 还可通过
`PROTENIX_EXTRA_TRAIN_DATA_REQUIRED_PATHS`（冒号分隔）声明额外 catalog/path
预检项。

## 5. 已确认与后续科学项

- 历史 baseline initial stage：weightedPDB-only、official initial loss，已确认且
  schema-3 strict-resume identity 保持不变；
- 历史 disordered-v2 initial stage：PDB / AFDB+MGnify / disordered mass 为
  `0.5 / 0.5 / 0.02`，其 schema-7 contract
  `c22ccd37db686dcad12d51f35bd51de23db9d9702c449f16dd74923fda124d59`
  保持不变；
- 当前 disordered-v3 initial stage：PDB / Short / Long / disordered mass 为
  `0.5 / 0.005 / 0.495 / 0.02`，使用独立 schema-7 contract
  `b76b782fee58b6353992d2ebf8c4138173bf013f5dd7d05344abeaf739a8f55b`；
- 当前数据增强：target occurrence 级 query-only/unpaired-shuffle/
  species-paired=`0.1 / 0.3 / 0.6`，template-use-request=0.2；unpaired-shuffle 中单蛋白
  固定 segment、多蛋白 requested segment/whole-chain=`0.5/0.5`，已写入 strict contract；
- 当前蒸馏集的 dataset mass、artifact 版本/hash 与 distillation label 语义已经进入
  schema-7 contract；
- v1/v2 checkpoint 不能 strict-resume 到 v3；v3 只允许续同一四源、同 pair ID 和同 MSA policy
  的 v3 checkpoint；
- 后续 fine-tune 1/2/3 应各自使用新的入口与 checkpoint 合同。
