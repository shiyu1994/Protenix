# 训练期 MSA raw 预算与最终 16,384 合并池

## 结论

本代码库默认把训练阶段 paired 与 unpaired MSA 候选读取分别限制为原始记录
顺序中的 top 10,000，并把限制放到去重和数值特征构建之前。两个 pool
独立记账，未用配额不互相转移；合并后的 assembled training pool 仍最多
16,384 行。AFDB/MGnify Short/Long 是显式的数据集级例外：每个预测单体只有
一个 direct sibling A3M，且它只作为 unpaired MSA 使用，因此该 unpaired raw
pool 可以读取 top 16,384。该策略不改变其他蛋白或 RNA 数据集。

本实现没有提前裁剪 MSA 列。`crop_size=384` 仍然先对整个 target 的 token
做 crop，蛋白、RNA/DNA 和配体都在同一个 crop 中；`max_nuc_tokens=128`
只额外限制其中的核酸 token 数。MSA 最后再按 crop 选中的 token 映射列。

## 精确语义

训练阶段当前正式配置为：

```text
default paired raw candidate budget                  = 10000
default unpaired raw candidate budget                = 10000
AFDB/MGnify Short/Long direct-unpaired raw budget    = 16384
final assembled pool                                 <= 16384
```

规则如下：

1. paired 和 unpaired 是两个独立候选池，默认各自最多读取 10,000 条原始
   record；Short/Long direct-unpaired 是上文明确的 16,384 例外。
2. 同一个池包含多个 source 时保持原配置顺序，前一个 source 用掉的行数
   会从该池的剩余预算中扣除。
3. 截断发生在 dedup 之前。因此默认池前 10,000 条中若有重复，去重后的实际
   深度可以小于 10,000；Short/Long unpaired 同理在前 16,384 条截断，均不会
   越过相应 raw 前缀继续补齐。
4. 两池不足时都不 padding，也不把一侧未用预算借给另一侧。两侧都达到预算时
   raw 候选最多 20,000 行，但下游合并仍按现有顺序截为最多 16,384 行。
5. A3M 的 row 0 仍位于输入前缀中。后续模型侧的随机 MSA 行抽样行为保持
   现有合同：full-MSA 分支不会额外固定 row 0。
6. `query-only` 分支仍跳过 homolog I/O，不受此优化改变。
7. test/eval/inference 不启用新增的训练期 raw 读取限制；但最终
   assembled test MSA 仍受现有 `max_size.test=16384` 约束，RNA 读取也仍保留
   历史单源 30,000 上限。
8. AFDB/MGnify Short/Long runtime 样本固定为单蛋白链。其同名 `.a3m`
   通过 `direct_protein_unpaired_msa_path` 进入 unpaired pool，使用 16,384
   的数据集级上限；它没有真实 paired homolog source。日志中的
   `prot_pair_num_alignments=1` 只是标准 query placeholder。

蛋白默认 source 顺序保持不变：

```text
普通 A3M unpaired: 配置中的 raw path / database 顺序
OpenFold3 unpaired: uniref90 -> cfdb -> mgnify
OpenFold3 paired:   uniprot（独立预算）
AFDB/MGnify Short/Long: sibling .a3m -> direct unpaired（16,384）
```

## Crop 与蛋白/核酸的关系

这里有两个不同维度，不应混为一谈：

```text
MSA row 维度：默认 paired/unpaired raw 各 top 10,000；
              AFDB/MGnify Short/Long unpaired top 16,384；
              assembled pool 最多 16,384
MSA column 维度：沿用现有逻辑，最终按整个 target 的 crop token 映射
```

- `crop_size=384` 是整个 target 的 token 上限，不区分蛋白、核酸或配体。
- `max_nuc_tokens=128` 是上述 384 个位置中的核酸 token 子上限。
- 标准蛋白残基和标准核苷酸通常各对应一个 token。
- atom-token 化的非标准残基可能对应多个 token；这些 token 共享
  `(asym_id, res_id)`，会映射到同一个 canonical MSA 列。因此它们会重复
  gather 同一列，但在 `max_nuc_tokens` 中按实际 token 个数占用名额。

本轮没有把 A3M 提前投影成 crop 的少量列。这样可以避免改变蛋白
paired/unpaired pairing、全宽 dedup、all-gap filtering 和 deletion count
语义。若后续继续做 RNA 列投影，需要同时保留全宽 row fingerprint、
all-gap 状态，并按完整 alignment 的前一列计算 deletion，不能只把
`selected_indices` 直接传给一个简单切片。

## 核心实现

- `protenix/data/msa/msa_utils.py`
  - 流式解析 A3M record 前缀，不再先把整个文件读入字符串。
  - `RawMsa` 在 dedup 前应用 `depth_limit`。
  - OpenFold aligned array 在 Python 字符串转换前切 top-N。
  - 截断后的 deletion matrix 使用紧凑独立前缀，不再通过 NumPy view
    挂住完整 50,000 行底层数组。
  - merge 保留原 source 行序、删除矩阵和独立对象语义。
- `protenix/data/msa/msa_featurizer.py`
  - paired/unpaired 分别维护跨 source 的剩余预算。
  - 普通 A3M、direct A3M、RNA A3M 和 OpenFold3 路径统一接入上限。
  - 避免 RawMsa -> A3M 文本 -> RawMsa 的重复全文往返。
  - 所有 chain 完成 representative/source 复用并写入 meta 后，立即释放
    source cache，避免 source、meta 和 assembly 三份矩阵同时存活。
  - profile 改为分块 `bincount`，结果与旧 broadcast one-hot 计算一致，
    但不再创建 `N × L × 32` 临时数组。
- `protenix/data/pipeline/dataset.py`
  - 只在 `stage == "train"` 时传入 paired/unpaired raw 上限；
  - 合并数据集级 MSA 覆盖，Short/Long unpaired 使用 16,384；
    `max_size.train=16,384` 继续作为最终 assembled pool 上限。
- `tests/test_msa_input_depth_limit.py`
  - 覆盖流式提前停止、dedup 前截断、跨 source 预算、paired/unpaired
    独立预算、空文件、OpenFold 删除矩阵、profile parity 和非标准 RNA
    多 token 映射。

## 真实慢样本结果

下表是默认双 10,000 合同之前、paired/unpaired 各 16,384 的前置实现结果，
只保留为瓶颈定位证据。它不能代表当前“默认双 10,000、Short/Long
direct-unpaired 16,384”合同的吞吐、深度或峰值；当前实现需要单独做相同样本
和 128-card matched benchmark。

测试使用 `nice -n 19` 和单线程 BLAS，在当前共享机器上执行，不占 GPU。
旧值来自 128-card run 的整 target `msa_featurizer_seconds`；新值是当前实现的
单链或局部路径，不是严格端到端 matched A/B，因此只能用于确认瓶颈是否被
消除，不能把比值当作正式训练吞吐提升。

| 样本/路径 | 旧整 target | 当前路径 | 备注 |
|---|---:|---:|---|
| `7am2_73` RNA | 72.96 s | 9.07 s | 6,956 行，小于 16,384；收益主要来自取消重复转换和低内存 profile |
| `6ybs_18` RNA（旧日志 target 为 `6zme`） | 76.33 s | 2.11 s | 26,839 行，实际只读取前 16,384 |
| `3j7q_1` RNA | 62.11 s | 3.23 s | 23,017 行，实际只读取前 16,384 |
| `3h0l` OpenFold3 三条蛋白链 | 41.33 s | 12.75 s | 最终 `(16384, 1050)`；paired/unpaired 各 8,192 |

`7am2_73` 的 full-width RNA 为 6,956 × 19,000。当前真实
`FeatureAssemblyLine` 在最后只输出 32 个 crop 列，但仍先对 19,000 列做
数值特征和 profile，因此峰值 RSS 约 3.19 GiB。它已经从旧日志的 72.96 秒
降到约 9.07 秒，但也明确标出了未来 RNA crop-aware 列投影仍可继续优化的
内存空间。

OpenFold3 `.npz` 目前把每个 source 保存为 scalar object。代码可以在数值
转换前切 top-N，但 `numpy` 仍必须先完整解压并反序列化该 source。因此
OpenFold3 的 CPU conversion/profile 长尾会下降，压缩文件 I/O 和解压峰值
不会随 top-N 成比例下降。要实现真正的 NPZ 限读，需要另行把离线格式改为可
切片的独立 `.npy`/memmap、Zarr 或等价结构。

`3h0l` 当前三链路径中，source load + meta merge 约 6.17 秒，
`FeatureAssemblyLine` 约 6.58 秒；其中字符串 MSA 数值化约 4.68 秒，是最大
剩余 CPU 热点。新 profile 三链合计约 0.22 秒；同一 A 链 matched profile
上与旧结果逐元素完全一致，中位耗时约快 7.1 倍。历史 41.33 秒来自旧训练日志，
当前 12.75 秒来自 MSA core 复现，不含外围结构读取，仍应视作近似 matched。

额外的 source-cache 生命周期复测在 assembly 前释放 cache 并显式
`gc.collect()`，checksum 完全不变，峰值从约 1.465 GiB 降至 1.227 GiB，
下降约 244 MiB（16.2%）；清理瞬间 live RSS 下降约 192 MiB。生产代码只释放
引用，不在每条样本上强制执行有额外开销的全局 GC。

## 验证

聚焦回归：

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python -m pytest -q \
  tests/test_msa_input_depth_limit.py \
  tests/test_msa_pairing_mode.py \
  tests/test_query_only_msa_features.py \
  tests/test_partial_query_only_msa.py \
  tests/test_virtual_segment_msa_shuffle.py \
  tests/test_openfold3_disordered_integration.py \
  tests/test_distillation_direct_io.py \
  tests/test_data_edge_cases.py
```

最终相关回归集合共 116 项通过；最后的 OpenFold source-cache 生命周期修改又
单独通过 30 项相关回归。`py_compile`、launcher `bash -n` 和
`git diff --check` 均通过。

另外应在正式 128-card 作业前跑一个 DataLoader-only 或短训练 smoke，检查：

- `msa_featurizer_seconds` 的 P95/P99/max；
- `N_msa`、protein/RNA paired/unpaired depth；
- 每个 MSA 三分类分支的比例；
- DataLoader wait 与整步 straggler；
- worker RSS，尤其是超长 RNA 和 OpenFold3 scalar-object NPZ。

## Checkpoint 合同边界

默认 paired/unpaired raw top-10,000（Short/Long unpaired 为 top-16,384）
在 dedup 前执行，可能让最终候选集合与旧“全量读取后再 crop”
路径不同，所以它不是 trajectory-exact 的纯实现替换。若从已有训练 checkpoint
继续，必须把它记录为新的 data-policy/contract 版本；不能在文档上宣称与旧
checkpoint 是相同的数据轨迹。模型参数、optimizer state 和 checkpoint 文件
结构本身没有被本轮代码修改。
