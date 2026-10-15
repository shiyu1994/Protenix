# finetune_demo.sh 使用文档

本文档说明当前仓库中 `finetune_demo.sh` 的常用运行方式。内容面向本地 `Protenix_train` 工作目录和 2026 年 Protenix 数据。

> 注意：本页描述的是可复用的底层入口及其历史默认值。当前 crop384/DB48
> 50k-update 正式训练必须通过 `run_crop384_db48_ddp_scratch_50k.sh` 启动；
> 仓库目录名中的 `200k` 仅为历史兼容，不代表当前更新数。该入口
> 使用 `configs/crop384_db48_data_profiles.sh` 将训练数据锁为
> `weighted_pdb_only_v1`（仅 weightedPDB、outer mass 1.0）。未来增加一个或
> 多个蒸馏数据集时，应新增版本化 data profile 和阶段入口，不能直接改变这个
> 已用于 checkpoint 合同的 profile。

## 适用范围

默认配置如下：

| 项目 | 默认值 |
| --- | --- |
| 工作目录 | `/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train` |
| Conda 环境 | `/mnt/project_abd/klp/miniconda3/bin/activate protenix` |
| 数据根目录 | `/mnt/project_abd/public_databases/protenix.20260101` |
| 模型 | `protenix-v2` |
| 初始化 checkpoint | `${PROTENIX_ROOT_DIR}/checkpoint/protenix-v2.pt` |
| 训练集 | `weightedPDB_before210930_v20260101,sabdab-train,distill-sabdab` |
| 训练集采样权重 | `0.5,0.3,0.2` |
| 测试集 | `posebusters_0925` |
| 输出根目录 | `/mnt/project_abd/klp/proteinx_output` |

脚本会自动激活 conda 环境、设置 `PYTHONPATH`、检查必要路径，并通过 `torchrun` 启动 `runner/train.py`。

## 运行前检查

进入工作目录：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train
```

只检查路径，不启动训练：

```bash
CHECK_ONLY=1 bash finetune_demo.sh
```

打印最终命令，不启动训练：

```bash
DRY_RUN=1 SKIP_PATH_CHECK=1 NPROC_PER_NODE=1 bash finetune_demo.sh
```

`DRY_RUN=1` 适合确认参数是否按预期展开，例如数据路径、GPU 数量、checkpoint 路径、验证开关等。

## 训练数据

默认训练数据由三部分组成：

```bash
train_sets=weightedPDB_before210930_v20260101,sabdab-train,distill-sabdab
train_sample_weights=0.5,0.3,0.2
```

`weightedPDB_before210930_v20260101` 使用 2026 Protenix 数据根目录：

```text
/mnt/project_abd/public_databases/protenix.20260101
```

`sabdab-train` 和 `distill-sabdab` 默认使用：

```text
/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix
```

其中 CSV 索引文件位于 `indices/`：

```text
indices/sabdab-train.csv
indices/distill-sabdab.csv
```

两个数据集的 protein MSA 和 template 都通过 sequence 索引到同一个 `mmcif_msa_template/`：

```text
sabdab-train.json
sab-20260416.json
mmcif_msa_template/
```

如需临时覆盖默认数据组合，`train_sets` 和 `train_sample_weights` 的长度必须一致：

```bash
train_sets=weightedPDB_before210930_v20260101,sabdab-train \
train_sample_weights=0.7,0.3 \
bash finetune_demo.sh
```

## 单机训练

使用默认参数训练。`NPROC_PER_NODE=gpu` 会使用当前机器可见的全部 GPU：

```bash
bash finetune_demo.sh
```

单机单卡小步 smoke test：

```bash
NPROC_PER_NODE=1 \
max_steps=4 \
eval_interval=0 \
EVAL_AT_END=false \
checkpoint_interval=-1 \
log_interval=1 \
run_name=protenix_finetune_smoke_1gpu \
base_dir=/mnt/project_abd/klp/proteinx_output_smoke \
bash finetune_demo.sh
```

单机多卡小步测试，例如使用 2 张卡：

```bash
MASTER_PORT=29521 \
NPROC_PER_NODE=2 \
max_steps=4 \
eval_interval=0 \
EVAL_AT_END=false \
checkpoint_interval=-1 \
log_interval=1 \
run_name=protenix_finetune_smoke_2gpu \
base_dir=/mnt/project_abd/klp/proteinx_output_smoke \
bash finetune_demo.sh
```

如果端口冲突，换一个 `MASTER_PORT`。

## 多机多卡训练

脚本使用 `torchrun` 的标准分布式参数：

```bash
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=<rank0_ip_or_hostname> \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
bash finetune_demo.sh
```

另一台机器使用相同的 `NNODES`、`MASTER_ADDR`、`MASTER_PORT`，并把 `NODE_RANK` 改为 `1`。

如果运行环境提供 `MLP_WORKER_NUM`、`MLP_ROLE_INDEX`、`MLP_WORKER_0_HOST`、`MLP_WORKER_0_PORT`，脚本会自动使用这些变量。多机训练时 `MASTER_ADDR` 不能是 `127.0.0.1`。

平台提交训练建议使用 `finetune_submit.sh`。该脚本是训练专用入口，会保留 `test_sets` 配置用于启动时初始化，但默认关闭训练过程中的周期验证和最后验证：

```bash
bash finetune_submit.sh
```

等价于在调用 `finetune_demo.sh` 前固定：

```bash
eval_interval=0
EVAL_AT_END=false
EVAL_ONLY=false
```

checkpoint 评估由独立 infer/eval 脚本在另一个进程中异步执行。

## 训练和验证解耦

EMA 更新只依赖训练步骤中的模型参数，不需要验证集参与。验证只是在某个 checkpoint 或当前模型状态上计算指标。当前脚本支持把训练和验证拆开跑。

训练时完全不跑验证，只保存 checkpoint：

```bash
EVAL_AT_END=false \
eval_interval=0 \
checkpoint_interval=100 \
run_name=protenix_finetune_train_only \
bash finetune_demo.sh
```

只关闭最后一步验证，保留周期验证：

```bash
EVAL_AT_END=false \
eval_interval=400 \
bash finetune_demo.sh
```

只在最后一步验证，不做周期验证：

```bash
EVAL_AT_END=true \
eval_interval=0 \
bash finetune_demo.sh
```

## 独立验证

使用训练产出的普通 checkpoint 和 EMA checkpoint 单独跑验证：

```bash
EVAL_ONLY=true \
LOAD_CHECKPOINT_PATH=/path/to/checkpoints/3.pt \
LOAD_EMA_CHECKPOINT_PATH=/path/to/checkpoints/3_ema_0.999.pt \
run_name=eval_step3 \
base_dir=/mnt/project_abd/klp/proteinx_eval_outputs \
bash finetune_demo.sh
```

只验证普通 checkpoint，不加载 EMA：

```bash
EVAL_ONLY=true \
ema_decay=-1 \
LOAD_CHECKPOINT_PATH=/path/to/checkpoints/3.pt \
LOAD_EMA_CHECKPOINT_PATH= \
run_name=eval_step3_base \
base_dir=/mnt/project_abd/klp/proteinx_eval_outputs \
bash finetune_demo.sh
```

只验证 EMA checkpoint：

```bash
EVAL_ONLY=true \
EVAL_EMA_ONLY=true \
LOAD_CHECKPOINT_PATH= \
LOAD_EMA_CHECKPOINT_PATH=/path/to/checkpoints/3_ema_0.999.pt \
run_name=eval_step3_ema \
base_dir=/mnt/project_abd/klp/proteinx_eval_outputs \
bash finetune_demo.sh
```

验证使用的测试集默认为 `posebusters_0925`。如需切换测试集，可覆盖：

```bash
test_sets=posebusters_0925 bash finetune_demo.sh
```

如果配置中存在多个测试集并且数据路径完整，可以用逗号连接：

```bash
test_sets=posebusters_0925,recentPDB_1536_sample384_0925 bash finetune_demo.sh
```

## 常用参数

| 变量 | 说明 | 示例 |
| --- | --- | --- |
| `PROTENIX_ROOT_DIR` | 数据根目录 | `/mnt/project_abd/public_databases/protenix.20260101` |
| `SABDAB_ROOT_DIR` | SAbDab 数据根目录 | `/mnt/project_abd/zhe/protenix_copy/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix` |
| `NPROC_PER_NODE` | 每台机器使用的 GPU 数 | `1`、`2`、`8`、`gpu` |
| `NNODES` | 机器数量 | `1`、`2` |
| `NODE_RANK` | 当前机器 rank | `0`、`1` |
| `MASTER_ADDR` | rank 0 地址 | `10.0.0.1` |
| `MASTER_PORT` | 分布式端口 | `29521` |
| `run_name` / `RUN_NAME` | 运行名称 | `protenix_finetune_exp01` |
| `base_dir` / `BASE_DIR` | 输出根目录 | `/mnt/project_abd/klp/proteinx_output` |
| `max_steps` | 最大训练步数 | `20000` |
| `checkpoint_interval` | checkpoint 保存间隔，`-1` 表示只保存最后一步 | `1000` |
| `eval_interval` | 周期验证间隔，`0` 表示关闭周期验证 | `0`、`400` |
| `EVAL_AT_END` | 是否在最后一步验证 | `true`、`false` |
| `EVAL_ONLY` | 只验证，不训练 | `true`、`false` |
| `EVAL_EMA_ONLY` | 只记录 EMA 验证结果 | `true`、`false` |
| `EVAL_WORST_SAMPLE_COUNT` | 每次验证记录的 worst-N 样本数量 | `20`、`50` |
| `LOAD_CHECKPOINT_PATH` | 普通模型 checkpoint 路径 | `/path/to/3.pt` |
| `LOAD_EMA_CHECKPOINT_PATH` | EMA checkpoint 路径，可显式置空 | `/path/to/3_ema_0.999.pt` |
| `EMPTY_CACHE_INTERVAL` | 低频执行 `torch.cuda.empty_cache()` 的间隔，`0` 表示关闭 | `0`、`100` |
| `WEIGHTED_TRAIN_SET` | 训练集名称 | `weightedPDB_before210930_v20260101` |
| `WEIGHTED_INDICES_FPATH` | 训练集索引文件 | `/path/to/indices.csv.gz` |
| `train_sets` | 训练集列表，逗号分隔 | `weightedPDB_before210930_v20260101,sabdab-train,distill-sabdab` |
| `train_sample_weights` | 训练集采样权重，长度需匹配 `train_sets` | `0.5,0.3,0.2` |
| `SABDAB_TRAIN_INDICES_FPATH` | `sabdab-train` 索引文件 | `/path/to/indices/sabdab-train.csv` |
| `DISTILL_SABDAB_INDICES_FPATH` | `distill-sabdab` 索引文件 | `/path/to/indices/distill-sabdab.csv` |
| `SABDAB_MSA_TEMPLATE_DIR` | SAbDab MSA/template 目录 | `/path/to/mmcif_msa_template` |
| `POSEBUSTERS_INDICES_FPATH` | posebusters 索引文件 | `/path/to/posebusters.csv` |

## 输出目录

每次运行会创建一个带时间戳的目录：

```text
{base_dir}/{run_name}_{YYYYMMDD_HHMMSS}/
  checkpoints/
  predictions/
  structures/
  dumps/
  errors/
    numerics/
  logs/
    train_metrics.jsonl
    eval_metrics.jsonl  # 有验证时生成
    eval_samples_rank0.jsonl  # 有验证时生成
    eval_worst_samples.jsonl  # 有验证时生成
    runtime_rank0.jsonl
    data_samples_rank0.jsonl
    heartbeat_rank0.jsonl
  manifest/
    run_manifest.json
    run_status.json
  config.yaml
```

训练 checkpoint 示例：

```text
checkpoints/3.pt
checkpoints/3_ema_0.999.pt
checkpoints/checkpoints.jsonl
```

`config.yaml` 是本次运行最终生效的配置，排查参数时优先看这个文件。
`manifest/run_manifest.json` 保存命令、环境、分布式信息、git 状态和数据集摘要。`manifest/run_status.json` 保存最新训练生命周期状态，可直接查看任务是 `running`、`completed` 还是 `failed`，以及最后 step、最新 checkpoint 和异常 traceback。`logs/heartbeat_rank0.jsonl` 保存 rank0 的状态心跳历史。`logs/train_metrics.jsonl` 和 `logs/eval_metrics.jsonl` 保存结构化指标；训练指标里包含 `dataset_window` 和 `runtime_window`。`dataset_window` 用于查看当前日志窗口内各训练集的采样数量、采样比例、token 比例和分数据集 loss 均值；`runtime_window` 用于查看窗口内吞吐、慢 rank、各阶段耗时、显存高水位和梯度健康状态，包括 `samples_per_sec`、`tokens_per_sec`、`atoms_per_sec`、`slowest_rank`、`memory` 和 `grad_health`，其中 `runtime_window.length_bins.token_bins` 按 `n_token` 分桶记录样本数、token/atom 数、loss 均值、step wall time、forward time 和 backward time。eval 聚合指标包含 `avg`、`median`、`pct90`、`min`、`max`。`logs/eval_samples_rank*.jsonl` 保存每个验证样本的指标，`logs/eval_worst_samples.jsonl` 保存每次验证中 loss 最高、lDDT 最低和 ranking score 最低的样本。`logs/runtime_rank*.jsonl` 保存每个 rank 的耗时和显存摘要，`errors/numerics/` 保存 NaN/Inf loss 的轻量诊断记录。

## 常见问题

### 路径检查失败

先确认数据根目录：

```bash
PROTENIX_ROOT_DIR=/mnt/project_abd/public_databases/protenix.20260101 CHECK_ONLY=1 bash finetune_demo.sh
```

如果只想检查命令展开，可以临时使用：

```bash
DRY_RUN=1 SKIP_PATH_CHECK=1 bash finetune_demo.sh
```

### 训练没有跳过验证

需要同时确认两个开关：

```bash
EVAL_AT_END=false eval_interval=0 bash finetune_demo.sh
```

`eval_interval=0` 只关闭周期验证；`EVAL_AT_END=false` 关闭最后一步验证。

### 只验证普通 checkpoint 时 EMA 报错

使用普通 checkpoint 单独验证时，关闭 EMA 并把 EMA 路径置空：

```bash
EVAL_ONLY=true ema_decay=-1 LOAD_CHECKPOINT_PATH=/path/to/3.pt LOAD_EMA_CHECKPOINT_PATH= bash finetune_demo.sh
```

### 多机启动卡住

检查以下变量是否在所有节点一致：

```bash
NNODES=<node_count>
MASTER_ADDR=<rank0_ip_or_hostname>
MASTER_PORT=<same_port>
NPROC_PER_NODE=<gpus_per_node>
```

每个节点的 `NODE_RANK` 必须不同，并从 `0` 开始。

### 显存碎片或显存回收问题

默认不在每步调用 `torch.cuda.empty_cache()`。如需低频释放缓存：

```bash
EMPTY_CACHE_INTERVAL=100 bash finetune_demo.sh
```

这会每 100 个训练 step 调用一次 `empty_cache()`。
