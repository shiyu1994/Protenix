结论先说：AlphaFold 3 官方没有公开 total loss 或各分项 loss 随 step 的数值曲线。官方公开的是验证集 LDDT 随 optimizer step 上升的曲线。因此我们不能严谨地说“AF3 的 MSE 从多少降到多少”，但可以完整分析：

  - AF3 每项 loss 的定义、权重和启停阶段；
  - AF3 不同能力的实际收敛速度；
  - 我们当前相同 loss 的下降是否合理。

  ## 1. AF3 真正公开的收敛曲线

  AF3 Fig. 2d 和 Extended Data Fig. 2 的纵轴是验证集 LDDT，不是训练 loss。图中的“90%/97%”是“达到该指标初始训练峰值的 90%/97%”，不是 LDDT=90/97。AF3 Nature 论文 (https://www.nature.com/articles/s41586-024-07487-w)

  从 Extended Data Fig. 2 读出的近似结果如下：

   验证能力                     达到初训峰值 90%    达到初训峰值 97%    最终 LDDT 约值    特征
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━
   Ligand intrachain                        2–3k                  8k                94    最快
  ───────────────────────────  ──────────────────  ──────────────────  ────────────────  ────────────────
   DNA intrachain                           3–4k                 11k                80    很快
  ───────────────────────────  ──────────────────  ──────────────────  ────────────────  ────────────────
   Protein intrachain                       3–5k                 13k             83–84    很快
  ───────────────────────────  ──────────────────  ──────────────────  ────────────────  ────────────────
   RNA intrachain                             5k                 17k             70–71    局部任务中最慢
  ───────────────────────────  ──────────────────  ──────────────────  ────────────────  ────────────────
   Protein–DNA interface                  10–12k            20k 左右             52–54    噪声较大
  ───────────────────────────  ──────────────────  ──────────────────  ────────────────  ────────────────
   Protein–ligand interface                7–10k            27k 左右             62–63    中等速度
  ───────────────────────────  ──────────────────  ──────────────────  ────────────────  ────────────────
   Protein–RNA interface                  12–15k            40k 左右             42–44    很慢且波动大
  ───────────────────────────  ──────────────────  ──────────────────  ────────────────  ────────────────
   Protein–protein interface              15–16k              约 60k             59–60    收敛最慢之一

  官方明确总结：

  - 所有 intrachain 指标在前 20,000 step 内达到初训最大值的 97%；
  - protein–protein interface 要到大约 60,000 step；
  - 有些能力很早达到峰值后反而下降，因此 AF3 依靠数据采样权重调整和验证集 early stopping，而不是仅看训练 loss；
  - crop 640/768 微调对局部结构提升很小，最大的收益来自蛋白界面。训练曲线与说明 (https://www.nature.com/articles/s41586-024-07487-w#Sec5)

  这说明 AF3 的核心学习顺序是：

  > 局部化学与单链折叠 → 配体/核酸界面 → 蛋白–蛋白全局装配。

  ## 2. AF3 的 loss 组成和阶段变化

  官方总目标为：

  [
  L =
  10^{-4}(L_{\rm pLDDT}+L_{\rm PDE}+L_{\rm resolved}
  +\alpha_{\rm PAE}L_{\rm PAE})
  +4L_{\rm diffusion}
  +0.03L_{\rm distogram}
  ]

  详见 AF3 Supplementary Information (https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf)。

   阶段       crop / diffusion batch    Structure / distogram    smooth-LDDT    bond        PAE
  ━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━
   Initial    384 / 48                  开                       开             权重 0      权重 0
  ─────────  ────────────────────────  ───────────────────────  ─────────────  ──────────  ─────────────────────
   FT1        640 / 32                  开                       关闭           权重 1      0
  ─────────  ────────────────────────  ───────────────────────  ─────────────  ──────────  ─────────────────────
   FT2        768 / 32                  开                       继续关闭       权重 1      0
  ─────────  ────────────────────────  ───────────────────────  ─────────────  ──────────  ─────────────────────
   FT3        768 / 32                  关闭                     不再参与       不再参与    开启 confidence/PAE

  因此跨阶段比较 total loss 是没有意义的：

  - FT1 关闭 smooth-LDDT，total loss 会突然大幅下降，即使模型性能完全没变化；
  - FT1 同时打开 bond，又会增加一个新项；
  - FT3 关闭 diffusion 和 distogram，只训练 confidence 系统，loss 尺度会缩小几个数量级。

  ## 3. 每个 loss 应该如何理解

  ### MSE

  这是经过刚体对齐、原子类型加权和噪声级别加权的坐标误差：

  - 蛋白原子权重 1；
  - DNA/RNA 权重 6；
  - ligand 权重 11；
  - 先将 GT 刚体对齐到预测，再计算误差。

  它通常应当是早期下降最快的结构项，对应模型迅速学会“从噪声恢复一个合理结构”。

  但 AF3 论文 Eq. 6 字面使用的 noise weight 分母是 ((\hat t+\sigma_{\rm data})^2)，我们当前实现使用的是 ((\hat t\sigma_{\rm data})^2)。官方训练源码没有公开，因此无法判断论文是排版问题还是实际实现差异。当前代码位置
  见 Protenix_train2_templatecache_top60_uniform20_id90/protenix/model/loss.py:1695。

  ### smooth-LDDT

  直接优化原子对距离是否落在 0.5、1、2、4 Å 阈值内：

  - 蛋白邻域 15 Å；
  - 核酸邻域 30 Å；
  - 当前外层权重为 4。

  它最直接对应 AF3 前期快速提升的 intrachain LDDT。AF3 在 FT1 将它关闭，很可能是因为局部几何已经基本学会，继续强化它对界面收益有限。

  ### Distogram

  64 类 token-pair 距离交叉熵，主要监督 trunk/pair representation。它下降说明 pairformer 学到更清晰的几何关系，但不能直接换算成最终坐标 LDDT。

  ### pLDDT、PDE、resolved

  这些是 confidence head 的分类交叉熵：

  - pLDDT：50 bins；
  - PDE：64 bins；
  - resolved：二分类；
  - 总权重都只有 (10^{-4})。

  AF3 对 confidence head 的 trunk 表征和 mini-rollout 坐标做 stop-gradient，所以这些 loss 的下降主要训练置信度预测器，不会直接推动 pairformer/diffusion 改善结构。我们也做了相同处理，见
  Protenix_train2_templatecache_top60_uniform20_id90/protenix/model/protenix.py:730。

  ### PAE

  前三个结构阶段权重为 0，最终 confidence 阶段才开启。当前 PAE loss 保持在：

  [
  \ln(64)=4.15888
  ]

  附近是完全正常的，表示 64-bin 输出仍接近均匀分布，并不是训练失败。

  ### Bond

  Initial 阶段权重为 0，FT1/FT2 才开启。当前日志里的 raw bond 即使下降，也只是坐标预测整体改善造成的间接结果，并没有参与反向优化。

  ## 4. 和我们当前训练的实测对照

  当前运行截点为 Step 7324；它不是完全从随机初始化开始，而是参数级加载了 v8 的 step 999，且 optimizer/scheduler 重新开始，见 /mnt/project_abd/klp/proteinx_output/
  protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/config.yaml:1631。

  比较本段训练最初 500 update 与最近 500 update：

   分项                   前 500 均值    最近 500 均值        变化
  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━  ━━━━━━━━━━
   Total weighted loss         2.3164           1.3071     -43.57%
  ─────────────────────  ─────────────  ───────────────  ──────────
   smooth-LDDT raw            0.38706          0.26768     -30.84%
  ─────────────────────  ─────────────  ───────────────  ──────────
   MSE raw                    0.18087          0.05260     -70.92%
  ─────────────────────  ─────────────  ───────────────  ──────────
   Distogram CE               1.47865          0.85950     -41.87%
  ─────────────────────  ─────────────  ───────────────  ──────────
   pLDDT CE                   3.22541          2.56034     -20.62%
  ─────────────────────  ─────────────  ───────────────  ──────────
   PDE CE                     3.41955          1.88773     -44.80%
  ─────────────────────  ─────────────  ───────────────  ──────────
   Resolved CE                0.13090          0.06792     -48.11%
  ─────────────────────  ─────────────  ───────────────  ──────────
   Bond raw，权重 0           0.01539          0.00330     -78.58%
  ─────────────────────  ─────────────  ───────────────  ──────────
   PAE raw，权重 0            4.15798          4.15829    基本不变

  最近 total loss 1.307 中的贡献比例大约是：

  - smooth-LDDT：81.9%
  - MSE：16.1%
  - distogram：2.0%
  - 三个 confidence 项合计：0.035%

  这是一条很合理的 AF3 Initial-style 训练曲线：

  1. MSE 下降最快，说明 denoiser 已经学到明显的坐标恢复能力。
  2. smooth-LDDT 下降更慢并逐渐主导 total loss，说明训练进入局部几何精修阶段。
  3. distogram、PDE、pLDDT 同时下降，没有出现 trunk 学习停滞或 confidence head 脱节。
  4. PAE 完全不降是当前权重为 0 的预期结果。
  5. 最近 500 update 相比前一个 500 update，total 仍下降约 0.84%，MSE 下降约 2.67%，还没有真正停止学习，只是进入缓慢阶段。

  特别注意：当前 smooth-LDDT raw=0.268，可以理解为训练噪声样本上的 soft-LDDT proxy 约 0.732，但绝不能直接和 AF3 验证集最终 LDDT 0.9 左右比较——一个是单步带噪训练目标，一个是完整 diffusion rollout 后的固定验证集指
  标。

  ## 5. 对我们训练计划的实际含义

  我们是 global target batch 512、diffusion batch 48；AF3 是 target batch 256、diffusion batch 48。因此：

  - 我们一次 update 处理的 trunk target 是 AF3 的 2 倍；
  - diffusion samples/update 是 24,576，AF3 是 12,288；
  - 当前约 8.3k 累计结构更新（前置约 1k + 当前约 7.3k）已经看过约 4.26M target，AF3 的 20k update 对应约 5.12M target。

  所以按“样本曝光量”粗略衡量，我们已经接近 AF3 局部指标进入平台的阶段，这和当前 MSE 快速下降后、smooth-LDDT 主导且整体变慢是吻合的。但我们的 PF64/c_z256、LR 1e-3、数据组成和 AF3 PF48/c_z128、LR 1.8e-3 不同，不能直
  接按 step 等比例推断。

  最重要的结论是：当前 loss 下降健康，但不能靠它判断蛋白界面是否学好。AF3 自己也观察到训练 loss 继续改善时，部分验证能力已经过拟合下降。我们下一步真正需要的是固定验证集，至少记录 AF3 同样的八类指标：

  - protein/ligand/DNA/RNA intrachain LDDT；
  - protein–protein、protein–ligand、protein–DNA、protein–RNA interface LDDT。

  尤其应该重点观察 20k–30k 附近的 protein–protein/interface 指标；按样本曝光量折算，我们的 30k update 已接近 AF3 60k update 的数据量。50k 可以作为计划终点，但最佳 checkpoint 很可能不是最后一个。
