# 显式频段加权能否提升 AT-DGNN 的脑电情感识别性能？——一项关于融合设计与随机种子的受控研究

**Can Explicit Frequency-Band Weighting Improve AT-DGNN for EEG Emotion Recognition? A Controlled Study of Fusion Design and Random Seeds**

---

## 摘要

**目的**：频率信息是脑电（EEG）情感识别的核心判别依据，但深度模型对"哪些频段更重要"的建模往往并不充分。本文以 AT-DGNN（BIBM 2024）为基线，检验一个自然假设：**为该模型引入显式的、逐样本自适应的频段加权，能否提升识别性能**。

**方法**：首先通过代码级分析说明，AT-DGNN 的 Tception 时域学习器在功能上等价于"可学习滤波器组 + 对数功率"，即一种学习得到的频谱分解，但其频段加权存在三点不足——**静态**（全样本共享）、**未与规范 EEG 频段对齐**、且三分支输出仅被**拼接**而从未被重新加权。为此本文提出两种显式加权设计：（i）**输入侧频段注意力**，用固定砖墙掩码把信号分解为 δ/θ/α/β/γ 五个规范频段，以"可学习静态先验 + 共享网络的逐样本自适应调制"生成权重并加权融合；（ii）**分支侧多尺度注意力**，为三个频率尺度不同的 Tception 分支学习权重。两者均在**未改动的** AT-DGNN 主干之上工作。本文的关键设计准则是**恒等性**：通过残差式融合与分数头零初始化，使模块在初始化时**逐比特等于基线**，从而任何性能变化都可归因于所学到的加权。

**结果**：在 MEEG 数据集上采用试次级 3 折交叉验证 × 15 个随机种子（每个配置 45 次独立训练）的按种子配对协议，得到三点结论：（1）当融合方式保证恒等性时，两种模块均**与基线无显著差异**（频段注意力 $\Delta=+0.04$ 个百分点，$p=0.78$；多尺度注意力最好变体 $\Delta=+0.12$ 个百分点，$p=0.40$），参数量仅增加 30–31。（2）**融合方式本身的影响远大于模块本身**：同一频段注意力模块，仅把"替换输入"改为"残差修正"，性能从 **−5.60 个百分点（$p=0.0033$，14/15 个种子变差，崩溃率由 11% 升至 24%）** 变为 **+0.04 个百分点（$p=0.78$）**，两者相差 **5.6 个百分点**。（3）**机制实测解释了中性结果**：模型学到的时间权重与频段权重都收敛到近似均匀（归一化熵均为 1.0000，最大偏离 1.34% 与 1.53%），且在试次级检验中没有任何频段的类间差异通过 Bonferroni 校正。

**结论**：在本文的单被试受控条件下，**显式频段加权未能提升 AT-DGNN 的性能**；模型自发地选择了近似均匀的加权，说明该数据上频段重要性的差异不足以提供可利用的判别增益。本文最有实用价值的结论是一条设计准则：**在向已有模型插入加权模块时，融合方式必须保证恒等性**——不精确的恒等路径会带来百分之几量级的显著劣化，其影响远超过模块所能带来的潜在收益。此外，本文给出的随机种子协议与统计层级分析表明，单次实验或片段级显著性检验都可能给出误导性结论。

**关键词**：脑电；情感识别；频段注意力；图神经网络；融合设计；多随机种子

**中图分类号**：TP183　　**文献标志码**：A

---

## 1 引言

### 1.1 背景

脑电（EEG）信号直接记录大脑皮层神经活动，具有时间分辨率高、不易伪装、能够反映个体内在情感状态等优点，是情感计算与情感脑机接口的重要信号来源。在基于深度学习的 EEG 情感识别中，两个问题长期受到关注：**如何构造电极之间的连接关系**，以及**如何从多频段信号中提取判别特征**。

频段信息的重要性有明确的神经生理依据。EEG 节律按频率划分为 δ（1–4 Hz）、θ（4–8 Hz）、α（8–13 Hz）、β（13–30 Hz）、γ（30–50 Hz）五个经典频段，不同频段与情感状态存在系统性关联：α 波与放松状态相关，β 波与紧张、警觉等高唤醒状态相关。正因如此，基于频段的微分熵（DE）、功率谱密度（PSD）等特征长期是 EEG 情感识别中性能最强的手工特征之一。

由此产生一个自然的假设（记为 **H1**）：

> **H1**：在已有的深度 EEG 情感识别模型中引入**显式的、逐样本自适应的频段加权**，能够提升识别性能。

### 1.2 基线在频段加权上的具体不足

本文以 AT-DGNN（BIBM 2024）为基线。其提出的 Tception 时域学习器由三个不同核长的并行分支构成：

$$
\textbf{Tception}_k(x)=\mathrm{PowerLayer}\!\left(\mathrm{Conv2d}_{(1,\ell_k)}(x)\right),\qquad
\mathrm{PowerLayer}(z)=\log\!\big(\mathrm{AvgPool}(z^2)\big)
$$

其中 $\ell_k\in\{100,50,25\}$ 为卷积核长度（0.5/0.25/0.125 s）。该结构在功能上等价于**一组可学习滤波器 + 对数功率**，即一种学习得到的频谱分解；三个分支的低频截止约为 $f_s/\ell_k\approx 2/4/8$ Hz，因而**在功能上对应不同的频率侧重**。然而，本文通过代码级分析指出其在频段加权方面存在三点不足：

1. **静态加权**：滤波器的频率响应完全由训练好的卷积核决定，对**所有样本相同**，模型无法表达"对当前这段信号而言 β 频段更具判别力"。
2. **未与规范频段对齐**：每个输出通道是任意学习到的频谱响应，不对应 δ/θ/α/β/γ 中的某一个，其重要性无法与神经生理学结论对照。
3. **仅拼接而未加权**：三个分支的输出仅通过 `torch.cat(·, dim=-1)` 沿时间维拼接，既无权重也无竞争机制；而后续的特征整合为固定核卷积，沿拼接后的坐标轴均匀作用，**无法**单独缩放某一个分支。

因此，基线具备频谱分解能力，但缺少一个显式的、逐样本自适应的频段加权机制。这正是 H1 所要检验的对象。

### 1.3 本文工作与主要发现

本文设计并实现了两种显式加权模块（§3），并在统一协议下与基线做受控比较（§4）。主要发现如下：

1. **融合方式的影响远大于模块本身（本文最有实用价值的结论）**。同一频段注意力模块，采用"替换输入"的融合方式时显著劣化 **−5.60 个百分点**（$p=0.0033$，14/15 个种子变差，崩溃率由 11% 升至 24%）；仅改为"残差修正"的融合方式后变为 **+0.04 个百分点**（$p=0.78$，与基线无差异）。二者相差 **5.6 个百分点**，而模块参数仅为 30 个。原因是前者在权重取等值时**不能精确还原输入**（实测最大偏差 $2.0\times10^{-3}$，波形 RMS 偏差约 14%），后者则可以（**逐比特一致**，最大绝对误差 0.0）。
2. **H1 未被支持**。在保证恒等性的前提下，两种模块（频段注意力与多尺度注意力）及其消融变体均与基线无显著差异，最好变体仅为 +0.12 个百分点（$p=0.40$）。
3. **机制实测解释了中性结果**。模型学到的时间-频段权重（5 维）与尺度权重（3 维）均收敛到近似均匀：归一化熵均为 **1.0000**，最大偏离分别仅 **1.34%** 与 **1.53%**；且没有任何频段的类间差异在试次级检验中通过 Bonferroni 校正。说明在该数据上频段重要性的差异不足以支撑可学习的加权增益，模型的最优策略即保持均匀加权（等价于退回基线）。
4. **统计方法学**：本文通过实测说明两点——随机种子重复的必要性（同一模型在 15 个种子上的准确率极差达 **30.56 个百分点**），以及"试次/片段"两级混淆带来的**伪重复**陷阱（同一份数据，片段级检验认为 3/5 个频段"显著"，而有效的试次级检验为 0/5）。

---

## 2 相关工作

### 2.1 EEG 情感识别中的频段特征

主流方法多对每个电极、每个频段计算微分熵（DE）或功率谱密度（PSD），再送入分类器。这类方法的共同前提是**频段划分事先固定**，且各频段的融合方式（通常为拼接）不参与学习。本文关注的正是"频段融合方式是否可学习、是否有益"这一问题。

### 2.2 EEG 图神经网络

图神经网络通过建模电极间交互提升性能，按图构造方式可分为固定图（电极距离或解剖归属）、静态可学习图（可学习邻接矩阵，全样本共享）、动态不可学习图（由节点特征自相似度生成逐样本图，AT-DGNN 使用的 $\hat{A}=D^{-1/2}(XX^{\top}+I)D^{-1/2}$ 属此类）与动态可学习图。本文工作位于**特征侧**而非图构造侧，与上述路线互补。

### 2.3 注意力机制与特征维加权

注意力通过对特征维或空间位置生成归一化权重实现特征选择。需要指出的是，softmax 归一化使每个元素被约束在均值的 $1/N$ 量级，当被加权维度很大时（例如上千个时间位置）权重难以产生有效选择性；而当维度较小（本文的 5 个频段与 3 个尺度）时具备充足的调节余量。此外，**向已有模型插入加权模块时的融合方式**鲜有被系统讨论，本文将其作为独立变量进行受控实验，并发现其影响远超模块本身。

---

## 3 方法

### 3.1 总体框架

本文的两种模块都只作用于**输入侧**，其后是**未作任何改动的** AT-DGNN 主干，因此性能差异可归因于加权模块本身：

```
                 ┌── 模块A（输入侧频段注意力）──┐
EEG x ── 固定频段分解 → 频段打分 → 加权融合 → x' ──┐
                 └──────────────────────────────┘   │
                                                     ▼
                 ┌── 模块B（分支侧多尺度注意力）──┐
                                     AT-DGNN 主干（未改动）
                 └──────────────────────────────┘
```

### 3.2 模块 A：输入侧频段注意力

#### 3.2.1 固定频段分解

用砖墙式频域掩码把输入划分为 $K=5$ 个规范频段：

$$
X_k=\mathrm{irFFT}\big(\mathrm{rFFT}(x)\odot M_k\big),\qquad
M_k(f)=\begin{cases}1,& f\in[f_k^{lo},f_k^{hi})\\0,&\text{otherwise}\end{cases}
$$

频段区间取 $\{(1,4),(4,8),(8,13),(13,30),(30,50)\}$ Hz。掩码互为补集，故

$$
\sum_{k=1}^{K}M_k(f)=1\quad(f\in[1,50)\ \text{Hz})
$$

即五个频段的掩码构成对 1–50 Hz 的**单位分解**。实测各频段的带内能量占比均为 **100.0%**（无频段间泄漏、无带内失真）。该实现方式也与 EEG 领域计算 DE/PSD 特征时的常规做法一致。

> **实现说明**：本文亦实现了基于线性相位 FIR 的分解方案作为对照。由于 Hann 窗 FIR 的过渡带宽度约为 $4f_s/L$（$L=129$ 时约 6 Hz），相邻频段重叠严重（实测带内能量占比仅 86%–98%），会污染频段权重的解释，故本文采用频域掩码。

#### 3.2.2 频段重要性打分与加权

$$
d_k=\log\!\left(\frac{1}{CT}\sum_{c,t}\big(X_k\big)_{c,t}^{2}+\epsilon\right),\qquad
\tilde d_k=d_k-\frac{1}{K}\sum_j d_j
$$

$$
s_k=\underbrace{b_k}_{\text{静态频段先验}}+\underbrace{\phi(\tilde d_k)}_{\text{逐样本自适应}},
\qquad
\beta_k=K\cdot\mathrm{softmax}_k(s)
$$

其中 $b_k$ 为可学习参数（刻画"频段 $k$ 总体上是否重要"），$\phi:\mathbb{R}\to\mathbb{R}^{h}\to\mathbb{R}$ 为在所有频段间**共享权重**的两层映射（$h=8$，刻画"依据当前样本的频段功率分布重新分配权重"）。

#### 3.2.3 两种融合方式（本文的核心设计变量）

**方式 1（替换，replace）**：

$$
x'=\sum_{k=1}^{K}\beta_k X_k
$$

其问题在于，即便权重取等值（$\beta_k\equiv1$），输出也只是 $\sum_k X_k$，即 $x$ 在 1–50 Hz 内的成分，**并不等于 $x$**：实测最大绝对偏差 $2.0\times10^{-3}$、波形 RMS 相对偏差约 14%（对应预处理在 1–50 Hz 之外约 2% 的能量残留）。换言之，该路径的"恒等"是**不精确**的。

**方式 2（残差修正，residual，本文默认）**：

$$
x'=x+\sum_{k=1}^{K}\big(\beta_k-1\big)X_k
$$

当 $\beta_k\equiv1$ 时 $x'=x$ **精确成立**。配合把 $\phi$ 的末层**零初始化**（使其初始输出为 0，从而 $\beta_k\equiv1$），模块在初始化时**逐比特等于基线**，之后训练只能朝"有利方向"偏离。实测该性质在三种消融模式下均成立（最大绝对误差 0.0）。

### 3.3 模块 B：分支侧多尺度注意力

针对 3.1 节指出的第三点不足（三分支仅拼接、后续固定卷积无法单独缩放某一分支），本文为三个频率尺度不同的 Tception 分支学习权重：

$$
d_j=\log\!\Big(\tfrac{1}{CET}\textstyle\sum y_j^2\Big),\quad
s_j=b_j+\phi\!\big(d_j-\bar d\big),\quad
\beta=J\cdot\mathrm{softmax}_j(s),\quad
y'_j=\beta_j y_j,\quad J=3
$$

$$
Z_{cat}=\big[\,y'_1\;\|\;y'_2\;\|\;y'_3\,\big]
$$

该模块采用**乘性加权**，$\beta_j\equiv1$ 时 $Z_{cat}$ 与基线**逐比特一致**（实测最大绝对误差 0.0），无需额外的融合方式设计。其消融模式包括 `static`（仅先验 $b_j$）、`adaptive`（仅自适应 $\phi$）与 `scalar`（一组与样本无关的可学习权重）。

### 3.4 复杂度与参数量

| 模块 | 参数量 | 相对基线 | 额外计算 |
|---|---|---|---|
| 模块 A（频段注意力） | $K+(2h+1)=5+17=30$ | $1.1\times10^{-5}$ | 一次 rFFT 与 $K$ 次 irFFT，可忽略 |
| 模块 B（多尺度注意力） | $J+J+(2h+1)=31$ | $1.2\times10^{-5}$ | 无（仅逐元素缩放） |

两者相对基线 2 680 350 个参数均可视为零代价。作为对照，若改为在高维特征上构造可学习注意力核（$F'^2$ 量级），参数量将高出 4–5 个数量级。

---

## 4 实验

### 4.1 数据集与预处理

**MEEG 数据集**：以音乐为情感诱发刺激的 EEG 情感数据集，32 名被试、32 通道（国际 10-20 系统），原始采样率 1000 Hz，标注采用效价与唤醒度两个维度。本文使用被试 0 的 Arousal 二分类标签：20 个试次 × 14 个片段 = 280 个样本，两类各 140 个（完全均衡）；**每个试次为单一类别**。

**预处理**：降采样至 200 Hz，1–50 Hz 带通滤波；每试次按 4 s 长、无重叠切分，得到每个片段 $32\times800$ 的输入；按通道做 z-score 标准化，均值与方差**仅由训练集统计**后应用于测试集。

### 4.2 实验设置与协议

**（1）被试内受控实验。**

| 项目 | 设置 |
|---|---|
| 折数 | 试次级 3 折交叉验证（同一试次的片段不跨训练/测试集） |
| 随机种子 | 15 个：$\{1,\dots,14,3407\}$，每配置 45 次独立训练 |
| 配对设计 | 同一随机种子下所有配置使用**相同的折划分与初始化** |
| 每折数据 | 训练 12 试次（168 片段）、验证 2–3 试次、测试 8 试次（112 片段） |
| 训练预算 | 固定 40 个 epoch（不使用早停，原因见下） |
| 优化 | batch 64，Adam，学习率 $1\times10^{-3}$，标签平滑 $\epsilon=0.1$ |
| 数值环境 | 关闭 TF32，启用 cuDNN 确定性内核 |

**不使用早停的原因**：验证集仅 28–42 个片段，实测验证准确率在第 1 个 epoch 即达 1.000，基于验证准确率的早停会在训练几乎未开始时终止（实测固定预算基线为 94.05%，而"最佳验证检查点"在某个种子上仅 42.86%）。

**（2）关于随机种子的说明。** 一个随机种子 $s$ 决定三件事：**折划分**、**权重初始化**与训练过程中的**批次顺序与 dropout 掩码**，因此"一个种子"等价于一次完整的重复实验。表 1 给出**同一模型、同一份数据**下仅改变种子的实测结果：15 个种子的准确率均值 87.22%、标准差 8.47%，**最小值 65.31%、最大值 95.86%，极差达 30.56 个百分点**。

**表 1** 同一模型在不同随机种子下的准确率（MEEG 被试 0，Arousal，按均值升序）

| 随机种子 | 逐折准确率（%） | 均值（%） |
|---|---|---|
| 6 | 45.9 / 57.1 / 92.9 | **65.31** |
| 13 | 76.5 / 94.9 / 58.3 | **76.59** |
| 3 | 91.8 / 75.5 / 67.9 | **78.40** |
| 8 | 78.6 / 75.5 / 97.6 | **83.90** |
| 14 | 87.8 / 71.4 / 94.0 | **84.41** |
| 1 | 86.7 / 74.5 / 95.2 | **85.49** |
| 9 | 87.8 / 70.4 / 100.0 | **86.05** |
| 5 | 96.9 / 94.9 / 69.0 | **86.96** |
| 7 | 91.8 / 85.7 / 97.6 | **91.72** |
| 2 | 98.0 / 86.7 / 94.0 | **92.91** |
| 11 | 88.8 / 98.0 / 96.4 | **94.39** |
| 12 | 93.9 / 100.0 / 91.7 | **95.18** |
| 10 | 96.9 / 94.9 / 94.0 | **95.29** |
| 4 | 92.9 / 95.9 / 98.8 | **95.86** |
| 3407 | 91.8 / 96.9 / 98.8 | **95.86** |
| — | 全部 15 个种子 | **87.22 ± 8.47** |

可见单次实验几乎不具判别力：同一方法既可能给出 65.31%，也可能给出 95.86%。因此本文以**跨种子均值与标准差**为主要指标，并同时报告**崩溃率**（准确率低于 0.70 的折占比）。对同一随机种子，所有配置使用相同的折划分与初始化，故比较为**配对比较** $\Delta_i=\mathrm{Acc}_{\text{变体}}(s_i)-\mathrm{Acc}_{\text{基线}}(s_i)$，并报告配对 $t$ 检验、Wilcoxon 符号秩检验与符号检验。

**（3）对比配置**：基线（AT-DGNN，直接复用归档结果，因 `--band-attn none` 已验证与其逐比特一致）、模块 A（`residual` 与 `replace` 两种融合）、模块 B（`both`/`static`/`adaptive`/`scalar`）。

### 4.3 主结果

**表 2** 主结果（3 折 × 15 种子，按种子配对；参数增量相对基线 2 680 350）。

| 配置 | 参数 | 准确率（跨种子, %） | 最差折 (%) | 崩溃率 | ΔACC (pp) | $p$（配对 $t$） |
|---|---|---|---|---|---|---|
| AT-DGNN（基线） | 2 680 350 | 87.22 ± 8.47 | 45.92 | 5/45 (11%) | — | — |
| 模块 A：频段注意力（**residual**） | +30 | 87.26 ± 8.65 | 42.86 | 6/45 (13%) | **+0.04 ± 0.54** | 0.776 |
| 模块 A：频段注意力（**replace**） | +30 | 81.62 ± 10.70 | 30.61 | **11/45 (24%)** | **−5.60 ± 5.92** | **0.0033** |
| 模块 B：多尺度注意力（both） | +31 | 87.00 ± 8.10 | 46.94 | 6/45 (13%) | −0.23 ± 0.76 | 0.285 |
| 模块 B：多尺度注意力（adaptive） | +31 | 87.34 ± 8.34 | 47.96 | 5/45 (11%) | +0.12 ± 0.50 | 0.395 |

**表 3** 配对比较的完整统计（$n=15$）。

| 对比 | ΔACC (pp) | 95% CI | 改善/变差/持平 | 崩溃率变化 | $p$（$t$） | $p$（Wilcoxon） | Cohen $d_z$ |
|---|---|---|---|---|---|---|---|
| 模块 A（residual）− 基线 | +0.04 ± 0.54 | [−0.24, +0.32] | 7/6/2 | +1 | 0.776 | 0.753 | 0.075 |
| 模块 A（replace）− 基线 | **−5.60 ± 5.92** | [−8.71, −2.50] | **1/14/0** | **+6** | **0.003** | **0.001** | −0.914 |
| 模块 B（both）− 基线 | −0.23 ± 0.76 | [−0.63, +0.17] | 5/7/3 | +1 | 0.285 | 0.182 | −0.287 |
| 模块 B（adaptive）− 基线 | +0.12 ± 0.50 | [−0.14, +0.38] | 6/5/4 | ±0 | 0.395 | 0.387 | 0.240 |

### 4.4 消融实验

**表 4** 两种模块的消融（均为 residual 融合，$n=15$）。

| 模块 | 静态先验 | 自适应调制 | ΔACC (pp) | $p$ | 说明 |
|---|---|---|---|---|---|
| 模块 A | ✓ | ✓ | +0.04 ± 0.54 | 0.776 | 默认配置 |
| 模块 A | ✓ | ✗ | −0.15 ± 0.55 | 0.319 | 仅"频段总体上是否重要" |
| 模块 A | ✗ | ✓ | −0.05 ± 0.44 | 0.705 | 仅逐样本重新分配 |
| 模块 B | ✓ | ✓ | −0.23 ± 0.76 | 0.285 | 默认配置 |
| 模块 B | ✓ | ✗ | −0.29 ± 0.62 | 0.098 | 仅尺度先验 |
| 模块 B | ✗ | ✓ | **+0.12 ± 0.50** | 0.395 | 仅逐样本调制（最好变体） |
| 模块 B | — | — | −0.29 ± 0.60 | 0.086 | scalar：一组与样本无关的权重 |

**所有变体与基线的差异均不显著（$p$ 均大于 0.05）**，且效应量极小（$|d_z|\le0.29$）。按观测效应量计算，检出模块 B 默认配置的效应需约 162 个种子，检出模块 A 的效应需约 2 373 个种子，即实测效应在统计上不可区分于零。

### 4.5 权重可解释性与机制分析

**表 5** 模块 A 学到的频段权重 $\beta_k$（均值归一化为 1；单模型，训练准确率已收敛）。

| 频段 | 频率 (Hz) | $\beta_k$ 均值 | 标准差 | 高唤醒度 | 低唤醒度 | 试次级 $p$ |
|---|---|---|---|---|---|---|
| δ | 1–4 | 1.0124 | 0.0007 | 1.0125 | 1.0123 | 0.086 |
| θ | 4–8 | 1.0051 | 0.0004 | 1.0050 | 1.0052 | 0.133 |
| α | 8–13 | 0.9982 | 0.0005 | 0.9981 | 0.9983 | 0.051 |
| β | 13–30 | 0.9943 | 0.0003 | 0.9944 | 0.9942 | 0.051 |
| γ | 30–50 | 0.9900 | 0.0003 | 0.9899 | 0.9900 | 0.428 |

- **最大偏离均匀 1.34%，归一化熵 1.0000**（完全均匀）；
- 可学习的静态先验 $b_k$ 呈轻微的**单调低频偏好**：$[+0.0099,+0.0036,-0.0037,-0.0068,-0.0114]$（δ 最高、γ 最低），但幅度极小；
- **没有任何频段的类间差异通过 Bonferroni 校正**（阈值 $p<0.01$）。

**表 6** 模块 B 学到的尺度权重 $\beta_j$。

| 分支 | 核长（采样点） | 近似低频截止 | $\beta_j$ 均值 | 标准差 |
|---|---|---|---|---|
| 1 | 100 | ~2 Hz | 1.0144 | 0.0003 |
| 2 | 50 | ~4 Hz | 0.9912 | 0.0001 |
| 3 | 25 | ~8 Hz | 0.9944 | 0.0003 |

最大偏离 1.53%、归一化熵 1.0000；静态先验为 $[+0.0157,-0.0075,-0.0033]$，同样呈现**轻微的低频（长核）偏好**。

**机制解释**：两个模块学到的权重都收敛到近似均匀，与 4.3 节的中性性能结果完全自洽——**模型自发地选择了"等权"，即退回基线**。这说明在该被试、该片段长度（4 s）的设定下，规范频段之间的重要性差异不足以支撑可学习的加权增益。

**统计层级的重要提示（伪重复）**：在模块 A 的频段类间检验中，**片段级检验认为 3/5 个频段"显著"，而有效的试次级检验为 0/5**。原因是 MEEG 的标签为试次级——一个试次内的 14 个 4 s 片段共享同一标签且高度相关，片段级检验把 140 个相关片段当作 140 个独立样本，使有效样本量被放大约 14 倍，产生虚假的极小 $p$ 值。本文统一采用试次级检验（$n=10$ vs $10$）并做 Bonferroni 校正。

### 4.6 跨被试（LOSO）实验

为检验上述结论是否可迁移到未见过的被试，本文实现了留一被试（LOSO）协议：每折以 1 名被试为测试集、其余被试为训练集，每名被试使用自身统计量做逐通道标准化（不使用标签），统计单位为**被试**（$n$ = 被试数）。该协议需要 MEEG 全部 32 名被试的预处理数据；本文当前仅获得被试 0，**该实验尚未完成**，代码与协议已就绪（每折在 31 名被试、约 8 680 个片段上训练 40 个 epoch，估计在 RTX 4060 上约 16 小时/变体）。

---

## 5 讨论

### 5.1 为什么显式频段加权没有带来增益

本文的中性结果有三个相互印证的层次：

1. **表示层面**：模块 A 的加权融合在数学上是一个**线性滤波操作**（输入的频段成分线性重组），而这一能力**基线第一层卷积本身即可学习**。因此模块 A 在功能上与基线高度冗余，模型最优策略是保持恒等。
2. **参数层面**：模块仅 30–31 个参数，相对基线 2.68 M 几乎为零，因此也不可能通过增加容量带来收益（这与"涨点来自参数量"的常见解释相反）。
3. **机制层面**：实测权重收敛到均匀（熵 1.0000），且类间差异在正确的统计层级上不显著——**数据本身不提供可利用的频段重要性差异**。

值得强调的是：**"无效"并不等于"无风险"**。模块 A 在替换式融合下显著劣化 5.6 个百分点，说明不加约束地插入加权模块可能带来实质损害。

### 5.2 设计准则：融合方式必须保证恒等性

本文最有实用价值的结论是一条可直接迁移的设计准则：

> **向已有模型插入加权（或门控）模块时，必须保证"权重取中性值时模块精确退化为原模型"。**

其理由在本文中得到了定量验证：同一模块、同样的参数量、同样的训练协议，仅因融合方式不同——

| 融合方式 | 中性权重下是否恒等 | ΔACC | $p$ | 崩溃率 |
|---|---|---|---|---|
| 替换 $x'=\sum_k\beta_kX_k$ | ❌ 不精确（最大偏差 $2.0\times10^{-3}$） | **−5.60 pp** | 0.003 | 11% → 24% |
| 残差 $x'=x+\sum_k(\beta_k-1)X_k$ | ✅ **逐比特** | +0.04 pp | 0.78 | 11% → 13% |

此外，把打分网络的末层**零初始化**（使模块在初始化时即为恒等）是该准则的实现要点之一。本文建议此类模块在论文中显式报告"中性权重下的等价性验证结果"。

### 5.3 对实验方法的启示

1. **单次实验缺乏判别力**：同一模型在不同随机种子上的准确率极差达 30.56 个百分点，因此报告单次结果或仅用少量种子（如 $n=5$）容易得出偏乐观的结论。本文建议报告多种子配对差值与符号检验，并给出功效分析或所需样本量。
2. **必须区分"试次"与"片段"两个统计层级**：片段级检验在本数据上会把 3/5 个频段判为"显著"，而试次级检验为 0/5。建议明确报告检验所在层级，并在试次层级上做检验与多重比较校正。
3. **"权重热图"不能代替机制验证**：注意力权重可能均匀到几乎不含信息（本文归一化熵均为 1.0000），建议至少报告**偏离幅度、归一化熵与跨种子一致性**三项量化指标。

### 5.4 局限

1. **单被试**（MEEG 被试 0）、单数据集、单标签（Arousal）、被试内协议；跨被试 LOSO 尚未完成（§4.6）。
2. **片段长度固定为 4 s**。频段重要性可能随片段长度变化，更长或更短的片段下结论可能不同。
3. **频段划分固定**（经典五频段），未探索可学习滤波器组或更细的频率分辨率。
4. **种子的重复不等同于被试的重复**：15 个种子共享同一名被试的同一份数据，各次结果并不独立；本文据此把跨被试实验列为主要待补证据。
5. **未做超参数搜索**：所有配置共用基线超参数（学习率 $10^{-3}$、40 epoch），这是为保证配对公平，但也可能低估模块潜力。

---

## 6 结论

本文以 AT-DGNN 为基线，检验了"显式频段加权能否提升 EEG 情感识别性能"这一假设，并得到如下结论：

1. **H1 未被支持**。在保证模块恒等性的前提下，输入侧频段注意力（+30 参数）与分支侧多尺度注意力（+31 参数）及其全部消融变体，均与基线无显著差异（最好变体 +0.12 个百分点，$p=0.40$）。
2. **模型自发选择了均匀加权**。两个模块学到的权重归一化熵均为 1.0000、最大偏离仅 1.34% 与 1.53%，且类间差异在试次级检验中均不显著——说明该数据上频段重要性的差异不足以提供可利用的增益。
3. **融合方式的影响远大于模块本身**。同一模块在替换式融合下显著劣化 5.60 个百分点（$p=0.0033$，崩溃率由 11% 翻倍至 24%），改为残差式融合后与基线无差异。据此本文提出一条可迁移的设计准则：**插入加权模块时必须保证中性权重下的精确恒等性**，并通过残差融合与打分头零初始化予以实现。
4. **实验方法层面**，本文用实测数据说明：多随机种子重复是必要的（极差 30.56 个百分点），且统计检验必须区分试次与片段两个层级（片段级 3/5"显著" vs 试次级 0/5）。

本文的价值不在于报告性能提升，而在于以受控实验给出**否定的、可核查的**结论，并把"如何向已有模型安全地插入加权模块"这一工程问题转化为一条有定量依据的设计准则。后续工作包括：在全部 32 名被试上完成跨被试 LOSO 验证；探索可学习的频段边界或滤波器组；以及把频段加权与电极维加权联合建模。

---

## 参考文献

1. Xiao M, Zhu Z, Xie K, et al. MEEG and AT-DGNN: Improving EEG emotion recognition with music introducing and graph-based learning[C]//2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2024: 4201–4208.
2. Zheng W L, Lu B L. Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks[J]. IEEE Transactions on Autonomous Mental Development, 2015, 7(3): 162–175.
3. Zheng W L, Liu W, Lu Y, et al. EmotionMeter: A multimodal framework for recognizing human emotions[J]. IEEE Transactions on Cybernetics, 2018, 49(3): 1110–1122.
4. Ding Y, Robinson N, Tong C, et al. LGGNet: Learning from local-global-graph representations for brain–computer interface[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.
5. Song T, Zheng W, Song P, et al. EEG emotion recognition using dynamical graph convolutional neural networks[J]. IEEE Transactions on Affective Computing, 2018, 11(3): 532–541.
6. Zhang S, Chu C, Zhang X, et al. EEG emotion recognition using AttGraph: A multi-dimensional attention-based dynamic graph convolutional network[J]. Brain Sciences, 2025, 15(6): 615.
7. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. 2017: 5998–6008.
8. Ba J L, Kiros J R, Hinton G E. Layer normalization[J]. arXiv preprint arXiv:1607.06450, 2016.
9. Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of Neural Engineering, 2018, 15(5): 056013.
10. Ding Y, Robinson N, Zeng Q, et al. TSception: A deep learning framework for emotion detection using EEG[C]//2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020: 1–7.
11. Zhong P, Wang D, Miao C. EEG-based emotion recognition using regularized graph neural networks[J]. IEEE Transactions on Affective Computing, 2020, 13(3): 1290–1301.
12. Bouthillier X, Delaunay P, Bronzi M, et al. Accounting for variance in machine learning benchmarks[J]. Proceedings of Machine Learning and Systems, 2021, 3: 747–769.

---

## Abstract

**Objective**: Frequency content is central to EEG-based emotion recognition, yet many deep models do not weight frequency bands explicitly. Taking AT-DGNN (BIBM 2024) as the baseline, this paper tests the natural hypothesis that adding an explicit, sample-adaptive frequency-band weighting improves recognition accuracy.

**Methods**: A code-level analysis shows that AT-DGNN's Tception temporal learner is a learnable filter bank followed by a log-power layer (a learned spectral decomposition) whose frequency weighting is static, not aligned with canonical EEG bands, and never re-weighted, because the three branches are merely concatenated. Two explicit modules are therefore proposed: (i) an **input-side band attention** that splits the signal into the five canonical bands with complementary brick-wall masks and weights them using a learnable static prior plus a shared sample-adaptive modulation; and (ii) a **branch-side multi-scale attention** that weights the three Tception branches. Both sit on top of the **unchanged** AT-DGNN backbone. The key design rule is **identity preservation**: with residual fusion and a zero-initialised score head, each module is bit-exactly the baseline at initialisation, so any change is attributable to what it learns.

**Results**: With trial-wise 3-fold cross validation over 15 random seeds (45 independent runs per configuration, paired by seed) on the MEEG dataset: (1) when the fusion preserves identity, both modules are statistically indistinguishable from the baseline (band attention $\Delta=+0.04$ pp, $p=0.78$; best multi-scale variant $+0.12$ pp, $p=0.40$) at a cost of 30–31 parameters; (2) the **fusion scheme matters far more than the module**: using replace fusion instead of residual fusion, the *same* band attention degrades accuracy by **5.60 pp ($p=0.0033$, 14/15 seeds worse, collapse rate 11% → 24%)**, a 5.6 pp swing; (3) the learned weights converge to near-uniform in both modules (normalised entropy 1.0000, maximum deviation 1.34% and 1.53%), and no band-level class difference survives trial-level Bonferroni correction, which explains the null result.

**Conclusion**: Under the controlled single-subject setting, explicit frequency-band weighting did not improve AT-DGNN; the model spontaneously selected near-uniform weighting. The most reusable outcome is a design rule: **when inserting a weighting module into an existing model, the fusion must be exactly identity at neutral weights** — an inexact identity path caused a significant multi-percent degradation, dwarfing any potential benefit of the module. The seed protocol and the trial-vs-segment analysis further show that single runs and segment-level significance tests can both be misleading.

**Key words**: EEG; emotion recognition; band attention; graph neural network; fusion design; random seeds
