# 基于频段注意力的 AT-DGNN 脑电情感识别方法

**Band Attention Enhanced AT-DGNN for EEG Emotion Recognition**

> **稿件状态说明（写作中，勿作为终稿引用）**
> - 本稿的方法、立论、实验设计、创新点**已定稿**（依据全部来自代码分析与已验证的实现）。
> - §4.3–§4.6 的**性能数字与可解释性数字为待回填项**，以 `【待回填】` 标注——数据来自正在运行的 15 种子配对实验（`experiments/results/bandattn_seed*.json`）。
> - 跨被试验证（§4.6）需 MEEG 32 被试数据；脚本已就绪，数据到位后可跑（约 2 天 GPU）。
> - **若 pilot 结果为中性或负向，§4.3–§4.5 的结论段必须据实改写**（见文末"作者注"）。

---

## 摘要

**目的**：脑电（EEG）情感识别中，频段信息（δ/θ/α/β/γ）对情感状态的判别具有明确的神经生理依据，但现有深度模型对"哪些频段更重要"的建模往往是不充分的。本文针对 AT-DGNN（BIBM 2024）这一代表性模型，指出其时域前端虽然实现了频谱分解，却**未对频段进行加权**，进而提出一种频段注意力模块，并对其进行被试内与跨被试评估。

**方法**：首先通过代码级分析指出，AT-DGNN 的 Tception 时域学习器由可学习一维卷积与功率层 $y=\log(\overline{x^2})$ 构成，本质上是一组**可学习滤波器组 + 对数功率**，即一种学习得到的频谱分解；但其频段加权存在三个不足：（i）**静态**（全部样本共享同一组权重）；（ii）**未与规范 EEG 频段对齐**（每个卷积核是任意学习到的频谱响应）；（iii）各分支输出仅被**拼接**（`concat`）而从未被重新加权。为此，本文提出频段注意力模块：用固定的砖墙式频段分解把输入划分为 5 个规范频段，为每个频段计算对数功率描述子，并以"可学习静态频段先验 + 共享网络的逐样本自适应调制"生成注意力权重，对 5 个频段加权求和后送入**未作改动**的 AT-DGNN 主干。

**结果**：该模块仅增加 **30** 个参数（约占基线的 $1.1\times10^{-5}$），且由于频段分解构成对 1–50 Hz 的**单位分解**，当权重取等值时模块**近似还原原始输入**，因此不改变模型的基本行为。在 MEEG 数据集上的被试内受控实验（试次级 3 折交叉验证 × 15 个随机种子，按种子配对）中，【待回填】；消融显示静态频段先验与逐样本自适应两部分的贡献分别为【待回填】；学到的频段权重与高/低唤醒度的对应关系为【待回填】。跨被试（LOSO）验证结果为【待回填，需 32 被试数据】。

**结论**：【待回填】。本文的主要贡献是把 AT-DGNN 中**静态、隐式、未对齐**的频谱加权改造为**显式、样本自适应、可解释**的频段加权，且以极低的参数代价实现，为 EEG 情感识别的频段建模提供了一条可直接复用的技术路线。

**关键词**：脑电；情感识别；频段注意力；图神经网络；跨被试；深度特征融合

**中图分类号**：TP183　　**文献标志码**：A

---

## 1 引言

脑电（Electroencephalogram, EEG）信号直接记录大脑皮层神经活动，具有时间分辨率高、不易伪装、能反映个体内在情感状态等优点，已成为情感计算与情感脑机接口领域的重要信号来源。在基于深度学习的 EEG 情感识别方法中，"如何构造电极之间的连接关系"以及"如何从多频段信号中提取判别特征"是两个核心问题。

**（1）频段信息的神经生理依据。** EEG 的节律活动按频率划分为 δ（1–4 Hz）、θ（4–8 Hz）、α（8–13 Hz）、β（13–30 Hz）、γ（30–50 Hz）五个经典频段。情感神经科学的研究表明，不同频段与情感状态存在系统性的关联：α 波与放松状态相关，β 波与紧张、警觉等高唤醒状态相关，额叶 α 波的不对称性与效价维度相关。正因如此，基于频段的微分熵（differential entropy, DE）、功率谱密度（PSD）等特征长期是 EEG 情感识别中性能最强的手工特征之一，也是 SEED 等公开数据集上的主流表示。

**（2）现有方法对频段加权的处理并不充分。** 以 AT-DGNN（BIBM 2024）为例，其提出的 Tception 时域学习器由三个不同核长的并行分支组成，每个分支为

$$
\textbf{Tception}_k(x)=\mathrm{PowerLayer}\!\left(\mathrm{Conv2d}_{(1,\ell_k)}(x)\right),\qquad
\mathrm{PowerLayer}(z)=\log\!\big(\mathrm{AvgPool}(z^2)\big)
$$

其中 $\ell_k\in\{100,50,25\}$ 为卷积核长度（对应 0.5/0.25/0.125 s）。该结构在功能上等价于**一组可学习滤波器 + 对数功率**，即一种学习得到的频谱分解。然而，本文通过代码级分析指出其在频段加权方面存在三点不足：

1. **静态加权**：滤波器的频率响应由训练好的卷积核决定，对**所有样本完全相同**，模型无法表达"对当前这段信号而言，β 频段更具判别力"这类逐样本的频段重要性。
2. **未与规范频段对齐**：每个输出通道是任意学习到的频谱响应，并不对应 δ/θ/α/β/γ 中的某一个，因而其"频段重要性"无法与神经生理学结论对照，可解释性受限。
3. **仅拼接而未加权**：三个分支的输出仅通过 `torch.cat(·, dim=-1)` 沿时间维拼接，既无权重、也无通道间的竞争或选择机制。后续的特征整合与滑动窗口处理均为固定参数的卷积与池化操作。

上述问题可以概括为：**AT-DGNN 具备频谱分解能力，但缺少一个显式的、逐样本自适应的频段加权机制。**

**（3）本文工作。** 针对上述问题，本文提出**频段注意力模块（Band Attention, BA）**，其核心思想是：把频段加权从"隐含在卷积核里"改造为"显式的注意力权重"，并使其同时具备样本自适应性与可解释性。模块包含三个部分：

1. **固定频段分解**：用砖墙式频域掩码把输入划分为 5 个规范频段，掩码互为补集，构成对 1–50 Hz 的单位分解（§3.3.1）；
2. **频段重要性打分**：为每个频段计算对数功率描述子，并以"可学习静态先验 + 共享网络的自适应调制"生成分数（§3.3.2）；
3. **加权融合**：对 5 个频段加权求和，权重经归一化后均值为 1，随后送入**未作任何改动**的 AT-DGNN 主干。

**（4）主要创新点。**

1. **提出频段注意力模块，将 AT-DGNN 的频谱加权显式化、样本自适应化、可解释化。** 与基线"静态、隐式、未对齐"的频谱加权相比，本文的权重是逐样本生成的、直接对应 5 个规范频段、并可读取与对照神经生理结论。
2. **模块代价极低且具备"零风险"性质。** 仅增加 30 个参数（基线 2 680 350 个参数的 $1.1\times10^{-5}$）；由于频段分解构成单位分解，权重取等值时模块近似还原原始输入，因此**不改变模型的基本行为**，可视为一个安全的增量模块（§3.3.3、§3.4）。
3. **以被试内受控实验与跨被试 LOSO 两级协议评估模块**，并给出频段权重的可解释性分析（§4）。

---

## 2 相关工作

### 2.1 EEG 情感识别中的频段特征

早期与主流方法多采用手工频段特征：对每个电极、每个频段计算微分熵（DE）或功率谱密度（PSD），再送入传统分类器或浅层网络。DE 在固定频段内的功率近似服从高斯分布，与对数功率高度相关，因而兼具判别力与计算简便性。多个公开数据集（SEED、SEED-IV、DEAP）上的强基线均以 DE 特征为输入。这类方法的共同前提是**频段划分是事先固定的**，且各频段的融合方式（通常为简单拼接）并未被学习。

### 2.2 EEG 图神经网络

图神经网络通过建模电极间的交互关系提升性能。按图构造方式可分为四类：**固定图**（由电极物理距离或脑区解剖归属定义）、**静态可学习图**（以可学习参数矩阵作为邻接矩阵，对所有样本共享）、**动态不可学习图**（以节点特征自相似度生成逐样本图，如 AT-DGNN 使用的 $\hat{A}=D^{-1/2}(XX^{\top}+I)D^{-1/2}$）、以及**动态可学习图**（如以 $\mathrm{softmax}(HAH^{\top})$ 生成邻接）。本文工作位于**特征侧**而非图构造侧，与上述路线互补。

### 2.3 注意力机制与特征维加权

注意力机制通过对特征维或空间位置生成归一化权重实现特征选择。在 EEG 领域，注意力被用于通道选择（电极级加权）、时间片段选择与特征维加权。需要指出的是，注意力权重的"有效性"与**被加权维度的大小**密切相关：softmax 归一化使得每个元素在数值上被约束为均值的 $1/N$ 量级，当 $N$ 很大时（例如上千个时间位置），权重难以产生有效的选择性；而当 $N$ 较小（例如本文的 5 个频段）时，权重具备充足的调节余量。频段轴恰好属于后者，这是本文选择在频段维而非时间维引入注意力的直接原因。

---

## 3 方法

### 3.1 总体框架

本文方法的整体流程为：原始 EEG 输入 $\to$ **频段注意力模块（本文新增）** $\to$ **AT-DGNN 主干（未作改动）** $\to$ 情感分类。

```
EEG 输入 (B,1,32,800)
   ↓
【本文新增】固定频段分解 → 5 个规范频段 (B,5,32,800)
   ↓
【本文新增】频段重要性打分 → β ∈ R^{B×5}（均值归一化为 1）
   ↓
【本文新增】加权融合 x' = Σ_k β_k x_k  → (B,1,32,800)
   ↓
【未改动】AT-DGNN 主干：
   Tception 多尺度时域学习器 → 特征整合 → 滑窗多头注意力+TCN
   → 局部滤波 → 脑区聚合 → 动态图卷积 ×3 → 全连接 → 分类
```

**设计原则**：模块只作用于**输入侧**，主干的所有结构与超参数保持不变，从而使性能变化可归因于频段加权本身。

### 3.2 基线：AT-DGNN 主干

设输入为 $x\in\mathbb{R}^{B\times1\times C\times T}$（$B$ 为批大小，$C=32$ 为电极数，$T=800$ 为 4 s 片段在 200 Hz 下的采样点数）。

**时域前端。** 三个并行分支使用核长 $\ell_k\in\{100,50,25\}$ 的一维卷积（其低频截止约 $f_s/\ell_k\approx 2/4/8$ Hz），每支后接功率层，输出沿时间维拼接：

$$
Z_{cat}=\big[\,\mathrm{PowerLayer}(\mathrm{Conv}_{(1,\ell_1)}(x))\;\|\;\mathrm{PowerLayer}(\mathrm{Conv}_{(1,\ell_2)}(x))\;\|\;\mathrm{PowerLayer}(\mathrm{Conv}_{(1,\ell_3)}(x))\,\big]
$$

**特征整合与滑窗处理。** $Z_{cat}$ 经 $1$ 维卷积降维后，以窗长 100、步长 20 切分，逐窗执行 8 头自注意力、残差与层归一化，再接时域卷积块，最后沿窗口维堆叠融合。

**空间建模。** 经逐通道局部滤波与脑区聚合得到 $N=14$ 个节点的表示，再经 3 层动态图卷积（邻接矩阵由节点特征自相似度生成）与全连接层输出分类结果。

**关键观察。** 如 §1 所述，上述 Tception 在功能上等价于**可学习滤波器组 + 对数功率**，即频谱分解；但其频段加权是**静态、隐式、且未与规范频段对齐**的，且分支输出仅被拼接。这正是本文要改进的对象。

### 3.3 频段注意力模块

#### 3.3.1 固定频段分解

采用砖墙式频域掩码将输入划分为 $K=5$ 个规范频段：

$$
X_k=\mathrm{irFFT}\big(\mathrm{rFFT}(x)\odot M_k\big),\qquad
M_k(f)=\begin{cases}1,& f\in[f_k^{lo},f_k^{hi})\\0,&\text{otherwise}\end{cases}
$$

其中频段区间为

$$
[f_k^{lo},f_k^{hi})\in\{(1,4),(4,8),(8,13),(13,30),(30,50)\}\ \text{Hz}
$$

**单位分解性质。** 由于掩码互为补集（$\sum_k M_k(f)=1$ 对 $f\in[1,50)$ 成立），有

$$
\sum_{k=1}^{K}X_k=\mathrm{irFFT}\big(\mathrm{rFFT}(x)\odot\textstyle\sum_k M_k\big)=x\ \text{在 }1\!-\!50\ \mathrm{Hz}\ \text{内的成分}
$$

即**当频段权重取等值时，加权融合的输出近似等于原始输入**（仅在 1–50 Hz 预处理之外的残余上存在差异）。该性质使模块成为一个不破坏原模型行为的"安全"增量。实测：真实 EEG 片段上 5 个频段的**带内能量占比均为 100.0%**（无频段间泄漏、无带内失真），等权重构的 RMS 相对误差为 0.1399，与预处理的带外残余能量（0–1 Hz 占 1.40%、50–100 Hz 占 0.64%，$\sqrt{0.0204}\approx0.14$）一致。

> **实现说明**：本文亦实现了基于线性相位 FIR 的频段分解（$M$ 由低通差分构造）作为对照。由于 Hann 窗 FIR 的过渡带宽度约为 $4f_s/L$（$L=129$ 时约 6 Hz），相邻频段重叠严重（实测各频段带内能量占比仅 86%–98%），会污染频段权重的解释，故本文采用频域掩码实现。

#### 3.3.2 频段重要性打分与加权

对每个频段计算对数功率描述子：

$$
d_k=\log\!\left(\frac{1}{C\,T}\sum_{c,t}\big(X_k\big)_{c,t}^{2}+\epsilon\right)\in\mathbb{R}
$$

为避免静态先验被描述子的整体偏移吸收，在频段维做中心化 $\tilde d_k=d_k-\frac1K\sum_j d_j$，随后生成频段分数：

$$
s_k=\underbrace{b_k}_{\text{静态频段先验}}+\underbrace{\phi(\tilde d_k)}_{\text{逐样本自适应}},\qquad
\phi:\mathbb{R}\to\mathbb{R}\ \text{在所有频段间共享权重}
$$

$$
\beta_k=K\cdot\mathrm{softmax}_k(s),\qquad
x'=\sum_{k=1}^{K}\beta_k\,X_k
$$

其中 $b_k$ 为可学习参数，刻画"频段 $k$ 在总体上是否重要"（**静态先验**）；$\phi$ 为一个两层的共享映射（$\mathbb{R}\to\mathbb{R}^{h}\to\mathbb{R}$，$h=8$），刻画"依据当前样本的频段功率分布，应当如何重新分配权重"（**样本自适应调制**）。softmax 乘以 $K$ 使权重均值为 1，从而保持特征的整体尺度，并与 §3.3.1 的单位分解性质配合，保证等权时退化为原始输入。

**可解释性。** $\beta\in\mathbb{R}^{B\times K}$ 具有两个便于解释的性质：（i）维度低（$K=5$）且语义明确（对应 5 个规范频段）；（ii）归一化后可直接比较不同频段的相对重要性。因此可统计 $\beta$ 的均值与类别条件均值，得到"哪些频段对高/低唤醒度更具判别力"，并与神经生理学结论对照。

**消融设计。** 模块的两个部分可独立开关：`static`（仅保留 $b_k$）、`adaptive`（仅保留 $\phi$）、`both`（两者相加，默认）。该设计用于定位性能变化的具体来源。

#### 3.3.3 复杂度与参数量分析

**参数量**：$K$ 个先验参数 $b_k$，加上 $\phi$ 的两层权重：$1\times h+h+h\times1+1=2h+1$。取 $K=5,h=8$，共

$$
5+(2\times8+1)=30
$$

即基线 2 680 350 个参数的约 $1.1\times10^{-5}$。作为对照，若改为对高维特征构造可学习注意力核（如 $F'^2$ 量级），参数与计算开销将高出若干数量级。

**计算量**：频段分解为一次 $\mathrm{rFFT}$（长度 $T$）与 $K$ 次 $\mathrm{irFFT}$，复杂度 $O(KT\log T)$；打分与加权为 $O(KCT)$。相对于主干的卷积开销（约 $10^{10}$ 量级乘加），可忽略。

### 3.4 与基线的等价性与可归因性

**等价性**：当权重取等值（$\beta_k\equiv1$）时，由 §3.3.1 的单位分解性质，模块输出近似为原始输入，此时网络退化为 AT-DGNN 基线。实现上，本文提供 `band_attn='none'` 开关，实测其前向输出与基线**逐比特一致**（最大绝对误差 $0.0$）。该性质保证了：（i）实验结果的差异可归因于频段加权本身；（ii）模块不会因实现细节而意外劣化基线。

**可归因性**：模块只作用于输入侧，主干结构、超参数与训练流程完全不变；消融实验进一步把模块拆为静态先验与自适应调制两部分。

---

## 4 实验

### 4.1 数据集与预处理

**MEEG 数据集**：以音乐为情感诱发刺激的 EEG 情感数据集，包含 32 名被试、32 通道（国际 10-20 系统）记录，原始采样率 1000 Hz，情绪标注采用效价（Valence）与唤醒度（Arousal）两个维度。

本文使用 Arousal 二分类标签。预处理沿用基线流程：降采样至 200 Hz，1–50 Hz 带通滤波；每个试次按 4 s 长、无重叠的方式切分为片段，得到每个片段 $32\times800$ 的输入；按通道进行 z-score 标准化，标准化所用的均值与方差**仅由训练集统计得到**并应用于测试集，以避免信息泄漏。

**样本构成**：被试 0 包含 20 个试次 × 14 个片段 = 280 个样本，两类各 140 个（完全均衡）；**每个试次为单一类别**（该性质对 §4.2 的统计检验层级至关重要，见 §4.5）。

### 4.2 实验设置

本文采用**被试内**与**跨被试**两级协议。

**（1）被试内受控实验（用于模块有效性评估）。**

| 项目 | 设置 |
|---|---|
| 折数 | 试次级 3 折交叉验证（同一试次的片段不跨训练/测试集） |
| 随机种子 | 15 个：$\{1,\dots,14,3407\}$，每个配置 45 次独立训练 |
| 配对设计 | 同一随机种子下所有配置使用**相同的折划分与初始化**，配置间按种子配对 |
| 每折数据 | 训练 12 试次（168 片段）、验证 2–3 试次、测试 8 试次（112 片段） |
| 训练预算 | 固定 40 个 epoch（不使用早停，原因见下） |
| 优化 | batch 64，Adam，学习率 $1\times10^{-3}$，标签平滑 $\epsilon=0.1$ |
| 数值环境 | 关闭 TF32，启用 cuDNN 确定性内核 |

**不使用早停的原因**：验证集仅 28–42 个片段，实测验证准确率在第 1 个 epoch 即达 1.000，基于验证准确率的早停会在训练几乎未开始时终止（实测固定预算基线为 94.05%，而"最佳验证检查点"在某个种子上仅 42.86%）。因此改为固定预算并报告最终模型。

**对比配置**：基线（AT-DGNN，直接复用归档结果，因 `band_attn='none'` 已验证与基线逐比特一致）、频段注意力（`both`）、静态先验（`static`）、自适应调制（`adaptive`），共 4 个配置。

**（2）跨被试 LOSO 协议（用于泛化能力评估）。**

| 项目 | 设置 |
|---|---|
| 划分 | 留一被试：每折以 1 名被试为测试集，其余 31 名被试为训练集；共 32 折 |
| 归一化 | 每名被试使用**自身**统计量做逐通道 z-score（不使用标签，无泄漏） |
| 训练预算 | 固定 40 个 epoch；从训练被试中留出 2 名作为验证集（仅用于次要指标） |
| 统计单位 | **被试**（$n=32$），配置间按被试配对检验 |
| 评估指标 | 准确率、宏平均 F1、最差被试、崩溃率（准确率低于阈值 0.60 的被试占比） |

**两级协议的意义**：被试内协议隔离"模块本身是否有效"，跨被试协议回答"该有效性是否可迁移到未见过的被试"。后者是情感识别走向实用的关键，也是现有方法报道差异最大的环节。

### 4.3 被试内实验结果

> 【待回填】数据来源：`experiments/results/bandattn_seed*.json`（3 折 × 15 种子，与归档基线按种子配对）

**表 1** 被试内结果（MEEG 被试 0，Arousal）。"跨种子"为先取 3 折均值再在 15 个种子上统计（$n=15$）。

| 配置 | 参数量 | 准确率（跨种子, %） | 宏平均 F1（%） | 最差折（%） | 崩溃率 | ΔACC (pp) | $p$ |
|---|---|---|---|---|---|---|---|
| AT-DGNN（基线） | 2 680 350 | 87.22 ± 8.47 | — | 45.92 | 5/45 (11%) | — | — |
| **AT-DGNN + 频段注意力（both）** | 2 680 380 (+30) | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 |
| 仅静态先验（static） | 2 680 380 (+30) | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 |
| 仅自适应调制（adaptive） | 2 680 380 (+30) | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 |

**结果分析**：【待回填】需给出（i）配对差值及其 95% 置信区间；（ii）配对 $t$ 检验、Wilcoxon 符号秩检验与符号检验结果；（iii）功效分析（按观测效应量给出达到 80% 功效所需种子数）。

### 4.4 消融实验

> 【待回填】

**表 2** 静态先验与自适应调制的贡献分离。

| 模块构成 | 静态先验 $b_k$ | 自适应 $\phi$ | ΔACC (pp) | 说明 |
|---|---|---|---|---|
| 基线 | ✗ | ✗ | — | — |
| static | ✓ | ✗ | 【待回填】 | 仅"频段总体上是否重要" |
| adaptive | ✗ | ✓ | 【待回填】 | 仅"依据当前样本重新分配权重" |
| both | ✓ | ✓ | 【待回填】 | 默认配置 |

### 4.5 频段权重的可解释性分析

> 【待回填】

**表 3** 学到的频段权重 $\beta_k$（均值归一化为 1）与类别条件差异。

| 频段 | 频率范围 (Hz) | $\beta_k$ 均值 | 标准差 | 高唤醒度 | 低唤醒度 | 差值 |
|---|---|---|---|---|---|---|
| δ | 1–4 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 |
| θ | 4–8 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 |
| α | 8–13 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 |
| β | 13–30 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 |
| γ | 30–50 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 |

**分析要点**：
1. **权重的非均匀程度**：报告最大偏离与归一化熵（1.0 表示完全均匀）。若权重显著偏离均匀，说明模块确实学到了频段选择；若接近均匀，则说明该被试上频段重要性差异有限——两种结果都需如实报告。
2. **与神经生理学结论的对照**：重点检验 β/γ 频段在**高唤醒度**下是否获得更高权重（已有研究表明 β 波与紧张、警觉等高唤醒状态相关）。
3. **静态先验 $b_k$ 与自适应部分 $\phi$ 的分工**：$b_k$ 反映跨样本的总体频段偏好，$\phi$ 反映样本间的调制幅度。

> **统计层级提示（重要）**：MEEG 的标签是**试次级**的——一个试次内 14 个 4 s 片段共享同一标签且高度相关。因此对 $\beta$ 做类别比较时，**必须按试次聚合后再检验**，而不能把 140 个片段当作独立样本（否则有效样本量被放大约 14 倍，产生虚假的极小 $p$ 值）。本文统一采用试次级检验（$n=10$ vs $10$），并对多重比较做 Bonferroni 校正。

### 4.6 跨被试（LOSO）实验结果

> 【待回填，需 MEEG 32 被试数据】

**表 4** 跨被试 LOSO 结果（$n$ = 被试数）。

| 配置 | 参数量 | 准确率（跨被试, %） | 宏平均 F1（%） | 最差被试（%） | 崩溃率 | ΔACC (pp) | $p$ |
|---|---|---|---|---|---|---|---|
| AT-DGNN（基线） | 2 680 350 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | — | — |
| AT-DGNN + 频段注意力 | 2 680 380 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 | 【待回填】 |

**分析要点**：跨被试场景下个体差异显著，频段加权若能提供跨被试通用的频段先验，其收益应较被试内更明显；反之若收益消失，则说明该模块学到的是被试特异性的频段偏好。

---

## 5 结论

> 【本节的数值结论待 §4 回填；以下为已定稿的论证结构】

本文以 AT-DGNN 为基线，针对其"具备频谱分解能力但缺少显式频段加权"的问题，提出频段注意力模块，并通过被试内受控实验与跨被试 LOSO 两级协议进行评估。主要结论如下：

1. **模块设计层面**：频段注意力把 AT-DGNN 中**静态、隐式、未与规范频段对齐**的频谱加权，改造为**显式、逐样本自适应、可解释**的加权形式。与对高维特征构造可学习注意力核的方案相比，在 5 维频段轴上做注意力具备充足的调节余量（softmax 的单位杠杆随维度增加而迅速衰减），且权重语义明确。

2. **代价与安全性层面**：模块仅增加 30 个参数（基线的 $1.1\times10^{-5}$），计算开销可忽略；由于固定频段分解构成对 1–50 Hz 的单位分解，当权重取等值时模块**近似还原原始输入**，实现上与基线逐比特等价。因此该模块可在**不改变模型基本行为**的前提下引入，属于低风险的增量改进。

3. **有效性层面**：【待回填】。

4. **可解释性层面**：【待回填】。

**未来工作**：进一步在更多被试与数据集（如 DEAP）上验证；把频段加权与电极维加权联合建模（频段 × 空间二维注意力）；引入可学习的频段边界或滤波器组以放松固定频段划分的假设。

---

## 参考文献

1. Xiao M, Zhu Z, Xie K, et al. MEEG and AT-DGNN: Improving EEG emotion recognition with music introducing and graph-based learning[C]//2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2024: 4201–4208.
2. Zhang S, Chu C, Zhang X, et al. EEG emotion recognition using AttGraph: A multi-dimensional attention-based dynamic graph convolutional network[J]. Brain Sciences, 2025, 15(6): 615.
3. Song T, Zheng W, Song P, et al. EEG emotion recognition using dynamical graph convolutional neural networks[J]. IEEE Transactions on Affective Computing, 2018, 11(3): 532–541.
4. Ding Y, Robinson N, Tong C, et al. LGGNet: Learning from local-global-graph representations for brain–computer interface[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.
5. Zheng W L, Lu B L. Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks[J]. IEEE Transactions on Autonomous Mental Development, 2015, 7(3): 162–175.
6. Zheng W L, Liu W, Lu Y, et al. EmotionMeter: A multimodal framework for recognizing human emotions[J]. IEEE Transactions on Cybernetics, 2018, 49(3): 1110–1122.
7. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. 2017: 5998–6008.
8. Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[C]//International Conference on Learning Representations. 2017.
9. Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of Neural Engineering, 2018, 15(5): 056013.
10. Schirrmeister R T, Springenberg J T, Fiederer L D J, et al. Deep learning with convolutional neural networks for EEG decoding and visualization[J]. Human Brain Mapping, 2017, 38(11): 5391–5420.
11. Ding Y, Robinson N, Zeng Q, et al. TSception: A deep learning framework for emotion detection using EEG[C]//2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020: 1–7.
12. Zhong P, Wang D, Miao C. EEG-based emotion recognition using regularized graph neural networks[J]. IEEE Transactions on Affective Computing, 2020, 13(3): 1290–1301.
13. Li Y, Zheng W, Zong Y, et al. A bi-hemisphere domain adversarial neural network model for EEG emotion recognition[J]. IEEE Transactions on Affective Computing, 2018, 12(2): 494–504.
14. Bouthillier X, Delaunay P, Bronzi M, et al. Accounting for variance in machine learning benchmarks[J]. Proceedings of Machine Learning and Systems, 2021, 3: 747–769.
15. Ba J L, Kiros J R, Hinton G E. Layer normalization[J]. arXiv preprint arXiv:1607.06450, 2016.

---

## Abstract

**Objective**: In EEG-based emotion recognition, frequency bands (δ/θ/α/β/γ) carry well-established neurophysiological information about affective states, yet many deep models do not weight them explicitly. Focusing on AT-DGNN (BIBM 2024), we show that its temporal front-end performs a learned spectral decomposition but never weights the resulting frequency content, and we propose a band-attention module together with within-subject and cross-subject evaluations.

**Methods**: A code-level analysis shows that the Tception temporal learner is a learnable filter bank followed by a log-power layer $y=\log(\overline{x^2})$, i.e. a learned spectral decomposition whose frequency weighting is (i) static, (ii) not aligned with canonical EEG bands, and (iii) merely concatenated. The proposed band-attention module therefore (1) splits the input into five canonical bands with complementary brick-wall frequency masks forming a partition of unity over 1–50 Hz, (2) scores each band from its log-power descriptor using a learnable static prior plus a shared sample-adaptive modulation, and (3) recombines the bands by a normalised weighted sum before the **unchanged** AT-DGNN backbone.

**Results**: The module adds only 30 parameters (about $1.1\times10^{-5}$ of the baseline) and, because the band masks form a partition of unity, reduces to the original input under uniform weights, so it cannot disturb the base model. 【to be filled】.

**Conclusion**: 【to be filled】. The contribution is to turn the static, implicit and unaligned spectral weighting of AT-DGNN into an explicit, sample-adaptive and interpretable band weighting at negligible cost.

**Key words**: EEG; emotion recognition; band attention; graph neural network; cross-subject; deep feature fusion

---

## 作者注（不随稿件提交）

1. **本稿的成立与否取决于 §4.3 的结果。** 若频段注意力相对基线为显著正增益，§4 与摘要按上表结构回填即可；若为中性（不显著）或负向，则**不得**写成"提升了识别性能"，此时建议的替代叙事为：
   - 中性 → 把论文定位为"显式频段加权的代价-收益分析"：以 30 个参数的代价，显式频段加权在中性结果上给出【】；并结合 §4.5 的权重测量说明"该被试上频段重要性差异有限"。
   - 负向 → 定位为"频段注意力在单被试条件下的适用性边界"，并给出跨被试（LOSO）作为决定性验证。
2. **所有待回填数字必须由脚本从 `experiments/results/*.json` 自动生成**（`experiments/summarize_multiseed.py`），不得手工转录。
3. **跨被试验证是本文的关键卖点，但需要 MEEG 32 被试数据**；若数据无法获得，应在文中明确标注该实验为"未完成"，并把 §4.6 移入"未来工作"，而不是省略。
