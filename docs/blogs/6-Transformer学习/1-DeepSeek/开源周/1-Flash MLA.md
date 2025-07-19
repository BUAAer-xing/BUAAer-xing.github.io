## 0-概述

FlashMLA 是 DeepSeek 推出的高效解码内核，专为 NVIDIA Hopper 架构（如 H100/H800）GPU 优化，旨在加速大语言模型（LLM）在自回归解码阶段的推理性能，特别适用于处理可变长度序列的场景。

其主要优化措施包括：
1. **低秩分解的多头潜在注意力机制（MLA）**
	- FlashMLA 引入了低秩分解的 MLA 机制，通过对 Key 和 Value 进行低维压缩，显著减少了 KV 缓存的内存占用，降低了计算复杂度，提升了长序列处理的效率。
2. **分页式 KV 缓存机制**
	- 采用块大小为 64 的分页 KV 缓存，有效解决了传统 KV 缓存的内存碎片化问题，提高了显存利用率，支持高效处理不同长度的序列数据。
3. **针对 Hopper GPU 的深度优化**
	- FlashMLA 充分利用 Hopper 架构的高带宽内存和 Tensor Core，结合 CUDA 核心的优化，实现了高达 3000 GB/s 的内存带宽和 580 TFLOPS 的计算性能，显著提升了推理效率。
4. **支持 BF16 精度计算**
	- 通过支持 BF16 精度，FlashMLA 在保持计算准确度的同时，降低了内存带宽压力，提高了计算效率。
5. **内核级的调度优化**
	- 在新版本中，FlashMLA 通过重构内核调度策略，实现了 CUDA 核心与 Tensor Core 操作的重叠执行，以及内存访问与计算的并行，进一步提升了计算资源的利用率。

## 1-背景介绍

### 1.1-引入

在深度学习，特别是自然语言处理（NLP）领域，注意力机制（Attention Mechanism）是一个非常重要的概念。Attention机制的起源可以追溯到对生物视觉注意力的模拟以及神经机器翻译的实际需求。Bahdanau等人的工作首次将Attention机制引入自然语言处理领域；而Transformer架构则将Attention机制推向了一个新的高度，使其成为现代自然语言处理的核心技术之一。随着DeepSeek的爆火，它们的MLA注意力方法更是将Attention机制的应用和优化发展到了极致。

Attention机制核心思想是在处理数据时，模型可以有选择性地关注输入的不同部分，进而提升模型的性能。目前，它已经出现了多个升级优化版本，MHA（Mutil Head Attention，多头注意力）、MQA（Mutil Query Attention，多请求注意力）、GQA（Group Query Attention，组请求注意力）、MLA（Multi-Head Latent Attention，多头潜注意力）等。

### 1.2-传统注意力机制

单头注意力只使用一个注意力头来计算权重，从而降低计算复杂度，同时保留注意力机制的核心思想。

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506103653.png)


![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250430155436.png)

计算过程

**(1) 计算 Query, Key, Value 矩阵**

每个输入词都会被映射成三个不同的向量：

- $Q$ 是查询（Query），其表示当前需要关注的内容，例如在机器翻译中，查询可能是目标语言句子中的一个词。
- $K$ 是键（Key），表示与查询进行匹配的内容，例如源语言句子中的词。
- $V$ 是值（Value），表示最终要提取的信息，通常与键对应。

定义转换矩阵：

$$Q=XW_Q,K=XW_K,V=XW_V$$

其中，$W_Q,W_K,W_V$ 是可学习的参数矩阵。


**(2)计算点积**

计算查询 Q 和键 K 的点积，得到注意力分数矩阵：

$$scores=QK^T$$


**(3)缩放**：

将点积结果除以 $d_k$：其中，$d_k$ 是 Key 向量的维度，$d_k$ 作为缩放因子，避免数值过大导致梯度消失问题。

$$\text{scaled\_scores}=\frac{QK^T}{\sqrt{d_k}}$$

**(4)softmax归一化**：对缩放后的点积结果应用softmax函数，得到注意力权重矩阵

$$\text{attention\_weights}=softmax(\frac{QK^T}{\sqrt{d_k}})$$

**(5)加权求和**：将注意力权重矩阵与值 V 相乘，得到加权求和的结果

$$\text{output}=\text{attention\_weights}×V$$


### 1.3-多头注意力机制

**多头注意力（Multi-Head Attention, MHA）** 是 Transformer 的核心机制之一，它是 **自注意力（Self-Attention）** 的扩展版本，使模型能够从多个角度关注不同的部分，提高模型的表达能力。相比于单一注意力头，多头注意力可以更好地捕捉不同级别的语义信息，并增强 Transformer 在复杂任务上的表现。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207203517.png)

值得注意的是：在**多头注意力**中，不止使用一个注意力计算，而是使用  h **个并行的注意力头**。<font color='red'><b>每个注意力头有自己独立的权重</b></font>：

$$Q_i = X W_{Q_i}, \quad K_i = X W_{K_i}, \quad V_i = X W_{V_i} \quad (i = 1,2,\dots,h)$$

然后，每个头独立计算注意力：

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

最终，我们将所有  h  个头的输出拼接（Concat），并通过一个线性变换  $W_O$  进行融合：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, …, \text{head}_h) W_O$$

• **Concat 操作**：将所有注意力头的结果拼接，形状变为  $(L, h \cdot d_k)$ 。
• **线性变换** $W_O$ ：降维到  $d_{model}$  维度，使得多头注意力的输出形状与输入相匹配。


## 2-KV Cache内容

### 2.1-理论分析

在推理阶段，**KV Cache（键值缓存）** 的主要作用是加速自回归生成过程。以 GPT 类模型为例，其文本生成是逐词进行的：模型首先接收输入问题，基于该输入生成第一个词；随后，生成第二个词时，模型不仅依赖原始输入，还依赖已生成的第一个词；后续每一步都在此前生成的上下文基础上，逐步生成下一个词。为了避免每一步都重复计算前面所有 token 的注意力表示，KV Cache 会缓存每层 Transformer 中历史 token 的键（Key）和值（Value），从而在生成新词时仅需对当前词与历史缓存进行注意力计算，大幅减少冗余计算，提高推理效率。

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506111817.png)

大模型在解码基本上都是通过自回归的方式进行。即：最新的Token输出依赖于先前生成或者预先填入的Token。举个例子，假如我们输入“窗前明月光下一句是”，那么模型每次生成一个token，输入输出会是这样（方便起见，默认每个token都是一个字符，其中\[BOS\]和\[EOS\]分别是起始符号和终止符号）。

```text
step0: 输入=[BOS]窗前明月光下一句是；输出=疑
step1: 输入=[BOS]窗前明月光下一句是疑；输出=是
step2: 输入=[BOS]窗前明月光下一句是疑是；输出=地
step3: 输入=[BOS]窗前明月光下一句是疑是地；输出=上
step4: 输入=[BOS]窗前明月光下一句是疑是地上；输出=霜
step5: 输入=[BOS]窗前明月光下一句是疑是地上霜；输出=[EOS]
```

仔细想一下，在生成“疑”字的时候，用的是输入序列中“是”字的最后一层hidden state，通过最后的分类头预测出来的。以此类推，后面每生成一个字，使用的都是输入序列中最后一个字的输出。可以注意到，下一个step的输入其实包含了上一个step的内容，而且只在最后面多了一点点（一个token）。那么下一个step的计算应该也包含了上一个step的计算。

由于decoder是causal的（即，一个token的注意力attention只依赖于它前面的token），在每一步生成过程中，我们实际上是在重复计算相同的前一个token的注意力，而<font color='red'><b>我们真正需要做的是仅计算新token的注意力</b></font>。这就是KV cache发挥作用的地方。通过缓存之前的k和v，我们可以专注于只计算新token的注意力。以下是每个Token的Attention分数的计算过程，可以发现，$step_i$相比$step_{i−1}$，之前的Attention score是不变的，那么<font color='red'><b>一个新的tokne进来，只需要计算当前token对应的kv就可以 了，后面直接拼起来就好了</b></font>。这里能做kv cache的主要原因是由于mask矩阵的作用，这就是causal模型的先天优势。


![image.png|center|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506114351.png)

![image.png|center|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506114510.png)


### 2.2-实际例子

#### 没有KV Cache的情况

每次计算都是完整的输入进行计算

假设我们的 Q, K, V 分别如下：

$$

Q = \begin{bmatrix}

0.212 & 0.04 & 0.63 & 0.36 \\

0.1 & 0.14 & 0.86 & 0.77 \\

0.31 & 0.36 & 0.19 & 0.72

\end{bmatrix}, \quad

K = \begin{bmatrix}

0.31 & 0.84 & 0.963 & 0.57 \\

0.45 & 0.94 & 0.73 & 0.58 \\

0.36 & 0.83 & 0.1 & 0.38

\end{bmatrix}

$$

$$

V = \begin{bmatrix}

0.36 & 0.83 & 0.1 & 0.38 \\

0.31 & 0.36 & 0.19 & 0.72 \\

0.31 & 0.84 & 0.963 & 0.57

\end{bmatrix}

$$

在计算完：
$$

\frac{QK^T}{\sqrt{d_k}}

$$

得到 attention 矩阵后，我们创建一个 **masking** 矩阵（其中 $-1e9$ 是一个极小值），将其与 attention 矩阵相加：

$$

M = \begin{bmatrix}

0 & 1 & 1 \\

0 & 0 & 1 \\

0 & 0 & 0

\end{bmatrix} \times (-1e9) =

\begin{bmatrix}

0 & -1e9 & -1e9 \\

0 & 0 & -1e9 \\

0 & 0 & 0

\end{bmatrix}

$$

$$

\frac{QK^T}{\sqrt{d_k}} =

\begin{bmatrix}

0.455605 & 0.40085 & 0.15466 \\

0.70784 & 0.6255 & 0.2654 \\

0.495935 & 0.5171 & 0.3515

\end{bmatrix}

$$

$$

\frac{QK^T}{\sqrt{d_k}} + M =

\begin{bmatrix}

0.455605 & -1e9 & -1e9 \\

0.70784 & 0.6255 & -1e9 \\

0.495935 & 0.5171 & 0.3515

\end{bmatrix}

$$

接下来，沿行应用 softmax，将这些值转换为概率分布。将 $softmax$ 应用于注意力矩阵后，所有这些极小的值 $-1e9$ 都将变为零：

$$
softmax\left(
\begin{bmatrix}
0.455605 & -1e9 & -1e9 \\
0.70784 & 0.6255 & -1e9 \\
0.495935 & 0.5171 & 0.3515
\end{bmatrix}
\right)

\begin{bmatrix}
1.0 & 0 & 0 \\
0.520573 & 0.479427 & 0 \\
0.346392 & 0.353802 & 0.299806
\end{bmatrix}
$$

#### 具有KV Cache的情况

不存储 Q 的情况，仅存储 K,V的情况：

$$
Q_1 =
\begin{bmatrix}
0.212 & 0.04 & 0.63 & 0.36 \\
	-	& - & - & - \\
	-	& - & - & -
\end{bmatrix}, \quad
Q_2 =
\begin{bmatrix}
	-	& - & - & - \\
0.1 & 0.14 & 0.86 & 0.77 \\
	-	& - & - & -
\end{bmatrix}, \quad
Q_3 =
\begin{bmatrix}
	-	& - & - & - \\
	-	& - & - & - \\
0.31 & 0.36 & 0.19 & 0.72
\end{bmatrix}
$$

$$
K_1 =
\begin{bmatrix}
0.31 & 0.84 & 0.963 & 0.57 \\
	-	& - & - & - \\
	-	& - & - & -
\end{bmatrix}, \quad
K_2 =
\begin{bmatrix}
0.31 & 0.84 & 0.963 & 0.57 \\
0.45 & 0.94 & 0.73 & 0.58 \\
	-	& - & - & -
\end{bmatrix}, \quad
K_3 =
\begin{bmatrix}
0.31 & 0.84 & 0.963 & 0.57 \\
0.45 & 0.94 & 0.73 & 0.58 \\
0.36 & 0.83 & 0.1 & 0.38
\end{bmatrix}
$$

$$
V_1 =
\begin{bmatrix}
0.36 & 0.83 & 0.1 & 0.38 \\
	-	& - & - & - \\
	-	& - & - & -
\end{bmatrix}, \quad
V_2 =
\begin{bmatrix}
0.36 & 0.83 & 0.1 & 0.38 \\
0.31 & 0.36 & 0.19 & 0.72 \\
	-	& - & - & -
\end{bmatrix}, \quad
V_3 =
\begin{bmatrix}
0.36 & 0.83 & 0.1 & 0.38 \\
0.31 & 0.36 & 0.19 & 0.72 \\
0.31 & 0.84 & 0.963 & 0.57
\end{bmatrix}
$$


$$
\frac{Q_1 K_1^T}{\sqrt{d_k}} =
\begin{bmatrix}
0.455605 & - & - \\
	-	& - & - \\
	-	& - & -
\end{bmatrix}, \quad
\frac{Q_2 K_2^T}{\sqrt{d_k}} =
\begin{bmatrix}
	-	& - & - \\
0.70784 & 0.6255 & - \\
	-	& - & -
\end{bmatrix}, \quad
\frac{Q_3 K_3^T}{\sqrt{d_k}} =
\begin{bmatrix}
	-	& - & - \\
	-	& - & - \\
0.495935 & 0.5171 & 0.3515
\end{bmatrix}
$$

相加后的结果与存储 $Q$ 时 masking 的结果相同：

$$
\frac{Q_1 K_1^T}{\sqrt{d_k}} +
\frac{Q_2 K_2^T}{\sqrt{d_k}} +
\frac{Q_3 K_3^T}{\sqrt{d_k}} =
\begin{bmatrix}
0.455605 & - & - \\
0.70784 & 0.6255 & - \\
0.495935 & 0.5171 & 0.3515
\end{bmatrix}
$$

应用 $softmax$：

$$
softmax\left(
\begin{bmatrix}
0.455605 & - & - \\
0.70784 & 0.6255 & - \\
0.495935 & 0.5171 & 0.3515
\end{bmatrix}
\right)
=
\begin{bmatrix}
1.0 & - & - \\
0.520573 & 0.479427 & - \\
0.346392 & 0.353802 & 0.299806
\end{bmatrix}
$$

可以观察到，与存储 Q 时的结果是一致的，这也代表在接下与V矩阵计算得到的 Attention 结果也将一样，这也就是为什么在 KV Cache 时不需要存储 Q 的原因。


![image.png|center|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506142305.png)



### 2.3-KV Cache在哪里使用？

当每生成一个新的 token 就会把这个新的 token **append** 进之前的序列中，在将这个序列当作新的输入进行新的 token 生成，直到$<eos>$ token 结束。这使得每次新序列输入时都需要取重复计算前面的$(n−1)$ 个 token 的 $(q,k,v)$，浪费了很多资源，KV Cache 就是在这里使用的，我们在每次处理新的序列时，可以同时将之前计算的 key, value 一同缓存，并传入下一次计算，这样就节省了很多计算的时间，避免了冗余计算。

### 2.4-KV Cache 节省哪部分内容？

首先，我们要知道，Self-Attention 通过将输入序列变换成三个向量来操作：查询向量（Query），键向量（Key）和值向量（Value）。这些向量是通过对输入进行线性变换得到的。注意力机制基于Q 向量和K向量之间的相似度来计算V向量的加权求和。然后，将这个加权求和的结果连同原始输入一起送入前馈神经网络，以产生最终输出。

这一过程允许模型专注于相关信息并捕捉长距离依赖关系。 那么回到问题，它节省了哪部分计算呢？<font color='red'><b>它节省了对于键（Key）和值（Value）的重复计算，不需要对之前已经计算过的 Token 的K和 V重新进行计算</b></font>。因为对于之前的 Token 可以复用上一轮计算的结果，避免了重复计算，只需要计算当前 Token 的$Q、K、V$。


## 3-KV Cache注意力机制改良

### 3.1-如何优化KV Cache

KV cache的峰值显存占用大小计算公式：

$$2 \times Length \times batch\_size \times [d \times n\_kv\_heads] \times Layers \times \text{k-bits}$$

由此我们可以看出影响KV cache的具体因素：
- **k-bits**: 数据类型，FP16 占2个bytes。（量化）
- **2**: 代表 Key/Value 两个向量
- **Length**: 输入输出序列的长度（循环队列管理窗口KV，减少长度kv）
- **Layers**：模型层数
- **d x n_kv_heads**：kv维度（MQA/GQA通过减少KV的头数减少显存占用）
- **batch_size** : KV Cache 与 batch size 度呈线性关系，随着 batch size 的增大，KV cache 占用的显存开销快速增大，甚至会超过模型本身。**推理时，batch size 决定模型一次性处理多少个输入。例如，batch size = 1 表示逐个样本处理；batch size = 32 表示并行处理 32 个样本**。
- 操作系统管理：现GPU的KV Cache的有效存储率较低低 （page-attention）

![image.png|center|300](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506143030.png)

在bf16格式下的13B模型中，我们只有大约10G的空间来储存kv cache

KV Cache 的引入也使得推理过程分为如下两个不同阶段，进而影响到后续的其他优化方法。
- **预填充阶段(Prefill)**：**发生在计算第一个输出 token 过程中，计算时需要为每个 Transformer layer 计算并保存 key cache 和 value cache**；
	- FLOPs 同 KV Cache 一致，<font color='red'><b>存在大量 GEMM (GEneral Matrix-Matrix multiply) 操作，属于 Compute-bound 类型计算</b></font>。
- **解码阶段(Decoder)**：**发生在计算第二个输出 token 至最后一个 token 过程中，这时 KV Cache 已存有历史键值结果，每轮推理只需读取 Cache，同时将当前轮计算出的新的 Key、Value 追加写入至 Cache**；
	- <font color='red'><b>GEMM 变为 GEMV (GEneral Matrix-Vector multiply) 操作，FLOPs 降低，推理速度相对预填充阶段变快，这时属于 Memory-bound 类型计算</b></font>。


根据公式总结，有四类方法：
1. `n_kv_heads`:`MQA`/`GQA`通过减少KV的头数减少显存占用
2. `Length` : 通过减少长度`L`, 以减少`KV`显存占用，如使用循环队列管理窗口`KV`
3. `KV-Cache`的管理：从`OS`(操作系统)的内存管理角度，减少碎片，如Paged Attention
4. `K-bits`: 从量化角度减少`KV cache`的宽度，如使用[LLM-QAT](https://zhida.zhihu.com/search?content_id=236521933&content_type=Article&match_order=1&q=LLM-QAT&zhida_source=entity)进行量化


### 3.2-减少头数（MQA/GQA）

- 一种手段是减少KV heads的数量，如果以`MQA(Multi-Query-Attention)`来说，`KV head 8->1` 之间节省7倍存储量，MQA (Multi Query Attention，多查询注意力) 是多头注意力的一种变体。其主要区别在于，在 MQA 中不同的注意力头共享一个K和V的集合，每个头只单独保留了一份查询参数。因此K和V的矩阵仅有一份，这大幅度减少了显存占用，使其更高效。由于MQA改变了注意力机制的结构，因此模型通常需要从训练开始就支持 MQA 。也可以通过对已经训练好的模型进行微调来添加多查询注意力支持，仅需要约 5% 的原始训练数据量 就可以达到不错的效果。包括 Falcon、SantaCoder、StarCoder 等在内很多模型都采用了 MQA 机制。
- 对于`GQA(Grouped-Query_Attention)`来说，平衡精度将KV head 8 -> N, $1<N<8$ 之间trading off精度和速度，GQA（Grouped Query Attention，分组查询注意力）是一种介于多头注意力和 MQA 之间的折中方案。它将查询头（Query Heads）分组，并在每组中共享一个键头（Key Head）和一个值头（Value Head）。表达能力与推理速度：GQA既保留了多头注意力的一定表达能力，又通过减少内存访问压力来加速推理速度。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506144050.png)


### 3.3-减少Length的长度

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506145854.png)

- **图(a)：Dense Attention（标准 Transformer 注意力）**
	- 结构：每个 token attend 到所有历史 token，包含全部 $T$ 个 token。
	- 复杂度：$O(T^2)$（非常昂贵）
	- PPL（Perplexity）：5641（非常高，性能差）
	- 缺点：
		- 随着文本长度增加，计算量指数级增长。
		- KV 缓存线性增长，最终占满显存。
- **图(b)：Window Attention（固定窗口注意力）**
	- 结构：只关注最近 $L$ 个 token，远处 token 被丢弃。
	- 复杂度：$O(TL)$（线性优化）
	- PPL：5158（仍然很高）
	- 缺点：
		- 一旦最初 token 被淘汰（evicted），模型无法访问上下文，导致性能下降。
		- 不适用于依赖长距离依赖的信息生成任务。
- **图(c)：Sliding Window + Re-computation（滑窗+重计算）**
	- 结构：滑动窗口 + 对被移除的历史 token 重新计算 attention。
	- 复杂度：$O(TL^2)$（每次都重计算历史 KV 缓存）
	- PPL：5.43（低，性能好）
	- 缺点：
		- 显著的计算冗余：每生成一个新 token，需重复前缀计算。
		- 延迟高，实时性差。
- **图(d)：StreamingLLM（本文方法）**
	- 结构：
	- 当前 token attend 到最近的 $L$ 个缓存 token。
	- 同时引入了 Attention Sink：一个“锚点”，让模型始终能访问初始上下文（黄色区块）。
	- 复杂度：$O(TL)$（线性）
	- PPL：5.40（接近最优）
	- 优点：
		- 保留关键上下文信息（靠 attention sink），避免性能退化。
		- 高效推理，计算不重叠，适合长文本。
		- 支持 KV Cache 截断策略，兼顾准确性与性能。

### 3.4-KV Cache的管理，减少碎片

在 Paged Attention 之前，业界主流 LLM 推理框架在 KV Cache 管理方面均存在一定的低效。HuggingFace Transformers 库中，KV Cache 是随着执行动态申请显存空间，由于 GPU显存分配耗时一般都高于 CUDA kernel 执行耗时，因此动态申请显存空间会造成极大的时延开销，且会引入显存碎片化。FasterTransformer 中，预先为 KV Cache 分配了一个充分长的显存空间，用于存储用户的上下文数据。例如 LLaMA-7B 的上下文长度为 2048，则需要为每个用户预先分配一个可支持 2048 个 tokens 缓存的显存空间。如果用户实际使用的上下文长度低于2048，则会存在显存浪费。**Paged Attention 将传统操作系统中对内存管理的思想引入 LLM，实现了一个高效的显存管理器，通过<font color='red'><b>精细化管理显存</b></font>，实现了在物理非连续的显存空间中以极低的成本存储、读取、新增和删除键值向量**。

具体来讲，Paged Attention 将每个序列的 KV Cache 分成若干块，每个块包含固定数量token 的键和值。在注意力计算期间，PagedAttention 内核可以有效地识别和获取这些块。因为块在内存中不需要连续，因而可以用一种更加灵活的方式管理 key 和 value ，就像在操作系统的虚拟内存中一样：可以将块视为页面，将 token 视为字节，将序列视为进程。序列的连续逻辑块通过块表映射到非连续物理块中。物理块在生成新 token 时按需分配。在 PagedAttention 中，内存浪费只会发生在序列的最后一个块中。这使得在实践中可以实现接近最佳的内存使用，仅浪费不到 4%。

- 优化KV-Cache存储模型：**PagedAttention**
- 对于以下连续存储模型，如果将Page Length改成4
- <font color='red'><b>PagedAttention不是去改变Attention的计算，而是改变KV-cache的存取方式</b></font>

![image.png|center|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506150550.png)

`vLLM`构建了`CacheEngine`先申请一块大的连续`GPU`存储，再自己统一做内存管理。


### 3.5-减少bits数，量化模型

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506150706.png)

- `LLM-QAT`在量化训练过程，将`KV-Cache`也做`Quantization`
- 在部署时`KV-Cache`如果是`16bit`，量化成`4bit`的话，显存直接减少4倍


### 3.6-总结

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506153548.png)


## 4-算子融合

### 4.1-引入

算子融合是深度学习模型推理的一种典型优化技术，旨在通过减少计算过程中的访存次数和 Kernel 启动耗时达到提升模型推理性能的目的，该方法同样适用于 LLM 推理。

以 HuggingFace Transformers 库推理 LLaMA-7B 模型为例，经分析模型推理时的算子执行分布如下图所示，该模型有 30 个类型共计 2436 个算子，其中 aten::slice 算子出现频率为 388 次。大量小算子的执行会降低 GPU 利用率，最终影响推理速度。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506151614.png)


目前业界基本都针对 Transformer layer 结构特点，手工实现了算子融合。以 DeepSpeed Inference 为例，算子融合主要分为如下四类：
- **归一化层和 QKV 横向融合**：将三次计算 Query/Key/Value 的操作合并为一个算子，并与前面的归一化算子融合。
- **自注意力计算融合**：将自注意力计算涉及到的多个算子融合为一个，业界熟知的 FlashAttention 即是一个成熟的自注意力融合方案。
- **残差连接、归一化层、全连接层和激活层融合**：将 MLP 中第一个全连接层上下相关的算子合并为一个。
- **偏置加法和残差连接融合**。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506151655.png)

由于算子融合一般需要定制化实现算子 CUDA kernel，因此对 GPU 编程能力要求较高。随着编译器技术的引入，涌现出 OpenAI Triton 、TVM 等优秀的框架来实现算子融合的自动化或半自动化，并取得了一定的效果。


### 4.2-高性能算子

针对 LLM 推理运行热点函数编写高性能算子，也可以降低推理时延。
- **GEMM 操作相关优化**：在 LLM 推理的<font color='red'><b>预填充阶段</b></font>，Self-Attention 和 MLP 层均存在多个 GEMM 操作，耗时占据了推理时延的 80% 以上。GEMM 的 GPU 优化是一个相对古老的问题，在此不详细展开描述算法细节。英伟达就该问题已推出 cuBLAS、CUDA、CUTLASS 等不同层级的优化方案。例如，FasterTransformer 框架中存在大量基于 CUTLASS 编写的 GEMM 内核函数。另外，Self-Attention 中存在 GEMM+Softmax+GEMM 结构，因此会结合算子融合联合优化。
- **GEMV 操作相关优化**：在 LLM 推理的<font color='red'><b>解码阶段</b></font>，运行热点函数由 GEMM 变为 GEMV。相比 GEMM，GEMV 的计算强度更低，因此优化点主要围绕降低访存开销开展。


### 4.3-FlashAttention

FlashAttention 是由 Tri Dao 等人提出的一种高效且精确的注意力机制计算方法，旨在解决 Transformer 在处理长序列时计算和内存开销呈二次增长的问题。该方法通过引入 IO 感知的设计，优化了 GPU 内存访问模式，从而在不牺牲精度的前提下，实现了显著的加速和内存节省。

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506152222.png)
要解决这些问题，需要做两件主要的事情：
- 在不访问整个输入的情况下计算 softmax
- 不为反向传播存储大的中间 attention 矩阵

为此 FlashAttention 提出了两种方法来分布解决上述问题：**tiling 和 recomputation**。
- tiling - 注意力计算被重新构造，将输入分割成块，并通过在输入块上进行多次传递来递增地执行softmax操作。
- recomputation - 存储来自前向的 softmax 归一化因子，以便在反向中快速重新计算芯片上的 attention，这比从HBM读取中间矩阵的标准注意力方法更快。

<font color='red'><b>由于重新计算，这虽然导致FLOPS增加，但是由于大量减少HBM访问，FlashAttention运行速度更快</b></font>。该算法背后的主要思想是分割输入，将它们从慢速HBM加载到快速SRAM，然后计算这些块的 attention 输出。在将每个块的输出相加之前，将其按正确的归一化因子进行缩放，从而得到正确的结果。


**Flash Attention**专注于标准多头注意力的高效实现，**通过减少访问显存次数，优化并行度提升计算性能**，但并不直接兼容MLA。

传统MHA 的计算复杂度为 O(n²)（n 为序列长度），并且需要存储大量的中间结果，这在长序列任务中会导致严重的显存压力和计算延迟。FlashAttention 的核心理念是避免显式计算和存储完整的注意力矩阵，而是通过**分块计算（tiling）** 和**融合操作**，将注意力计算优化为接近O(n)的复杂度，同时大幅减少GPU内存访问。

- **1）分块处理**: 将输入序列分割成小块（tiles），逐块计算注意力，避免一次性加载整个矩阵。
- **2）显存优化**: 通过在线计算 softmax 和融合操作，减少中间结果的存储需求。
- **3）硬件架构友好**: 充分利用GPU高速内存（如共享缓存）和并行计算能力。

## 5-多头潜在注意力（MLA）

### 5.1-核心解释

FlashMLA是针对**Hopper架构**GPU的**推理**加速开源项目，是优化变长序列场景Decode阶段的显存碎片化和时延等问题，当前开源<font color='red'><b>仅支持BF16精度</b></font>。

DeepSeek使用的**Multi-Head Latent Attention技术可大大节省KV缓存**，从而显著降低了计算成本。

MLA的本质是<font color='red'><b>对KV的有损压缩，提高存储信息密度的同时尽可能保留关键细节</b></font>。该技术首次在DeepSeek-V2中引入，与分组查询和多查询注意力等方法相比，**MLA是目前开源模型里显著减小KV 缓存大小的最佳方法**。

![image.png|center|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506154230.png)

MLA的方法是将KV矩阵转换为低秩形式：**将原矩阵表示为两个较小矩阵（相当于潜向量）的乘积，在推断过程中，仅缓存潜向量，而不缓存完整的键KV**。这规避了分组查询注意力和多查询注意力的查询的信息损失，从而在降低KV缓存的前提下获得更好的性能。

MLA虽好，但明显没有针对现代加速框架的FlashAttention或PageAttention解决方案。这也使得DeepSeek R1在实际部署时需要单独优化KV吞吐性能。

![image.png|center|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250506161053.png)


Flash MLA 的核心是高效的 MLA 解码内核，关键技术包括：
1. **低秩矩阵压缩**：MLA 使用低秩矩阵，将KV缓存压缩为潜向量，减少内存占用。通过解压潜向量生成独特的KV头（KV Head）。
2. **针对GPU 优化**：FlashMLA针对Hopper GPU 的Tensor Core进行youh优化，实现了可达3000 GB/s 的显存带宽和 580 TFLOPS 的计算性能（H800 SXM5 配置）。使用了SM90的关键特性GMMA、namedbarrier同步、cp.async。
3. **Row-wise/Block-wise优化**：细粒度划分，在shared memory中原位处理计算，减少了额外的中间计算过程的显存占用，减少显存访问次数。
4. **Split-KV 分块处理**：将KV拆分给多个SM（Stream Multiprocessor）处理（或者多次迭代），然后在局部把partial计算结果合并。
5. **变长序列支持**：通过 tile_scheduler_metadata 和 num_splits 参数，，FlashMLA 支持变长序列的并行处理，以缓解负载不均衡问题。


- **分页KV缓存**
	- 参考PageAttention论文，实现类似于OS中的页表管理GPU Global Memory，分页KV Cache机制（Block size = 64）突破传统连续显存分配限制，提升显存利用率。
	- 每个batch的KV Cache以block为粒度进行分块存储
- **TMA异步Tensor加载指令**
	- 实现HBM到Share Memory(SRAM)的零拷贝，实现理论峰值带宽。
	- 使用TMA和多buffer，实现计算过程中同时异步加载数据，实现计算访存掩盖。
	- 访存和计算在不同的warp group实现相互掩盖
- **动态负载均衡**
	- 根据输入序列的长度和batch大小，以block为粒度，动态负载到对应的SM。
	- 针对变长序列，根据输入序列的长度和batch大小，以分页大小为粒度调度到不同的SM上，提升负载均衡。
	- 额外combine kernel 进行跨SM batch规约
- **Warp Specialization**
	- 多 Warp Group 实现矩阵和向量计算相互掩盖

---

**TMA（Tensor Memory Accelerator）**：
在 CUDA 的 **Hopper 架构中**，TMA（Tensor Memory Accelerator）是一种新引入的硬件功能，旨在实现全局内存与共享内存之间的高效异步张量数据传输。与传统使用 cp.async 的方式相比，TMA 支持由单线程发起的数据搬运操作，支持最高 5 维张量，通过复制描述符完成地址计算、边界检查等繁重工作，显著简化了编程模型并降低了资源占用。同时，TMA 支持线程块集群间共享内存通信，结合 CUDA 的异步屏障机制和流水线控制，可以实现计算与数据搬运的完全重叠，极大提升吞吐量，特别适用于大规模矩阵乘法、注意力机制等计算密集型任务。

**WG（Warp Group）**：
Warp Group 是 NVIDIA 在 **Hopper 架构**中引入的一种由多个 warp（通常为 4 个，即 128 个线程）组成的线程协作机制，旨在实现更高效的资源共享与线程同步。相比传统 warp，Warp Group 支持跨 warp 的低开销通信、共享寄存器与共享内存，并引入新的同步指令如 `__wg_barrier()`，特别适用于与 TMA 协同处理大规模张量数据搬运与计算任务，如 FlashAttention 和 GEMM 等，在提升吞吐量和执行效率方面具有显著优势。

---

### 5.2-FlashMLA的启示（针对升腾芯片）

- 优化实现Cube和Vector的高效流水掩盖，极致的tiling，提升访存带宽利用率和计算访存掩盖。
- 变长序列基于AI Core实现动态负载均衡优化
- 借助异构系统算力，将部分token计算动态卸载到CPU上，提升时延和吞吐






