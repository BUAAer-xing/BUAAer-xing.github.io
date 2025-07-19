
机器学习：不是试图在代码中明确定义如何执行一个任务，而是去**构建一个具有可调参数的灵活架构**，像是一堆旋钮和滑块，然后拿着大量实例，即给定一个输入是应该输出什么，**设法调整各种参数值**，以<font color='red'><b>模仿</b></font>这种表现。

## 0-Overview

Transformer是一种神经网络架构，根本上改变了人工智能的方法。Transformer在2017年的开创性论文“Attention is All You Need”中首次被提出，此后成为深度学习模型的首选架构，驱动着如OpenAI的GPT、Meta的Llama以及谷歌的Gemini等文本生成模型。除了文本，Transformer还应用于音频生成、图像识别、蛋白质结构预测甚至游戏玩法，展现了其在多个领域的多功能性。

从根本上讲，<font color='red'><b>文本生成Transformer模型基于下一个词预测的原则：给定用户的文本提示，最可能跟随该输入的下一个词是什么？</b></font>Transformer的核心创新和力量在于其自注意力机制的使用，使其能够处理整个序列并更有效地捕获长距离依赖关系。

### Transformer 架构
 
每个文本生成型 Transformer 都由以下三个关键组成部分构成：
1. **Embedding（嵌入层）**：对文本进行第一步处理，把输入切分成小块，并将其转化为向量。
	- 文本输入被分割为更小的单元，称为 token，可以是单词或子单词。
	- 这些 token 被转换为数值向量，称为嵌入，**嵌入能够捕获单词的语义信息**。
3. **Transformer Block（Transformer 块）**：这是模型处理和转换输入数据的基本构建模块。每个块包括：
	- **Attention Mechanism（注意力机制）**：Transformer 块的核心组件。
		- 它允许 token 之间相互“<font color='red'><b>交流</b></font>”，从而捕获上下文信息以及单词之间的关系。
	- **MLP（多层感知机）层**：一个独立于每个 token 操作的前馈神经网络。
		- **注意力层**的目标是<font color='red'><b>在 token 之间传递信息</b></font>，而 <b>MLP </b>的目标是<font color='red'><b>精细化每个 token 的表示</b></font>。
4. **Output Probabilities（输出概率）**：最终的**线性层**和 **softmax 层**将处理过的嵌入转换为概率，帮助模型预测序列中下一个 token。


## 1-Embedding



在transform架构中，embedding(嵌入层)的作用是<font color='red'><b>将输入文本数据转换为适合神经网络处理的数值表示，同时捕获词汇的语义信息和词间的关系</b></font>。

### 1.1-文本向量化

原始文本数据由单词或词元（token）构成，计算机无法直接处理这些离散的符号。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250127153620.png)

因此，embedding 层的首要任务是<font color='red'><b>将这些 token 映射为连续的、固定维度的数值向量</b></font>。
- （1）token需要计算机进行处理，因此，该token必须是数字化的
- （2）需要表示每个token之间的相互关系，故数字化后的token的数值需要体现出该关系才行（详见1.2）

这些向量表示了词汇的特征，使得神经网络能够进行计算。

### 1.2-捕获语义信息

Embedding 层不仅仅是简单的映射，其生成的数值向量能够捕获词语的语义相似性。

例如，在嵌入空间中，语义上相似的单词（如“猫”和“狗”）会拥有相近的向量表示，而完全无关的单词（如“猫”和“原子能”）在空间中的距离会较远。
$$
\textbf{猫} = (1, 0, 0, 0),\,\textbf{狗} = (0, 1, 2, 1),\,\textbf{原子能} = (0, 1000, 300, 32)
$$
$$
\|\textbf{狗} - \textbf{猫}\| = \|(-1, 1, 2, 1)\| = \sqrt{7}
$$
$$
\|\textbf{原子能} - \textbf{猫}\| = \|(0, 999, 298, 31)\| = \sqrt{1087766}
$$
---
那如何根据不同的token去进行编码呢？在机器学习中有两个比较基础的操作，分别是：**tokenizer**和**one-hot**。这两者做的事情都是对一段文本里面最基础的语义单元（token）进行数字化。
- **tokenize(标记器/分词器)**：在机器学习和自然语言处理（NLP）中，**Tokenize** 的作用是将文本数据分割成较小的单元，通常是单词、子词、或字符，这些单元被称为“token”（标记）。这一过程是文本数据预处理的关键步骤，因为机器学习模型无法直接处理原始文本，而需要将其转换为结构化的数据形式。
	- **作用：**
		1. 将输入的句子或文本分解成可供模型处理的标记序列。
		2. 准备后续的词向量嵌入或其他编码操作。
		3. 提高模型对语言结构的理解，例如使用 WordPiece 或 BPE 分词处理未登录词问题。
	- **常见方法：**
		- 按空格或标点分词（如英语的单词分割）。
		- 使用字节对编码（BPE）或 SentencePiece 等方法将文本划分为子词单位。
		- 针对中文、日文等语言，可以使用基于字或基于词的分词工具，如 jieba、THULAC 等。
	- **示例：**
		- 输入文本："机器学习很有趣！"
		- 分词结果（按词分割）：\["机器学习", "很", "有趣", "！"\]
	- **总结**: 
		- <font color='green'><b>数字化的方式比较简单，给每一个不同的token，分配一个独立的ID，就相当于是把所有的token，都投射到了一根一维的数轴上</b></font>。
	- **问题:**
		- 它把所有的token都投射到了一个一维的空间上，这会导致<font color='red'><b>这个空间里的信息过于密集，从而无法表达出一些比较复杂的语义</b></font>。把所有的语义都变成了长度问题，完全没有利用维度关系，去表示语义关系。
			- 比如，苹果、香蕉很近，但当苹果表示手机时，应该和华为离的比较近，但使用该一维表示方式，会导致该关系表示不明确。
			- 比如，苹果1，香蕉2，想表示苹果和香蕉的语义，最直接的为1+2=3，但3这个ID会被梨占用，导致无法表示该语义，该问题的核心还是在于，表示语义的空间维度过低，导致无法表示出更加复杂的组合语义。
- **one-hot(独热编码)**：**One-hot encoding** 是一种将离散变量（如分类变量）转换为数值表示的编码方法，通常用于表示类别数据。**每个类别被编码为一个长度为类别总数的二进制向量，其中只有一个位置为 1，其余位置为 0**。这种方式在机器学习中用于对分类特征进行数值化，使其可以输入到模型中进行计算。
	- **作用：**
		1. 用于将分类变量（如单词、类别标签）转换为模型可处理的数值格式。
		2. 保持类别之间的独立性，避免引入错误的顺序关系。
		3. 辅助后续的向量化处理或作为简单的输入形式。
	- **优缺点：**
		- **优点：** 简单直观，适用于类别较少的情况。
		- **缺点：** 如果类别数量过多（如字典大小非常大），会导致高维稀疏矩阵，增加存储和计算的负担。
	- **示例：**
		- 类别集合：\["猫", "狗", "鸟"\]
		- 编码结果：
			- 猫 → \[1, 0, 0\]
			- 狗 → \[0, 1, 0\]
			- 鸟 → \[0, 0, 1\]
	- **总结**:
		- <font color='green'><b>把二进制的每一位都对应一个token，也就是说，这个token的种类有多少个，该向量的长度就会有多长。与tokenizer进行对比的话，one-hot就相当于是对每个token都分配了一个单独的维度，最后组成的就是，有多少种token，就会有多少个维度的高维空间。</b></font>
	- **问题:** 它把所有的token都映射到不同的高维空间中，所有的token都是一个独立的维度，所有的token互相之间都是正交的，就会很难体现出token互相之间的联系。one-hot编码的空间维度过于高，token互相之间的语义关系，全部都是靠维度之间的关系去体现的，并没有充分把空间的长度给利用起来。
		- 比如，苹果100，香蕉010，表示苹果和香蕉的组合语义可以为110，但是两者之间的相关程度则无法进行表示，因为两者向量之间的内积为0。
	- **注意**：在自然语言处理中，one-hot 编码通常用于简单的文本表示，但由于高维度和稀疏性，常被词向量（如 Word2Vec、GloVe）或<font color='red'><b>嵌入层</b></font>(embedding)所替代。

![image.png|center|200](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250127160321.png)

---

### 1.3-词嵌入矩阵与解嵌入矩阵

直接使用高维的稀疏表示（如 one-hot 编码）会导致维度爆炸问题，并且计算效率低下。而 embedding 层通过学习一个相对低维的稠密向量表示，使得模型能够高效处理大量的词汇。

![center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207155948.png)

 在文本transform类的大模型中，每个模型都有一个预设的<font color='red'><b>词汇库</b></font>，包含所有可能的词汇，这些词汇（token）的个数，就是嵌入矩阵的列数，**嵌入矩阵(Embedding Matrix)的每一列都对应词汇库中的一个具体的token**。这些列决定了第一步中，每个token对应的向量。<font color='red'><b>词嵌入矩阵的初始数值是随机的，但将基于数据进行学习</b></font>。词嵌入矩阵的行个数，也就是该词向量的维度大多是高维的，比如在GPT-3中，词嵌入矩阵的行有12288个维度，列的个数大约为50k(50257个)，这表明，在GPT-3中，词汇库中大约有50k个token以及每个token对应的空间维度为12288。这样设计，可以同时兼顾分词器以及独热编码的优点。

当模型在训练阶段调整权重以确定不同的单词将如何被嵌入向量时，它们最终的嵌入向量，在空间中的方向往往具有某种语义意义。也就是在上面1.2中提到的，**具有相似含义的token之间往往离的更近，也就是说它们两者之间的向量内积往往会比较大**。

![image.png|center|500](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207160342.png)

值得注意的是，在embedding的过程中，即**根据输入文本创建向量组时，每个token对应的向量都是直接从嵌入矩阵中拉出来的**，所以，最开始每个向量只能编码单个单词的含义，没有上下文信息。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207162640.png)

注意，除了嵌入矩阵，解嵌入矩阵也是必要的，在大模型中，如何根据最后一个向量，来从词汇库中，选择最大概率的词进行输出，就需要用到该解嵌入矩阵。同时，和其他权重矩阵一样，解嵌入矩阵的初始值也是随机的，但将在训练过程中进行学习。

### 1.4-提供输入特征维度

Embedding 层的输出是固定维度的向量（通常是模型定义的维度，如 512 或 1024），这为后续的 Transformer 块（如多头自注意力机制和 MLP 层）提供了标准化的输入维度。这里的维度，也就是词空间向量的维度。（在NLP里面，这里的每个维度对应的是不同token中所含有的基础语义，而在图像的应用中，维度可以理解为是图像的一个个通道）

### 1.5-参数学习：

正如在1.3中所提到的那样，<font color='red'><b>Embedding 层的权重（即词向量）是可以训练的参数</b></font>。在模型训练过程中，embedding 层通过<font color='red'><b>反向传播</b></font>学习到**最能适应当前任务（如翻译、文本生成）的词向量**，从而使得生成的嵌入表示能够更好地反映词汇的上下文及任务相关的语义关系。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207205240.png)

在此处，训练出词嵌入矩阵和解嵌入矩阵，也是通过神经网络来进行实现的，比如13年谷歌提出的word2vec方法。这也被称为静态词嵌入方法，因为它训练出来的嵌入矩阵，无论句子如何变化，不同句子中的同一个词的向量表示都是一样的。也就是说，它只负责将句子中的词（token）直接映射成一个数值向量。而GPT则不同，虽然在最初的embedding阶段，是固定的，但是，由于后面具有注意力机制的缘故，使得GPT可以具有依据上下文改变最终token对应向量数值的能力，这也被称为动态词嵌入。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207205353.png)

更通俗来说，word2vec的方法，更像是训练出来一本词典，可以将不同句子中的各个token映射成不同的词向量表示，以供后面的
翻译模型、情感预测模型等去进行使用。


## 2-Positional Encoding

### 2.1-需要位置编码的原因

在 Transformer 中，输入序列是通过**自注意力机制（Self-Attention）进行处理的。然而，自注意力机制本身没有序列顺序信息**，它在计算注意力权重时，<font color='red'><b>所有的token是并行处理的</b></font>，不像 RNN 那样依赖前后时间步的信息。因此，Transformer 需要一种方式来**注入位置信息**，否则模型无法区分 **“I love NLP”** 和 **“NLP love I”** 的区别。

**解决方案**：Transformer 在输入词嵌入（Word Embedding）后，额外加入**位置编码（Positional Encoding）**，让模型知道每个单词的位置信息。

### 2.2-位置编码

Transformer 采用了一种**固定的、可解析的三角函数编码方式**，具体公式如下：
$$f(t)=
\begin{cases}
\sin\left(\frac{1}{10000^{2\cdot i/D}}\cdot t\right), & \text{如果}d=2i \\
\cos\left(\frac{1}{10000^{2\cdot i/D}}\cdot t\right), & \text{如果}d=2i+1 & 
\end{cases}$$
$$PE(pos, 2i) = \sin(pos / 10000^{2i/d})$$
$$PE(pos, 2i+1) = \cos(pos / 10000^{2i/d})$$
其中：
- $pos$ ：单词在序列中的位置（Position）。
- $i$ ：向量的维度索引（即嵌入维度的某个索引）。
- $d$ ：词向量的维度（如 512）。
- **偶数索引维度使用正弦函数（sin），奇数索引维度使用余弦函数（cos）**。

这个设计的核心思想：
• **不同位置的编码是唯一的**，让模型能够区分不同位置的单词。
• **相近位置的编码值相似**，方便模型学习局部依赖关系。
• **远距离的单词编码仍然有区分度**，利于长程依赖建模。

最终，将会输出该token的位置编码的信息，该信息是一个一维数组，该数组的长度是该token所对应的词向量空间的维度大小。

**位置编码的特点**
1. **固定不变（Fixed, Not Learned）**
	- 位置编码是**手工设计的公式**，不是通过训练得到的参数，因此不随训练数据而变化。
2. **连续可微（Continuous & Differentiable）**
	- 三角函数的平滑变化使得相近位置的编码值相似，这对注意力机制的学习很有帮助。
3. **长短序列都适用（Scalable）**
	- 由于使用的是数学公式，不论输入句子的长度如何，都可以直接计算位置编码，而不需要额外的训练。

### 2.3-位置编码和词嵌入融合

在 Transformer 的输入层，最终输入是：
$$\text{Input} = \text{Word Embedding} + \text{Positional Encoding}$$
即：
- **词嵌入（Word Embedding）** 提供语义信息。
- **位置编码（Positional Encoding）** 提供位置信息。

通过相加（Element-wise Addition），词嵌入会被**轻微偏移**，从而包含位置信息，使得 Transformer 能够区分不同的单词顺序。

## 3-Attention

注意力机制的核心思想是：**让模型在处理某个位置的信息时，能够动态地关注输入序列中不同位置的相关信息**。也就是说，在进行词嵌入，将词转换为词向量后，对词和词之间的语义进行理解，就需要依赖注意力机制去进行解决了。 类似于人类阅读时，会根据当前内容自动聚焦到关键词语或上下文。例如，在翻译句子时，生成某个目标词可能需要重点关注原句中的某些词。

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207192443.png)

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207210915.png)

三个可调整权重矩阵的作用：
- $W_q$  负责提取查询信息（用于计算注意力）。
- $W_k$  负责提取键信息（用于匹配查询计算权重）。
- $W_v$  负责提取值信息（用于加权求和得到最终输出）。
它们共同决定了 Transformer <font color='red'><b>在注意力机制中如何选择和聚合信息</b></font>。

注意力机制通过 **查询-键-值(Query-Key-Value, QKV)** 的三元组实现。以下是标准计算步骤：
1. **输入表示**：
    - 输入序列的每个元素（如词向量）通过线性变换生成三个矩阵：
        - **Query（Q）**：当前需要计算注意力的位置
        - **Key（K）**：用于被查询的键
        - **Value（V）**：实际被提取的信息
2. **计算注意力分数**：
    - 通过Query和Key的点积计算相似度，得到注意力分数：
        $Attention Score=Q⋅K^T$
    - 使用缩放因子（Scaled Dot-Product）避免点积过大：
        $Scaled Score=\frac{Q⋅K^T}{\sqrt{d_k}}(d_k\text{是Key的维度})$
3. **Softmax归一化**：
    - 将分数转换为概率分布（注意力权重）： 按照Q矩阵的词向量行(列)进行softmax操作
        $Attention Weights=Softmax(Scaled \, Score)$
4. **加权求和**：
    - 用注意力权重对Value加权求和，得到最终输出：
        $Output=Attention \, Weights⋅V$
        ![400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207212351.png)


### 3.1-Self-Attention

Query、Key、Value均来自同一输入序列。用于捕捉序列内部的长距离依赖关系（如句子中词与词的关系）。

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207193857.png)

自注意力用于让模型在处理输入序列的某个位置时，动态关注同一序列中其他位置的关联信息。 

假设输入序列为矩阵 \($X \in \mathbb{R}^{n \times d}$\)（n 为序列长度，d 为特征维度）：  
1. **生成Q、K、V**：  
   $Q = X W^Q, \quad K = X W^K, \quad V = X W^V$ 
   其中 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ 是可学习参数矩阵。
2. **计算注意力分数**：  
   $\text{Attention Score} = \frac{Q K^T}{\sqrt{d_k}}$ （缩放点积注意力，避免梯度消失）
3. **Softmax归一化**：  
   $\text{Attention Weights} = \text{Softmax}(\text{Attention Score})$
4. **加权求和**：  
   $\text{Output} = \text{Attention Weights} \cdot V$



### 3.2-Cross-Attention

Query来自一个序列，Key和Value来自另一个序列。常用于编码器-解码器结构（如机器翻译中，解码器关注编码器的输出）。

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207194139.png)

交叉注意力用于**两个不同序列之间的交互**，例如在编码器-解码器结构中，解码器生成目标序列时动态关注编码器的输出。

假设编码器输出为 $H_{\text{enc}} \in \mathbb{R}^{m \times d}$，解码器输入为 $H_{\text{dec}} \in \mathbb{R}^{n \times d}$：  
1. **生成Q、K、V**：  
   $Q = H_{\text{dec}} W^Q, \quad K = H_{\text{enc}} W^K, \quad V = H_{\text{enc}} W^V$
   （注意：Q来自解码器，K和V来自编码器）
2. 后续步骤与自注意力相同：  
   - 计算缩放点积分数 → Softmax → 加权求和。


### 3.3-Multi-Head Attention

**多头注意力（Multi-Head Attention, MHA）** 是 Transformer 的核心机制之一，它是 **自注意力（Self-Attention）** 的扩展版本，使模型能够从多个角度关注不同的部分，提高模型的表达能力。相比于单一注意力头，多头注意力可以更好地捕捉不同级别的语义信息，并增强 Transformer 在复杂任务上的表现。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207203517.png)

值得注意的是：在**多头注意力**中，不止使用一个注意力计算，而是使用  h **个并行的注意力头**。<font color='red'><b>每个注意力头有自己独立的权重</b></font>：

$$Q_i = X W_{Q_i}, \quad K_i = X W_{K_i}, \quad V_i = X W_{V_i}  \quad (i = 1,2,…,h)$$

然后，每个头独立计算注意力：

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

最终，我们将所有  h  个头的输出拼接（Concat），并通过一个线性变换  $W_O$  进行融合：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, …, \text{head}_h) W_O$$

• **Concat 操作**：将所有注意力头的结果拼接，形状变为  $(L, h \cdot d_k)$ 。
• **线性变换** $W_O$ ：降维到  $d_{model}$  维度，使得多头注意力的输出形状与输入相匹配。

最终，经过多头注意力等步骤的一个词向量值为：

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207213217.png)


### 3.4-Masked-Attention

**掩码机制（Masking）** 在 Transformer 中用于**控制注意力计算范围**，防止模型关注无关或未来的信息。

**填充掩码（Padding Mask）** 主要用于编码器和解码器，屏蔽 \<PAD\> 位置，确保注意力机制不会计算填充部分的权重，从而避免无效计算和错误学习。
- 在 NLP 任务中，句子长度通常不同，因此需要**填充（Padding）** 较短的句子，以匹配批处理（Batch Processing）的固定长度。
- 填充 \<PAD\> 只是为了对齐句子长度，**它们不应影响注意力计算**，因此，需要进行掩码的屏蔽。
- 填充掩码的核心思想是：**在计算注意力时，将填充部分的注意力分数设为极小值（如 -∞），使得 softmax 归一化后它们的权重接近 0**。

**未来掩码（Look-Ahead Mask）** 主要用于解码器，防止模型在训练时看到未来的单词，确保解码过程具有自回归特性，即只能基于当前和过去的信息预测下一个单词。
- Transformer 解码器是**自回归（Auto-Regressive）模型，它在推理时逐步生成下一个单词。在训练时，如果解码器能够看到完整的目标句子**，那么它可以“作弊”，提前知道答案。因此，**必须屏蔽未来的信息，使得模型只能看到当前和之前的单词**。
- **目标**：只允许模型关注当前及之前的单词，而不能关注未来单词。通过构造**一个下三角矩阵**，屏蔽所有未来的位置。在注意力计算中，**对未来位置添加极大负值，使得 softmax 后的注意力权重趋于 0**。


## 4-Feed Forward（全连接神经网络）

在 **Transformer** 架构中，**前馈神经网络（Feed Forward, FFN）** 是每个 **Transformer 层**（Encoder/Decoder 层）中自注意力机制后的关键组成部分，作用是对每个词向量的表示进行进一步的<font color='red'><b>非线性变换</b></font>和<font color='red'><b>特征提取</b></font>。

**前馈网络（FFN）的计算公式**
每个位置的输入向量 经过前馈神经网络的计算如下：
  
$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1) W_2 + b_2$$
其中：
- $W_1 \in \mathbb{R}^{d \times d_{ff}} ， b_1 \in \mathbb{R}^{d_{ff}}$ ：第一层的权重和偏置，负责将输入维度d投影到较高的维度$d_{ff}$ 。
-  $W_2 \in \mathbb{R}^{d_{ff} \times d} ， b_2 \in \mathbb{R}^{d}$ ：第二层的权重和偏置，将高维度特征映射回原始维度。
- **ReLU（或 GELU）** 是非线性激活函数，用于引入非线性表达能力。
在标准 Transformer（如 BERT、GPT、T5）中， 通常设定为 ，即前馈层的隐藏维度是输入维度的 **4 倍**。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207214022.png)

**前馈网络（FFN）的作用**
1. **增强表达能力**：
	- 自注意力机制（Self-Attention）主要用于信息交互，但其本质上仍是加权和操作，主要作用是建模 **不同 token 之间的关系**。
	- **FFN 通过非线性变换引入更强的特征表达能力**，使模型可以学习**更复杂的表示**，从而增强单个 token 的信息。
2. **局部特征提取**：
	- 虽然自注意力机制可以建模远程依赖关系，但它主要基于输入的加权组合，没有额外的特征提取能力。
	- **FFN 类似于 CNN 的作用，能够提取局部特征和模式，提高模型的表现力。**
3. **提升计算效率**：
	- 由于 **FFN 在每个 token 上的计算是独立的**（即 **逐 token 计算**，而不是像注意力机制那样涉及所有 token 之间的计算），它可以 **高度并行化**，加快训练速度。

FFN 结合 **自注意力（Self-Attention）**，使 Transformer 既能全局建模 token 之间的交互关系，又能局部提取复杂特征，从而提升模型的整体表现力。

## 5-Softmax

![[2-Softmax函数]]


## 6-Transform的训练和推理

### 6.1-训练

![image.png|center|200](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207200717.png)

Transformer 训练采用**自监督学习**，首先将输入文本转换为 token，并通过**词嵌入+位置编码**处理后送入多层**自注意力机制（Self-Attention）和前馈网络（FFN）**。在编码器中，输入序列被转换为上下文相关的表示；解码器则通过**掩码注意力（Masked Attention）** 和**交叉注意力（Cross-Attention）** 逐步生成目标序列。训练过程中，使用**交叉熵损失（Cross-Entropy Loss）** 计算预测输出与真实输出的误差，并通过**反向传播（Backpropagation）** 和**优化器（如 AdamW）** 更新参数，以最小化损失函数，使模型学习到更好的语言表示。

### 6.2-推理

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207201152.png)

Transformer 在推理时只执行**前向传播**，即输入经过编码器后，解码器通过**自回归（Autoregressive）方式<font color='red'><b>逐步生成输出序列</b></font>。在每一步，解码器根据当前已生成的部分，结合编码器的上下文信息，预测下一个单词的概率分布，并选取最优的词进行续写。常用的推理策略包括贪心搜索（Greedy Decoding）**、**Beam Search** 和 **Top-k/Top-p 采样**，以生成流畅且符合上下文的文本。



















