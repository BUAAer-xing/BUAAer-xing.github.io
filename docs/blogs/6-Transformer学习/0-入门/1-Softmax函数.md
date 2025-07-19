
Softmax 函数是一种广泛用于深度学习和机器学习中的激活函数，通常用于多分类问题的输出层。它的主要作用是将一个实数向量转换为概率分布，使得输出的各个分量在 $[0,1]$ 之间，并且所有分量的和为 1。

![image.png|center|500](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250207170456.png)

前面的输入一般称为Logits，后面的输出一般称为Probabilities。

## 1-Softmax 公式

对于输入向量 $\mathbf{z} = [z_1, z_2, …, z_n]$，Softmax 函数的计算公式为：

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$
其中：
- $z_i$ 是输入向量中的第 i 个元素。
- $e^{z_i}$ 代表指数运算，确保所有数值为正。
- 分母 $\sum_{j=1}^{n} e^{z_j}$ 负责归一化，使得所有输出的和为 1。

## 2-Softmax 的性质

1. **归一化性质**：Softmax 计算后的输出满足概率分布性质，即所有元素都在 (0,1) 之间，且总和为 1。
2. **单调性**：如果 $z_i > z_j$，则 $\text{Softmax}(z_i)$ 也大于 $\text{Softmax}(z_j)$。
3. **对数几何解释**：Softmax 的结果可以被视为一种指数权重的归一化处理，使得较大的输入值对应更大的概率。
	- 它使最大值最接近于1，而较小的数值则会非常接近于0。


## 3-带有温度的Softmax函数

带有温度参数 T（Temperature）的 Softmax 具有不同的数值特性，温度参数用于调整 Softmax 的输出分布，使其更加平滑或更加尖锐。
$$\text{Softmax}T(z_i) = \frac{e^{z_i / T}}{\sum{j=1}^{n} e^{z_j / T}}$$
其中：
- $T > 0$ 是温度参数（Temperature）。
- 当 $T = 1$ 时，该公式与标准 Softmax 完全一致。
- 当 $T > 1$ 时，Softmax 输出更加<font color='red'><b>平滑</b></font>，所有类别的概率更加接近。
- 当 $T < 1$ 时，Softmax 输出更加<font color='red'><b>尖锐</b></font>，概率集中在值较大的类别上。

### 应用场景

1. 知识蒸馏（Knowledge Distillation）
	- 训练教师模型时使用较高温度 T 生成较平滑的概率分布，使学生模型能够学习到更丰富的信息，而不是简单地模仿 one-hot 标签。
2. 强化学习（Reinforcement Learning）:在策略梯度方法（Policy Gradient）中，温度参数 T 控制策略的探索和利用程度：
	- **高温度**（T > 1）时，策略更随机，鼓励探索不同的动作。
	- **低温度**（T < 1）时，策略更确定，倾向于选择高概率的动作。
3. 自回归文本生成：在 Transformer 语言模型（如 GPT）中，调整 T 可以控制生成文本的多样性：
	- **高温度（T > 1）**：增加生成的随机性，使文本更具创造性但可能不连贯。
	- **低温度（T < 1）**：减少随机性，使文本更确定，但可能变得刻板。




