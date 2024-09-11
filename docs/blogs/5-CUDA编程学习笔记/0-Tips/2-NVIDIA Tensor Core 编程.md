## 1-概述

自 Volta 架构以来，NVIDIA Tensor core 一直致力于通用矩阵乘法(GEMM)操作的加速器。由于人工智能计算通常以 GEMM 运算为主，因此 NVIDIA 张量核对于加速人工智能应用至关重要。

## 2-CUDA Cores

在讨论Tensor core 的架构和实用性时，首先需要提及CUDA Cores的话题。CUDA（Compute Unified Device Architecture）是NVIDIA专有的并行处理平台和GPU API，而CUDA核心是NVIDIA显卡中的标准浮点单元。作为NVIDIA GPU微体系结构的一个定义特征，这些已经存在于过去十年发布的每个NVIDIA GPU中。每个CUDA核心能够执行计算，并且每个CUDA核心可以在一个时钟周期内执行一次操作。虽然不如CPU核心功能强大，但当用于深度学习时，许多CUDA核心可以通过并行执行进程加速计算。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240630233825.png)
<center> <font face='华文宋体' size='4'> 图 1 ：CUDA 上的处理流程 </font> </center>

在Tensor Cores发布之前，CUDA核心是加速深度学习的关键硬件。由于它们每个时钟周期只能处理一个计算，仅限于CUDA核心性能的GPU也受到可用CUDA核心数量和每个核心的时钟速度的限制。为了克服这一限制，NVIDIA开发了Tensor Core。
- 如果使用CUDA Core 执行16\*16的矩阵计算，即使在完全并行的情况下，也需要4个时钟周期（一个线程负责计算结果矩阵中的一个结果，该线程要计算4次乘法和四次加法。）
- 如果使用Tensor Core 执行16\*16的矩阵计算，则可以在一个时钟周期内，执行完所有需要的乘法和加法，输出最终的结果。

使用CUDA core 计算矩阵乘法的伪代码：
```cpp
__global__ void matMulCUDA(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 计算当前线程对应的矩阵行
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程对应的矩阵列
    float value = 0;
    if (row < N && col < N) { // 确保线程索引在矩阵范围内
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col]; // 计算矩阵乘法的点积
        }
        C[row * N + col] = value; // 存储计算结果
    }
}
```

使用Tensor core 计算矩阵乘法的伪代码：
```cpp
#include <mma.h>
using namespace nvcuda::wmma;
__global__ void matMulTensorCore(half *A, half *B, float *C, int N) {
    // 定义fragment矩阵
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    // 加载数据到fragments
    load_matrix_sync(a_frag, A, N);
    load_matrix_sync(b_frag, B, N);
    fill_fragment(c_frag, 0.0f);
    // 执行矩阵乘法
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    // 存储结果
    store_matrix_sync(C, c_frag, N, mem_row_major);
}
```

## 2-Tensor Core

NVIDIA Tensor Cores 公司专门从事混合精度的 GEMM 操作，即 GEMM 输入矩阵的精度较低，而 GEMM 输出矩阵的精度较高。张量核心是专门的核心，可以实现混合精度训练。第一代这些专用核心通过融合乘加计算来实现这一点。这允许两个4 x 4 FP16矩阵相乘并添加到一个4 x 4 FP16或FP32矩阵中。混合精度计算之所以被命名为如此，是因为虽然输入的矩阵可以是低精度的FP16，但最终输出将是FP32，在输出中只有极小的精度损失。实际上，这会快速加速计算，并对模型的最终有效性几乎没有负面影响。随后的微体系结构已经扩展了这种能力，甚至支持更不精确的计算机数字格式！

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240701003416.png)
<center> <font face='华文宋体' size='4'> 图2 ：每一代 GPU 支持不同精度的表格比较 </font> </center>

### 2-1 Tensor Core 的工作原理

#### 第一代Tensor Core （4\*4）

第一代Tensor Core是由NVIDIA在其Volta架构的GPU中引入的，具体是在2017年发布的Tesla V100 GPU中首次亮相。
1. **硬件设计和目标**：
   - 第一代Tensor Core专门设计用于加速深度学习中的矩阵运算，特别是矩阵乘法累加（Matrix Multiply and Accumulate，MMA）操作。
   - 其目标是提高神经网络训练和推理的效率，满足深度学习模型日益增长的计算需求。
2. **计算能力**：
   - **每个Tensor Core可以执行4x4矩阵的乘法累加操作**，即 $D = A \times B + C$，其中 $A$、$B$是4x4的半精度浮点数矩阵（FP16），而 $C$ 和结果 $D$ 是单精度浮点数矩阵（FP32）。
   - 这种设计允许Tensor Core在保持较高计算精度的同时，通过混合精度计算显著提高性能。
<div style={{ textAlign:'center' }}>
  <video width="50%" height="50%" controls autoplay loop>
    <source src="https://blog.paperspace.com/content/media/2022/05/ezgif.com-gif-maker.mp4" type="video/mp4" />
  </video>
</div>
<center> <font face='华文宋体' size='4'> 视频1：第一代具有tensor core的GPU运算方式 </font> </center>
3. **高效计算**：
   - <font color='red'><b>第一代Tensor Core能够在一个时钟周期内完成一次4x4矩阵乘法累加操作</b></font>。
   - 通过大量并行Tensor Core的协同工作，Volta GPU显著提高了深度学习任务中的矩阵运算效率。
4. **硬件支持**：
   - Tesla V100 GPU包含640个Tensor Core，每个SM（Streaming Multiprocessor）中有8个Tensor Core。
   - 这些Tensor Core与CUDA Core、共享内存和其他GPU资源紧密集成，支持高效的数据传输和计算。

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240630155501.png)
<center> <font face='华文宋体' size='4'>  图 3 ：Tensor core 的 GEMM 示意图 </font> </center>
![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240630200620.png)
<center> <font face='华文宋体' size='4'> 图 4 ： 2*4矩阵与4*2的矩阵相乘，在tensor core中的示意图 </font> </center>

由于 NVIDIA Tensor Core 是专门为 **GEMM** 设计的，<font color='red'><b>使用 NVIDIA Tensor Core 的 GEMM 吞吐量远远高于使用 NVIDIA CUDA Core 所能达到的吞吐量</b></font>，CUDA Core 更适合于更一般的并行编程。

#### 第二代Tensor Core（4\*4、8\*8、16\*16）

第二代Tensor Core是在NVIDIA的Turing架构（发布于2018年）中引入的，并且在之后的Ampere架构中得到了进一步改进。
主要特性和改进:
1.	支持更多的数据类型：
	- 第二代Tensor Core不仅支持FP16，还引入了对INT8和BF16（Brain Floating Point 16）的支持。
2.	混合精度计算：
	- 第二代Tensor Core继续支持混合精度计算，使用FP16进行计算并在累加时转换为FP32。这种方式能够在保持计算精度的同时提高性能。
<div style = {{ textAlign:'center' }} ><video width="50%" height="50%" position="center" controls autoplay loop><source src="https://blog.paperspace.com/content/media/2022/05/ezgif.com-gif-maker--1--1.mp4" type="video/mp4" /></video></div>
<center> <font face='华文宋体' size='4'> 视频2：矩阵A B相乘示意图 </font> </center>
3.	性能提升：
	- 第二代Tensor Core在每个时钟周期内可以执行更多的运算，从而提高了整体计算性能。
4.	灵活的矩阵尺寸支持：
	- 第二代Tensor Core支持更灵活的矩阵尺寸，不仅限于4x4矩阵操作。（还可以有8\*8、16\*16等）

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240630155922.png)
<center> <font face='华文宋体' size='4'> 图 5：NVIDIA GEMM 吞吐量 Turing Tensor Core VS Pascal CUDA Core </font> </center>

#### 第三代Tensor Core

第三代Tensor Core是在NVIDIA的Ampere架构（发布于2020年）中引入的。与前两代相比，第三代Tensor Core在性能、支持的数据类型和操作灵活性上都有显著提升。以下是第三代Tensor Core的主要特性和改进：

主要特性和改进

1.	更多的数据类型支持：
	- 第三代Tensor Core支持更多的数据类型，包括FP64（双精度浮点数）。这使其在科学计算和高性能计算（HPC）任务中更加实用。
	- 支持FP16、BF16、TF32、INT8等多种数据类型，为深度学习和其他计算任务提供了更高的灵活性和性能。
2.	更高的计算性能：
	- 每个Tensor Core在一个时钟周期内可以执行更多的运算，进一步提高了计算密度和性能。
	- 引入了TF32（TensorFloat-32）格式，这是一种专为深度学习设计的浮点格式，能够在不显著影响模型精度的情况下显著提高训练速度。
3.	增强的计算能力：
	- Ampere架构中的第三代Tensor Core可以在一个时钟周期内执行更多的矩阵乘法累加操作，特别是对更大规模的矩阵支持更好。
	- <font color='red'><b>每个Tensor Core在一个时钟周期内可以执行一次16x16矩阵乘法累加操作</b></font>。

对于 NVIDIA 安培架构，每个 SM 有4个张量核心。特别是，NVIDIA A100图形处理器拥有108个流式多处理器(SM) ，总共拥有432个张量核心。

![center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240630161404.png)
<center> <font face='华文宋体' size='4'> 图 6：NVIDIA GA100 Full GPU with 128 SMs </font> </center>

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240630161704.png)
<center> <font face='华文宋体' size='4'> 图7：每个 NVIDIA Ampere SM 有4个Tensor Core </font> </center>

NVIDIA 张量核心完全可编程。Tensor Core 编程 API 已经在 `nvcuda: : wmma` 名称空间下的 mma.h 头部声明。

#### 第四代Tensor Core
NVIDIA在其最新的Hopper架构（2022年发布）中引入了第四代Tensor Core。第四代Tensor Core在每个时钟周期内可以执行更大规模的矩阵乘法累加操作。
主要特性和改进
1.	支持更大规模的矩阵计算：
	- 第四代Tensor Core可以处理大小为16x16的矩阵乘法累加操作，这意味着在一个时钟周期内，它可以执行16x16的矩阵乘法累加操作。
2.	更高的计算性能：
	- 每个Tensor Core在一个时钟周期内可以执行更多的运算，从而显著提高计算密度和性能。
	- 引入了更高效的处理器架构和优化算法，以进一步加速深度学习训练和推理任务。
3.	支持多种数据类型：
	- 第四代Tensor Core继续支持多种数据类型，包括FP16、BF16、TF32、INT8和FP64，使其在深度学习、科学计算和高性能计算（HPC）任务中更加灵活和高效。
	
<div style={{ textAlign:'center' }}>
  <table>
    <tr>
      <th>代数</th>
      <th>架构名称</th>
      <th>一次矩阵乘法累加计算的矩阵大小</th>
    </tr>
    <tr>
      <td>第一代</td>
      <td>Volta (V100)</td>
      <td>4x4</td>
    </tr>
    <tr>
      <td>第二代</td>
      <td>Turing (T4)</td>
      <td>16x16</td>
    </tr>
    <tr>
      <td>第三代</td>
      <td>Ampere (A100)</td>
      <td>16x16</td>
    </tr>
    <tr>
      <td>第四代</td>
      <td>Hopper (H100)</td>
      <td>16x16</td>
    </tr>
  </table>
</div>

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240701012102.png)
<center> <font face='华文宋体' size='4'> 图8 ：各个架构Tensor Core支持的类型 </font> </center>
## 3-Tensor Core编程

NVIDIA CUDA允许用户在warp级别上编程Tensor Core GEMM操作$D=AB+C$。虽然每个Tensor Core只能针对不同数据类型的一些特定小尺寸执行矩阵乘法，但是大型GEMM可以分解为多个小型GEMM和累积。


一个简单的两个矩阵相乘的Demo。
```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
using namespace std;
using namespace nvcuda::wmma;
const int M = 16;
const int N = 16;
const int K = 16;
__global__ void matrixMultiplyTensorCore(__half *a, __half *b, float *c, int m, int n, int k) {
    fragment<matrix_a, M, N, K, __half, row_major> a_frag;
    fragment<matrix_b, M, N, K, __half, col_major> b_frag;
    fragment<accumulator, M, N, K, float> c_frag;
    fill_fragment(c_frag, 0.0f);
    load_matrix_sync(a_frag, a, k);
    load_matrix_sync(b_frag, b, k);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(c, c_frag, n, mem_row_major);
}
int main() {
    __half *h_a = new __half[M * K];
    __half *h_b = new __half[K * N];
    float *h_c = new float[M * N];
    for (int i = 0; i < M * K; ++i) {
        float x = (float)rand() / RAND_MAX;
        h_a[i] = __float2half(x);
    }
    for (int i = 0; i < K * N; ++i) {
        float x = (float)rand() / RAND_MAX;
        h_b[i] = __float2half(x);
    }
    __half *d_a, *d_b;
    float *d_c;
    cudaMalloc((void **)&d_a, M * K * sizeof(__half));
    cudaMalloc((void **)&d_b, K * N * sizeof(__half));
    cudaMalloc((void **)&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice);
    dim3 gridDim(M*N*K/256);
    dim3 blockDim(256);
    matrixMultiplyTensorCore<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << h_c[i * N + j] << " ";
        }
        cout << endl;
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}
```

计算两个大型矩阵相乘的思路，将矩阵分成多个小块进行相乘。

给定一个GEMM操作 $D = AB + C$，其中 $D \in \mathbb{R}^{m \times n}$， $A \in \mathbb{R}^{m \times k}$， $B \in \mathbb{R}^{k \times n}$， $C \in \mathbb{R}^{m \times n}$，这些矩阵可以被划分为更小的矩阵。

$$
A =
\begin{bmatrix}
A^{d \times d}_{1,1} & A^{d \times d}_{1,2} & \cdots & A^{d \times d}_{1,k/d} \\
A^{d \times d}_{2,1} & A^{d \times d}_{2,2} & \cdots & A^{d \times d}_{2,k/d} \\
\vdots & \vdots & \ddots & \vdots \\
A^{d \times d}_{m/d,1} & A^{d \times d}_{m/d,2} & \cdots & A^{d \times d}_{m/d,k/d}
\end{bmatrix}
$$

$$
B =
\begin{bmatrix}
B^{d \times d}_{1,1} & B^{d \times d}_{1,2} & \cdots & B^{d \times d}_{1,n/d} \\
B^{d \times d}_{2,1} & B^{d \times d}_{2,2} & \cdots & B^{d \times d}_{2,n/d} \\
\vdots & \vdots & \ddots & \vdots \\
B^{d \times d}_{k/d,1} & B^{d \times d}_{k/d,2} & \cdots & B^{d \times d}_{k/d,n/d}
\end{bmatrix}
$$

$$
C =
\begin{bmatrix}
C^{d \times d}_{1,1} & C^{d \times d}_{1,2} & \cdots & C^{d \times d}_{1,n/d} \\
C^{d \times d}_{2,1} & C^{d \times d}_{2,2} & \cdots & C^{d \times d}_{2,n/d} \\
\vdots & \vdots & \ddots & \vdots \\
C^{d \times d}_{m/d,1} & C^{d \times d}_{m/d,2} & \cdots & C^{d \times d}_{m/d,n/d}
\end{bmatrix}
$$

$$
D =
\begin{bmatrix}
D^{d \times d}_{1,1} & D^{d \times d}_{1,2} & \cdots & D^{d \times d}_{1,n/d} \\
D^{d \times d}_{2,1} & D^{d \times d}_{2,2} & \cdots & D^{d \times d}_{2,n/d} \\
\vdots & \vdots & \ddots & \vdots \\
D^{d \times d}_{m/d,1} & D^{d \times d}_{m/d,2} & \cdots & D^{d \times d}_{m/d,n/d}
\end{bmatrix}
$$

$D$ 中的每个小矩阵被计算为多个小 GEMM 和累加。因此，有以下的公式：

$$
D^{d \times d}_{i_m, i_n} = \sum_{i_k=1}^{k/d} A^{d \times d}_{i_m, i_k} B^{d \times d}_{i_k, i_n}
$$


## 参考文献

【1】  [NVIDIA Tensor Core Programming](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)

【2】 [Understanding Tensor Cores](https://blog.paperspace.com/understanding-tensor-cores/)
