
## 0-概述

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250509111335.png)

DeepGEMM 是 DeepSeek 团队为 NVIDIA Hopper 架构（如 H100/H800）GPU 优化的高性能 <font color='red'><b>FP8 通用矩阵乘法</b></font>（GEMM）库，旨在提升大规模 AI 模型，特别是混合专家（MoE）模型的训练和推理效率。其主要优化措施包括：
- 首先，DeepGEMM 采用了轻量级的 Just-In-Time（JIT）编译机制，在运行时根据具体的矩阵形状动态生成和优化 CUDA 内核，避免了传统模板库的复杂性，提高了开发和部署的灵活性。该库仅包含约 300 行核心内核代码，便于理解和维护。
- 其次，为了充分利用 Hopper 架构的 Tensor Core，**DeepGEMM 实现了双层累加机制，结合 CUDA 核心和 Tensor Core 的计算能力，提升了 FP8 低精度计算的数值稳定性和准确性**。此外，库中还引入了精细化的缩放策略，对每 128 通道进行独立缩放，进一步增强了低精度计算的鲁棒性。
- 在内核优化方面，DeepGEMM 对 **FFMA（Fused Multiply-Add）** 指令进行了深入分析和调整，通过修改 <font color='red'><b>SASS 汇编</b></font>中的 yield 和 reuse 位，实现了指令级的调度优化，提升了内核的执行效率。此外，**库中还采用了 Tensor Memory Accelerator（TMA）、软件流水线和 Warp 专用化**等技术，最大化地利用了内存带宽和计算资源。
- 针对 MoE 模型的特点，DeepGEMM 支持分组 GEMM 操作，包括连续分组（contiguous-grouped）和掩码分组（masked-grouped）等形式，优化了小批量矩阵乘法的性能，提升了专家模型的计算效率。在实际测试中，DeepGEMM 在 H800 GPU 上实现了高达 1550 TFLOPS 的计算性能，超过了许多手工优化的库。

DeepGEMM 通过 JIT 编译、双层累加、精细化缩放、指令级调度优化和 MoE 特化支持等多项优化措施，显著提升了 FP8 低精度矩阵乘法的性能和稳定性，为大规模 AI 模型的高效训练和推理提供了有力支持。

## 1-BG

### 1.1-JIT编译机制

JIT（Just-In-Time）编译技术是一种**运行时动态编译技术**，它在程序执行过程中将**中间代码（如字节码）即时编译为机器码**，并将其执行，以提升程序的执行效率。JIT 是静态编译（如 C++ 编译器）和解释执行（如传统的 Python/JavaScript 解释器）之间的一种折中方案，广泛用于 Java、.NET、PyPy、V8（JavaScript 引擎）等环境中。

在 JIT 编译中，源代码通常先被编译为中间表示（IR），如 Java 的字节码或 .NET 的 IL（Intermediate Language），然后在程序运行时，JIT 编译器根据程序执行的热点路径，将这些中间代码转换为机器码，并缓存起来以供重复使用。JIT 编译可以结合运行时信息（如分支频率、内联机会、缓存命中率）进行激进优化，因此比传统静态编译更能贴近实际运行时行为。

DeepGEMM 的 JIT 编译系统本质上是一个专门为 CUDA kernel 设计的**模板驱动的运行时代码生成与调度框架**，它通过模块化封装与编译器抽象，实现在 PyTorch 中动态生成、编译和调度 CUDA 内核函数。这使其适合用于 **自动调优 kernel、FP8/BF16 混合精度优化、多流 pipeline 调度等高性能计算场景**。

---

影响的参数，比如：

这些参数决定生成的 CUDA C++ 源代码内容，属于内核逻辑的“模板参数”：
- **T**：如 `float`, `__nv_bfloat16`, `__nv_fp8_e4m3`等，控制内核使用的数据类型。用于生成 vector_add\<T\> 这样的模板内核。
- **问题规模参数（如 M, N, K）**：在 GEMM 场景下常用于内核模板参数，影响线程布局或分块策略。
- **内核变种标签（如是否使用 swizzle、pipeline）**：可作为宏定义传入，用于条件编译以开启不同内核特性。
👉 **作用**：控制生成的 CUDA 源码结构和功能，直接影响内核性能与适配性。

---

所有内核运行时编译，无需预编译：
- GEMM 形状、block 大小、流水线 stage 为编译期常量，减少寄存器占用
- 自动选择配置（无 auto-tuning，但选择是确定性的）
- MMA pipeline 全展开，极大提升小矩阵性能


### 1.2-Hopper架构新机制

**TMA（Tensor Memory Accelerator）**： 在 CUDA 的 **Hopper 架构中**，TMA（Tensor Memory Accelerator）是一种新引入的硬件功能，旨在实现全局内存与共享内存之间的高效异步张量数据传输。与传统使用 cp.async 的方式相比，TMA 支持由单线程发起的数据搬运操作，支持最高 5 维张量，通过复制描述符完成地址计算、边界检查等繁重工作，显著简化了编程模型并降低了资源占用。同时，TMA 支持线程块集群间共享内存通信，结合 CUDA 的异步屏障机制和流水线控制，可以实现计算与数据搬运的完全重叠，极大提升吞吐量，特别适用于大规模矩阵乘法、注意力机制等计算密集型任务。

**WG（Warp Group）**： Warp Group 是 NVIDIA 在 **Hopper 架构**中引入的一种由多个 warp（通常为 4 个，即 128 个线程）组成的线程协作机制，旨在实现更高效的资源共享与线程同步。相比传统 warp，Warp Group 支持跨 warp 的低开销通信、共享寄存器与共享内存，并引入新的同步指令如 `__wg_barrier()`，特别适用于与 TMA 协同处理大规模张量数据搬运与计算任务，如 FlashAttention 和 GEMM 等，在提升吞吐量和执行效率方面具有显著优势。


## 2-优化措施

### 2.1-分块FP8量化和分阶段累加

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250509111335.png)

<font color='red'><b>分块FP8量化</b></font>：
- **每个块内分别计算缩放系数进行量化**，缩放系数保存成矩阵/向量，用于将**计算结果反量化**。
- 改善权重矩阵outliner问题，将其影响控制在块内，从而提升量化精度。

<font color='red'><b>分阶段累加</b></font>：
- FP8矩阵乘结果累加存在精度问题，可通过两阶段计算改善精度问题。即在Tensor Core上累积了一定数量的结果后，转到高精度CUDA Core上进行计算，保证最终计算结果的精度，兼顾**矩阵乘性能和计算稳定性**。

### 2.2-warp-specialization

---
在 NVIDIA Hopper 架构中，**warp specialization** 是一种将线程块（thread block）中的不同 warps（每个 warp 包含 32 个线程）分配给不同任务的编程技术。这种方法**通过将数据加载（producer）和计算（consumer）任务分离到不同的 warps 中，并利用共享内存和低开销的同步机制（如 named barriers）进行协调，从而提高了内核的性能和资源利用率**。

在 Hopper 架构中，warp specialization 的优势尤为显著，主要体现在以下几个方面：
1. **异步执行模型**：Hopper 引入了 Tensor Memory Accelerator (TMA) 等硬件单元，支持异步的数据传输和计算操作。通过 warp specialization，可以将数据加载和计算任务分配给不同的 warps，使得这些任务能够并行执行，从而提高了整体吞吐量。
2. **高效的资源利用**：将不同任务分配给专门的 warps，可以减少资源争用，提高寄存器和共享内存的利用率。例如，数据加载 warps 可以专注于内存访问，而计算 warps 则专注于执行计算指令，从而避免了资源的相互干扰。
3. **简化的同步机制**：通过使用 named barriers 等同步原语，可以在不同的 warps 之间实现高效的同步，确保数据的一致性和正确性。这种机制比传统的全线程块同步（如`__syncthreads()`）更为高效，减少了同步开销。

相比之下，传统的 multi-stage 管线方法依赖于编译器生成的指令序列来实现数据加载和计算的重叠，这在处理复杂的内核（如 Flash Attention）时可能面临挑战。而<font color='red'><b>warp specialization 提供了更明确的任务分离和同步机制，使得内核设计更加灵活和高效</b></font>。

---

遵循 CUTLASS 的设计理念，**DeepGEMM 中的内核均为 warp 特化**，**支持数据搬运、张量核心 MMA 和 CUDA 核心 promotion 的重叠执行**，具体流程见简图：
![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250509141358.png)
- **TMA warps**：负责通过 **Tensor Memory Accelerator (TMA)** 发起并执行异步的数据加载（从 global memory 到 shared memory）。黄色框为 **TMA Issue**（发起加载），蓝色框为 **Data load**（实际搬运数据）。
- **Math warps 0/1**：用于执行计算（矩阵乘加操作），绿色框为 **WGMMA**（Warp Group Matrix Multiply Accumulate，Hopper 架构中新指令），黄色框为 **Promotion**，即从 shared memory 中读取数据并转移到寄存器中，准备执行计算。


### 2.3-Hopper架构的新特性

在 Hopper 架构中，<font color='red'><b>Tensor Memory Accelerator (TMA) 是专为高吞吐张量数据传输设计的硬件机制，旨在将全局内存（global memory）与共享内存（shared memory）之间的数据搬运变得高效、异步和线程块友好</b></font>。

在 Hopper 架构中，**Tensor Memory Accelerator (TMA)** 是专为高吞吐张量数据传输设计的硬件机制，旨在将全局内存（global memory）与共享内存（shared memory）之间的数据搬运变得高效、异步和线程块友好。以下是你列出的术语解释及其意义：

**LHS / RHS / 缩放因子的 TMA 加载**
- **LHS（Left-Hand Side）**：矩阵乘法中左侧输入矩阵（如 A in C = \alpha A B + \beta C）。通过 TMA 从 global memory 加载至 shared memory。
- **RHS（Right-Hand Side）**：右侧输入矩阵（如 B）。同样通过 TMA 加载。
- **缩放因子（scaling factor）**：指矩阵乘法中标量 \alpha、\beta，若其为张量形式或 broadcast 向量（如在 FlashAttention 中），也可能通过 TMA 加载。
这些加载通常由一个或多个 **TMA warps** 使用 tma.load 指令异步发起，最大化带宽利用。

**输出矩阵的 TMA 存储**
执行完计算后，结果矩阵（即 C）需要从 shared memory 写回 global memory。TMA 支持：
- **异步 store**：即计算 warp 完成后，交由 TMA warp 将结果块写入 global memory。
- 可实现 overlap（重叠）计算与写回，避免写操作阻塞计算。

**自动决定 LHS 或 RHS 的 TMA 多播（Multicast）**

在 Hopper 中，TMA 支持 **multicast** 模式 —— 一次加载，多个 CTA（线程块）共享。此机制下，硬件/编译器可根据矩阵访问模式决定是否：
- 对 **LHS** 或 **RHS** 启用 TMA multicast，从而节省 global memory 带宽；
- 将某一片共享内存广播到多个计算线程块。

例如在 Transformer 中，不同头（head）常共享某些矩阵块，这就特别适合多播。

 **TMA 描述符预取（TMA Descriptor Prefetch）**
 
TMA 操作依赖于描述符（descriptor）来定义加载的数据形状、对齐方式、步长等。
- Hopper 支持 **预取 TMA descriptor** 到 L1 cache，使得随后的 tma.load latency 更低；
- 通过 tma.prefetch 类指令实现，提升调度性能，尤其对大量小 TMA 事务效果显著。

### 2.4-常规优化

- 使用 [stmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix) 指令
    - stmatrix 的主要用途是 **将 MMA 运算后的子结果矩阵写入共享内存**，用于后续的矩阵合并、规约或从共享内存写回 global memory。使用它能保证写入操作和 Tensor Core 的 warp-level 分布相兼容，避免 warp 中不同线程对共享内存访问冲突。
- 针对不同 warpgroups 的寄存器控制
- 使用 3D TMA 或 swizzling 减少 **bank 冲突**
- 更大 block 大小（如 256×128 🐳）
- 最大化重叠，例如 TMA store 与非 TMA RHS 缩放加载 🐳

### 2.5-统一调度器

- 单一调度器用于所有分组/非分组内核
- 使用 [Rasterization](https://github.com/NVIDIA/cutlass/blob/eefa171318b79cbe2e78514d4cce5cd0fe919d0c/media/docs/efficient_gemm.md#threadblock-rasterization) 提升 L2 命中率
	- **Threadblock Rasterization** 是一种控制 **线程块如何映射到 GEMM 网格子问题** 的策略，目标是最大化缓存重用、提升带宽利用率。在 CUTLASS 中由 ThreadblockSwizzle 模板机制支持，能根据问题大小、tile shape 和并行划分策略动态选择最合适的调度方法。

### 2.6-FFMA SASS 优化 🐳

发现 CUTLASS FP8 kernel 从 NVCC 12.2 到 12.3 性能大幅提升，原因是某些 FADD 指令中 yield 位被交错置位。参考 [CuAssembler](https://github.com/cloudcores/CuAssembler)，发现此位或控制 warp 切换。我们开发了 [interleave 脚本](app://obsidian.md/deep_gemm/jit/interleave_ffma.py)，在二进制中修改 FFMA 指令，不仅置位 yield，还置位 reuse（避免 warp 切换时复用寄存器），对精细缩放场景性能提升明显（可达 10% 以上）。

## 3-DeepGEMM调用流程图

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250509145751.png)

1. TMA+multicast实现访存加速
	- 支持将数据从Global Memory复制到多个SM
2. PTX warp级 矩阵指令集 stmatrix
	- 加速结果传回Shared Mem
3. FFMA SASS interleaving
	- 编译后修改FFMA指令，提升性能

