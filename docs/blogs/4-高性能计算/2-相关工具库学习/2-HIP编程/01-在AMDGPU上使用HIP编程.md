# 在AMDGPU上使用HIP编程

本笔记为AMD官方教学视频的总结。

## AMD GPU Hardware

### 硬件概述

这一部分主要是介绍一些关于AMD GPU 硬件部分的一些知识，从而便于理解下一节中所介绍的一些编程概念。

#### GCN 硬件概述

AMD GPU 由一个或者多个Shader Engine 以及一个 Command Processor 组成。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240201204509.png)
<center> <font face='华文宋体' size='4'> 图1 AMD GPU 的大致构成 </font> </center>

并且，Shader Engine 进一步可以被细分为 Computer Unit 。

每个 Shader Engine 都有一个 Workload Manager 用于将计算工作分配给 Computer Unit，而Command Processor 负责从一个或者多个 Command Queue 中读取命令包，这些 Command Queue 通常驻留在用户可见的DRAM中，因此，无需执行操作系统级的内核调用。

（在这里解释一下为什么：**用户可见的DRAM是应用程序直接访问的内存区域**，将Command Queue放置在这里**允许应用程序直接管理和操作队列**，因此无需通过内核进行中介。这样可以避免在内核级别执行系统调用的开销）

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240201212058.png)
<center> <font face='华文宋体' size='4'> 图2 AMD GPU 的详细构成 </font> </center>

Command Processor 从 Command Queue 中读取命令，提交给 Workload Manager ，然后，由 Workload Manager 将计算任务分配给 Computer Unit。

下面是截止到2020年，一些常见的AMD的GPU上各个原件的数量。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240204141826.png)
<center> <font face='华文宋体' size='4'> 图3 AMD GPU上各个原件的数量 </font> </center>
除了最后一行以外，其它的都是独立显卡，最后一行是搭载集成显卡的CPU。

#### AMD GPU 计算机术语

##### GPU Kernel

**GPU kernel 是**用在GPU上<font color='red'><b>启动并执行多个并行任务或者线程</b></font>的**程序**，比如：GMEE（矩阵和矩阵乘法），triangular solve（三角求解器）， vector copy（向量复制），scan ，convolution等。

---

GPU kernel（图形处理器核心）是指在图形处理单元（GPU）上执行的一个小型计算程序或函数。GPU kernel 是在GPU上并行处理的任务的一部分，通常由程序员编写，以利用GPU的并行计算能力。

GPU kernel 是在数据并行计算模型中执行的，这意味着<font color='red'><b>相同的操作被应用于许多数据元素</b></font>，从而允许 GPU 高效地处理大规模数据集。

---

##### Workgroup （Thread Block）

 线程被划分到一个或者多个 Workgroup 中，**Workgroup 是同时位于GPU上的一组线程，Workgroup 中的所有线程也同时位于同一计算单元上，因此，可以使用该计算单元上的资源进行同步和通信**。（注：在CUDA术语中和Workgroup对应的概念是Thread Block）

---

在并行计算中，任务被划分为多个工作项（work item），而这些工作项被组织成工作组。工作组是在GPU设备上执行的一组相关的工作项，它们可以协同工作并共享局部内存。

具体来说，工作组是一组具有相同计算任务的工作项的集合。每个工作项代表一个独立的计算任务，而工作组则是这些工作项的集合。工作组内的工作项可以协同工作，它们可以通过共享局部内存进行通信和协同计算，以提高计算效率。

---

内核可以由一个或者多个Workgroup构成， Workgroup 的数量由程序员控制，通常由问题规模而定。 

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240204145412.png)
<center> <font face='华文宋体' size='4'> 图4 GPU Kernel 以及 Workgroup 的关系 </font> </center>

##### Wavefront （warp）

Workgroup可以进一步细分为多个 Wavefront ，**Wavefront是通过锁步方式执行相同指令并遵循相同控制流的资源或者通道集合**。 然而，单个通道可以被遮蔽，这样一个 Wavefront 内会存在不同的分支，可以把Wavefront想象成一个矢量化的线程。线程这个术语也用于单独的通道或者工作项。

在CUDA中，Wavefront 相当于 Warp（线程束）。

一个Workgroup可以由一个或者多个Wavefront组成，同样的，Wavefront的数量由程序员而定，AMD的硬件允许每个工作组最多有16个Wavefront。

（辨析一下：Workgroup是在同一个运算单元上的线程的集合，其内的线程并不一定都执行相同的指令。但是 Wavefront 是在同一个运算单元上，执行相同指令的但是操作数据可能不同的线程的集合。）

每个工作组拥有64个工作项、通道或者线程。

---

Wavefront（或 warp）是一组连续的线程（或工作项），它们在 GPU 上同时执行相同的指令。Wavefront 内的线程（或工作项）被分组到相同的处理单元中，以便能够以SIMD（单指令多数据）的方式执行相同的指令。

**每个 wavefront 内的线程（或 warp 内的线程）执行相同的指令，但可能处理不同的数据**。这样的设计可以充分利用 GPU 的并行性，提高计算效率。Wavefront 的大小是由硬件决定的，通常是一定数量的线程。

---

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240204153531.png)
<center> <font face='华文宋体' size='4'> 图 5 GPU Kernel 、Workgroup和Wavefront之间的关系 </font> </center>

#### GPU的调度方式

 通用处理器（CPU）从命令队列获取内核包，并创建工作组，然后将其分配给Shader Engine 上的 Workload Manager，然后，由Workload 为 
Workgroup 创建 Wavefront  ， 并将它们（Wavefront）发送到Compute Unit执行，（注意：**工作组中的所有Wavefront都在同一个Compute Unit上**）。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240204160318.png)
<center> <font face='华文宋体' size='4'> 图 6 GPU的调度方式 </font> </center>

### AMD GPU 体系结构
 
#### GPU 内存 和  I/O系统

与GPU内存相连的是HBM或GDDR内存，均可用于GPU上的所有计算单元，并通过内存控制器进行连接和控制。PCIe控制器用于连接主机内存，Infinity Fabric 控制器用于连接系统中的其他GPU，连接PCIe的Infinity Fabric 控制器或DMA引擎用于在主机存储器和设备存储器之间或多个设备之间进行异步传输。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240204175855.png)
<center> <font face='华文宋体' size='4'> 图 7 GPU内存和I/O系统框架图 </font> </center>

工作流程：
1. CPU向命令队列提交DMA传输块，命令队列处理器完成后，将其从命令队列中取出。
	- 同样的，这些命令队列位于用户可见的DRAM中，因此，无需操作系统级的内核调用。
2. Command Processor 读取传输块，并解析传输请求。
3. Command Processor 将其请求提交给DMA Engines。
	 - 这些工作都与CU的运行以及其他传输过程同时进行。
4. DMA Engines 负责系统内存到HBM之间的双向传输或设备之间的传输。

#### GCN 计算单元内部结构

##### Scalar Unit

每个 Compute Unit 有一个Scalar Unit（标量单元）：

**Scalar Unit（标量单元）**
- 由Wavefront中所有线程共享的处理单元
- 用于流控制以及指针计算
- 每个标量单元都有自己的Scalar General Purpose Registers（标量通用寄存器池），这是一个8KB的寄存器堆。

##### Vector Units

每个 Compute Unit 有四个Vector Units（矢量单元），Vector Unit是一个 16 通道宽的SIMD（单指令多数据流）。

**Vector Units（矢量单元）== SIMD（单指令多数据流处理器）具有：**
- 一个16-lane（16通道）IEEE-754 vector ALU（vALU）
- 一个64KB的 Vector register file（vGPR）寄存器堆
	- 该寄存器堆具有 256 个寄存器
	- 每个寄存器由 64 个 4B 宽的 entries （条目）组成
	- 如果需要8B宽的entries进行双精度计算，使用一对寄存器即可。
- 一个用于10个Wavefront的指令缓冲寄存器
	- 每个计算单元可同时启用多达40个Wavefront。

任一 Wavefront 可以在单个SIMD上连续执行四个周期，如果每个周期执行一个指令，Wavefront 将分为 4 个批次，每个批次占用 16 个通道，执行完成全部64个通道，因为每个CU具有4个Vector Units，因此，总吞吐量为每周期64（4\*16）个单精度运输。


##### Local Data Share（LDS）

每个CU都有相应的局部数据共享，也就是对workgroup中所有线程可见的 Scratchpad Memory（便签存储器）

大小为64KB

##### 总览

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240204190640.png)
<center> <font face='华文宋体' size='4'> 图 8 CU 内部结构图 </font> </center>

这SIMD中的16个数据流中每一个不会就是了wavefront吧？？？所以一个workgroup里才最多有16个wavefront。所以，每一次执行相同代码的数据流在一个workgroup中最多有16个，但是一个CU中包含4个SIMD，所以可以有四个周期？？？

#### Nvidia 与 AMD 对照表

| CUDA 术语 | AMD 术语 | 介绍 |
| :--: | :--: | :--- |
| Streaming Multiprocessor | Compute Unit(CU) | One of many parallel vector processors in a GPU that contain parallel ALUs. <br/> All waves in a workgroups are assigned to the same CU. |
| Kernel | Kernel | Functions launched to the GPU that are executed by multiple parallel workers on the GPU. <br/>Kernels can work in parallel with CPU. |
| Thread block | Workgroup | Group of wavefronts that are on the GPU at the same time.<br/>Can synchronize together and communicate through local memory. |
| Warp | Wavefront | Collection of operations that execute in lockstep, run the same instructions, and follow the same control-flow path.<br/>Individual lanes can be masked off. <br/>Think of this as a vector thread.<br/>A 64-wide wavefront is a 64-wide vector op. |
| Thread | Work item/Thread | GPU programming models can treat this as a separate thread of execution,though you do not necessarily get forward sub-wavefront progress. |
| Global Memory | Global Memory | DRAM memory accessible by the GPU that goes through some layers cache |
| Shared memory | Local memory | Scratchpad that allows communication between wavefronts in a workgroup. |
| Local memory | Private memory | Per-thread private memory, often mapped to registers. |

## GPU Programing Concepts

### 什么是HIP？

HIP（Heterogeneous-compute Interface for Portability）是AMD异构计算可移植接口，HIP是基于C++及内核语言运行时的API。

开发者可以用其创建同时兼容AMD GCM 硬件和Nvidia CUDA设备的可移植应用程序，并且完全开源，开发者可以在ROCm和Github上找到HIP。

其设计语言类似于CUDA，实际上，在使用HIP取代CUDA后，大多数API调用依然非常接近CUDA语法，目前支持CUDA运行时功能的最强子集，但某些功能（比如统一内存等）还在开发中，不过，在大多数情况下，均可以支持CUDA API所提供的功能。

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240204193533.png)
<center> <font face='华文宋体' size='4'> 图9 cuda与hip </font> </center>

###  主机与设备代码

当开始使用HIP或者CUDA代码时，首先要考虑到源代码将会呈现出两种风格，分别为主机代码和设备代码（Host code and Device code）。

主机代码：主机代码常用的C++应用程序，其入口时main函数，并且支持通过HIP API 命令在GPU上创建内存缓冲区，在GPU和CPU之间移动内存以及启动GPU内核程序。

设备代码：设备代码的语法非常类似于C语言，可以通过具有**全局属性的内核程序**进入设备代码区，设备部分的所有代码都运行在之前所提到的GCN硬件的SIMD单元上， 


### 内核、内存以及主机代码结构

首先，需要把工作从应用程序(CPU)转换到GPU内核。Kernel在启动时，通常会将工作映射到三维网格上执行。在Kernel内部，每个线程都可以访问其线程块和线程中的三维网格中的坐标。

（注意：**网格是问题映射到的东西，它不是一个物理的存在，它是一个思考问题的方式**。）

可以选择使用一维、二维或者三维坐标进行映射，AMD devices（GPUs）支持 1D、2D和3D网格，但是，大多数的问题更多的是映射到1D网格上去进行解决。

 网格可以划分为多个线程块，每个块又可以继续划分为多个线程（work item 同一个指令），每个块的大小相同。使用线程块（workgroup 同一个CU）来执行期望在GPU内核中完成的工作。

（注意：在CUDA中，执行的模型是32线程宽的warp（线程束），而HIP中，执行的模型是64线程宽的wavefront（波前））

#### 1D网格中的线程块

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240204210637.png)
<center> <font face='华文宋体' size='4'> 图 10 1D网格 </font> </center>

问题被划分到由线程块组成的网格中，每个线程块的内部都是一系列线程，如图所示，每个线程块都有自己的颜色，每个小方块代表一个线程。在设备代码中，每个线程都可以访问其各自的块ID，也就是线程块在网格中的位置。同时还可以访问其对应的线程ID。我们还可以获取这个网格中的线程块数以及块中的线程数。

每个线程都有权限去进行访问：
- 它们各自线程块（workgroup == CU）的块ID： `blockIdx.x`
- 在线程块中，它们各自的线程ID：`threadIdx.x`
- 它们线程块的维度：`blockDim.x` （线程块中的线程数？？）
- 在网格中，所有线程块的个数：`gridDim.x`

#### 2D网格中的线程块

![image.png|center|300](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240204211658.png)
<center> <font face='华文宋体' size='4'> 图 11 2维网格 </font> </center>

二维网格中的情况与一维类似， 如上图所示，其中，每个线程块都有自己的颜色，每一个小块代表一个线程。

- 每一个颜色是一个线程块、每一个小的方块是一个线程
- 在2D网格中，每个线程块和线程都有二维的坐标结构
- 在2D网格中，每个线程都有权限去进行访问：
	- 它们各自的线程块ID坐标：`blockIdx.x`、`blockIdx,y` ，也就是说，每个线程都可以访问其所属块对应的X和Y坐标。
	- 在线程块中，它们各自的线程ID：`threadIdx.x`、`threadIdx.y` ，每个线程也都可以访问块内线程各自的X和Y坐标。
	- 同一维所示。

#### Kernel

##### Kernel函数
举一个并行加速的例子：

串行循环的代码：
```cpp
for(int i=0; i<n; i++){
	h_a[i] *= 2.0;
}
```
此循环遍历数组中的条目，并将每个条目乘以2。


转换为GPU kernel 的写法为：
```cpp
__global__ void myKernel(int N, double *d_a) {
	int i = threadIdx.x + blockIdx.x*blockDim.x; 
	if (i<N) {
		d_a[i] *= 2.0;
	}
}
```
如果转换为GPU 内核，会得到上面类似的内核代码，工作在线程内部完成，不会再有循环，因为这一部分属于设备代码，将会在SIMD单元中执行它。
这是一个CUDA核函数（kernel function）的定义，用于对数组进行并行操作。
-  `__global__`: 这是一个CUDA关键字，表示接下来的函数是在GPU上执行的全局函数，即核函数。**这是因为从主机启动的内核程序必须声明为void，并且具有全局属性**。
-  `void myKernel(int N, double *d_a)`: 这是核函数的声明，接受两个参数，一个整数 `N` 和一个指向双精度浮点数数组的指针 `d_a`。
	- 注意：**传递给该内核的指针都必须指向设备内存**。
	- 理想情况在，我们认为此设备代码中的所有线程均“同时”执行。但是，这个同时执行其实是很难去实现的。
	- 但从程序概念理解来看，认为内核在设备上同时执行其所有线程将很有帮助。
-  `int i = threadIdx.x + blockIdx.x * blockDim.x;`: 这一行计算了当前线程在整个网格中的全局索引。`threadIdx.x` 表示当前线程在其块中的索引，`blockIdx.x` 表示当前块在整个网格中的索引，`blockDim.x` 表示块中的线程数。通过这些信息，**可以计算出当前线程在整个网格中的全局索引**。
	- 每个线程都可以访问其唯一的块内线程ID、线程块ID和线程块内线程的大小，从而**可以用于计算全局索引**。
-  `if (i < N) { d_a[i] *= 2.0; }`: 这是一个条件语句，确保当前线程正在处理数组中有效的元素。如果全局索引 `i` 小于数组的大小 `N`，则对数组中的元素进行乘法操作，将其值乘以2.0。
	- 由于所有的块都是等维的，并且在该数组中，启动了许多块用于执行所有的工作，可能会出现比现有线程更多的线程数，并且工作项也正在内核中执行，所以需要进行if判断，防止数组越界。
这个核函数是一个简单的并行数组乘法操作，通过将数组中的每个元素乘以2.0，实现并行加速。在实际使用中，这个核函数将被在GPU上多个线程同时执行，以提高整体计算性能。

##### Kernel启动

**dim3结构体：**

`dim3` 是一个在CUDA和HIP编程中用于表示三维网格和块大小的数据结构。它通常用于指定启动配置中的网格维度和块维度。

在CUDA和HIP中，使用 `dim3` 结构来表示三个维度的数量，其定义如下：

```cpp
struct dim3 {
    unsigned int x, y, z;
};
```

- `x`：表示网格或块的在 x 轴上的维度。
- `y`：表示网格或块的在 y 轴上的维度。
- `z`：表示网格或块的在 z 轴上的维度。

通常，`dim3` 用于指定在启动核函数时的网格和块的维度，例如：

```cpp
dim3 threads(256,1,1);                // 每个线程块中各个线程的分布情况
dim3 blocks((N+256-1)/256,1,1);       // 网格中线程块的分布情况
```

- blocks表示线程块的分布为一维分布，因为此时的yz坐标轴的数值均为1，一维分布的线程块的个数为：(N+256-1)/256个。
- threads表示每个线程块内线程的分布情况，也是一维分布，此时每个线程块内的线程的个数为256个线程。

----

**🙋：在这里解释一下，在启动GPU内核时，为什么要花费精力在线程块分布和线程块内线程的分布上？**

答：这是为了自然地映射到GCN硬件，GPU内的硬件负责将线程块动态调度到GPU的每个计算单元上，同时确保每一个线程块内的所有线程都在同一个计算单元上执行，这意味着它们可以共享 L1 Cache，并使用共享内存的高速缓存。线程块(workgroup)本身会在一系列wavefront上执行，这就是硬件上64线程宽的SIMD单元集合（一个CU）（单指令多数据流）。

有时将执行的设备代码看作矢量化代码可能会更好理解一些，其中每个块或wavefronts都是一系列线程，这些线程在64线程宽的SIMD集合（CU）上进行执行。

将这些块划分为wavefronts是因为具有非常快的上下文切换，如果一个wavefront停滞在某些依赖的数据上并且正在读取主GPU内存时，CU可以快速切换到另一个wavefront继续执行。

<font color='red' face='宋体-简'><b>由于在64线程宽的SIMD中执行代码，所以建议将块的大小设置为64的倍数。（PS：设置为256时，正好是一个CU的4个SIMD同时执行一段相同的代码）</b></font>，并且将每个块划分为多个wavefront，以便计算单元可以在它们之间进行快速的上下文切换。

---

**hipLaunchKernelGGL()函数：**

`hipLaunchKernelGGL` 是用于在 HIP 编程模型中启动核函数的函数。该函数用于在设备上执行由开发者定义的 HIP 核函数，并允许指定启动配置，包括网格和块的维度。

函数原型如下：

```cpp
hipError_t hipLaunchKernelGGL(
    const void* function_address,
    dim3 numBlocks, // 网格上线程块(workgroup)的分布
    dim3 dimBlocks, // 每个线程块中，各个线程的分布情况
    size_t sharedMemBytes,
    hipStream_t stream,
    void** kernelParams,
    void** extra
);
```

主要参数说明：

- `function_address`：指向 HIP 核函数的指针。
- `numBlocks`：`dim3` 结构，表示启动的网格的维度。
- `dimBlocks`：`dim3` 结构，表示启动的块的维度。
- `sharedMemBytes`：每个块使用的共享内存的字节数，用于动态分配共享内存。也就是前面所提到的LDS空间，Local Data Share。
- `stream`：用于执行核函数的 HIP 流。

- `kernelParams`：一个包含核函数参数的指针数组。（<font color='red'><b>内核函数本身的所有参数</b></font>）
- `extra`：保留参数，通常传入 `nullptr`。

`hipLaunchKernelGGL` 调用**核函数**在设备上启动，并指定了<font color='red'><b>网格和块的维度</b></font>。这种方式与 CUDA 中的 `<<<...>>>` 语法相似，但使用了 HIP 的函数调用风格。

```cpp
hipLaunchKernelGGL(mykernel,blocks,threads,0,0,N,a); //HIP编程
mykernel<<<blocks,threads,0,0>>>(N,a); //CUDA编程
```

#### 设备内存

#####  分配和管理设备内存

分配和管理设备内存：主机端代码有一个API，用于指示运行时分配设备内存，这个API通过hipMalloc 调用来进行实现。

💡：`size_t` 是一种数据类型，通常用于表示对象的大小或者元素的数量。<font color='red'><b>size_t 是一个无符号整数类型</b></font>，其大小足够大，足以表示系统中可能存在的最大对象的大小。

示例如下：
```cpp
#include <hip/hip_runtime.h>
#include <cstdlib>
int main() {
    int N = 1000;
    size_t Nbytes = N * sizeof(double);
    double *h_a = (double*)malloc(Nbytes);
    double *d_a = NULL;
    hipMalloc(&d_a, Nbytes); //HIP编程所需要的代码

    // 使用h_a和d_a进行其他操作

    free(h_a);
    hipFree(d_a); //HIP编程所需要的代码
    return 0;
}
```

上面的示例用于在 HIP（Heterogeneous-Compute Interface for Portability）编程模型中进行内存分配和释放。下面是代码的主要功能：
-  `int N = 1000;`: 定义整数变量 `N` 并赋值为 1000。
-  `size_t Nbytes = N * sizeof(double);`: 计算数组所需内存的总字节数。这里假设数组元素类型为 `double`。
-  `double *h_a = (double*)malloc(Nbytes);`: 使用 `malloc` 在主机上分配大小为 `Nbytes` 字节的内存，并将其赋给指针 `h_a`。这是用于存储双精度浮点数数组的主机内存。
	- 通过malloc来分配常用的主机内存
- `double *d_a = NULL;`: 定义设备上的指针 `d_a`，初始化为 `NULL`。
- `hipMalloc(&d_a, Nbytes);`: 使用 HIP 函数 `hipMalloc` 在设备上分配大小为 `Nbytes` 字节的内存，并将设备指针赋给 `d_a`。
	- 通过`hipMalloc`来分配设备内存，分配的设备内存在使用后可以通过`hipFree`进行释放。
- `free(h_a);`: 使用 `free` 函数释放之前在主机上分配的内存。
- `hipFree(d_a);`: 使用 HIP 函数 `hipFree` 释放之前在设备上分配的内存。
-  `return 0;`: 返回程序成功运行的标志。
该程序演示了<font color='red'><b>在 HIP 编程环境中进行主机和设备内存分配以及释放的基本操作</b></font>。在实际应用中，这样的内存管理通常是在进行 GPU 加速计算时所必需的。

##### 主机内存和设备内存之间的移动

**hipMemcpy函数：**

主机和设备之间的内存移动是通过hipMemcpy调用实现的。

`hipMemcpy` 是用于在 HIP 编程中进行内存拷贝的函数，其功能类似于 CUDA 中的 `cudaMemcpy`。它可以用来在主机和设备之间、设备和设备之间进行数据传输。

以下是 `hipMemcpy` 函数的基本用法：

```cpp
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyType);
```

主要参数说明：

- `dst`：目标地址，即数据将要被复制到的地方。
- `src`：源地址，即数据将要被复制的地方。
- `sizeBytes`：要复制的字节数。
- `copyType`：指定数据传输的方向，有以下几种取值：
  - `hipMemcpyHostToHost`：从主机内存复制到主机内存。
  - `hipMemcpyHostToDevice`：从主机内存复制到设备内存。
  - `hipMemcpyDeviceToHost`：从设备内存复制到主机内存。
  - `hipMemcpyDeviceToDevice`：在设备之间复制。

示例用法：

```cpp
// 从主机内存(h_a)拷贝数据到设备内存(d_a)中
hipMemcpy(d_a,h_a,Nbytes,hipMemcpyHostToDevice); 

// 从设备内存(d_a)拷贝数据到主机内存(h_a)中
hipMemcpy(h_a,d_a,Nbytes,hipMemcpyDeviceToHost);

// 从a设备内存(d_a)拷贝数据到b设备内存(d_b)中 
hipMemcpy(d_b,d_a,Nbytes,hipMemcpyDeviceToDevice);
```

**hipMemcpy2D函数：**

`hipMemcpy2D` 函数用于在 HIP 编程中执行二维内存拷贝，类似于 CUDA 中的 `cudaMemcpy2D`。它允许在主机和设备之间以及设备之间进行二维数据的传输。

以下是 `hipMemcpy2D` 函数的基本用法：

```cpp
hipError_t hipMemcpy2D(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    hipMemcpyKind kind
);
```

主要参数说明：
- `dst`：目标地址，即数据将要被复制到的地方。
- `dpitch`：目标内存的跨度（每行元素的字节数）。
- `src`：源地址，即数据将要被复制的地方。
- `spitch`：源内存的跨度（每行元素的字节数）。
- `width`：每行中要传输的元素数。
- `height`：要传输的行数。
- `kind`：指定数据传输的方向，同 `hipMemcpy` 中的 `hipMemcpyKind`。
 
#### 代码示例

##### API错误检查

大多数HIP API 函数会返回一个类型为hipError_t的错误代码，如果API函数返回无错误，会获得一个已定义的**hipSuccess值**（也就是0值），否则将返回一个非零错误代码。

我们可以通过`hipGetLastError`和`hipPeekLastError`获取或者查看上一次错误。

同时，还可以通过简单的#Define 检查获取每个错误代码对应的错误字符串。

```cpp
#define HIP_CHECK(command) { 
	hipError_t status = command;
	if (status != hipSuccess) { 
		std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl;
		std::abort(); 
		}
	}
```

##### 整合上面所有代码，展示示例

```cpp
#include "hip/hip_runtime.h"

int main() {
    int N = 1000;
    size_t Nbytes = N * sizeof(double);

    double *h_a = (double*)malloc(Nbytes); // 主机内存
    double *d_a = NULL;

    HIP_CHECK(hipMalloc((void**)&d_a, Nbytes));

    HIP_CHECK(hipMemcpy(d_a, h_a, Nbytes, hipMemcpyHostToDevice)); // 将数据复制到设备

    hipLaunchKernelGGL(myKernel, dim3((N + 256 - 1) / 256, 1, 1), dim3(256, 1, 1), 0, 0, N, d_a); // 启动核函数
    
    HIP_CHECK(hipGetLastError());// 就是捕捉上面启动时所犯的错误

    HIP_CHECK(hipMemcpy(h_a, d_a, Nbytes, hipMemcpyDeviceToHost));

    free(h_a); // 释放主机内存

    HIP_CHECK(hipFree(d_a)); // 释放设备内存

    return 0;
}

```

### 设备管理同步与MPI编程

本节将了解管理设备执行的内容，并理解主机设备异步运行的方式。

#### MPI

如果一个系统中有多个GPU，并且多个主机线程想要调用HIP API或者多个MPI等级，会发生什么？

- 1️⃣首先，主机通过调用`hipGetDeviceCount` 查询其系统中可见的设备数。
	- `hipGetDeviceCount` 的作用是<font color='red'><b>获取当前系统中可用的 GPU 设备数量</b></font>。这个函数通常会返回一个整数值，表示系统中可用的 GPU 设备数量。在使用 HIP 进行 GPU 加速计算时，通常需要先调用这个函数获取设备数量，然后根据需求选择特定的 GPU 设备进行计算任务的分配。
	- 也就是主机可以询问当前系统中可用的设备数量
```cpp 
		int numDevices = 0;
		hipGetDeviceCount(&numDevices); //需要传入引用，从而获得相应的数据
```
- 2️⃣然后，通过运行时机制向指定设备发出指令，从而调用`hipSetDevice`，所有HIP API调用都位于当前所选设备的上下文中，
	- `hipSetDevice` 是 HIP API 中的一个函数，用于<font color='red'><b>将当前线程的 GPU 上下文切换到指定的 GPU 设备上</b></font>。在进行 GPU 加速计算时，通常需要在多个 GPU 设备之间进行切换或选择特定的 GPU 设备进行计算任务。
	- 也就是主机可以指示特定的设备去执行相应的指令
```cpp
		int deviceID = 0;
		hipSetDevice(deviceID);
```
- 3️⃣最后，查询当前运行所选择的设备
	- `hipGetDevice` 是 HIP API 中的一个函数，<font color='red'><b>用于获取当前线程所使用的 GPU 设备的索引号。</b></font>在进行 GPU 加速计算时，**有时候需要知道当前线程正在使用的是哪个 GPU 设备，以便进行相应的操作或查询设备的相关信息**。`hipGetDevice` 函数就是用来获取当前线程所使用的 GPU 设备的索引号的。
	- 也就是说主机可以获得当前所选线程所在设备的设备号
```cpp
		int deviceID;
		hipGetDevice(&deviceID);
```

因此，通过在应用程序的不同部分执行调用，如果我们能够主动交换所选的活动设备，主机代码就可以管理多个设备。即：主机可以通过在运行时交换当前选定的设备来管理多个设备。

MPI等级可以用于多个不同设备的设置，这样每个等级具有相对应的设备，或者可以设置多个等级，通过超额订阅向单个设备发出命令。

主机也可以获得设备的属性，使用下面的命令：

```cpp
		hipDeviceProp_t props;
		hipGetDeviceProperties(&props, DeviceID);
```

`hipDeviceProp_t` 是 HIP API 中定义的结构体类型，用于表示 GPU 设备的属性信息。该结构体包含了各种关于 GPU 设备的详细信息，如设备名称、计算能力、内存信息等。开发者可以使用 `hipGetDeviceProperties` 函数来获取指定 GPU 设备的属性信息，并将这些信息填充到 `hipDeviceProp_t` 类型的变量中。

如果期望通过运行时可以自动选择系统中的指定设备，应该使用该结构体。

#### 阻塞和非阻塞API函数

该阻塞以及非阻塞是相对于主机系统的。

比如，前面所提到的内核启动函数`hipLaunchKernelGGL`是对主机的非阻塞调用。
- 非阻塞调用，它表示命令包发送到命令处理器之后，主机将立即继续执行，而不再等待设备完成内核执行。因此在内核占用设备运行的这段时间内，可以同时运行其他工作，例如：在主机上进行MPI通信等。（如果知道内核将花费一些时间，那么这是在主机上进行一些工作(即 MPI comms)的好时机）

然而，`hipMemcpy`是阻塞调用，阻塞调用意味着：
- 在函数返回后，可以对指针中指向的数据进行安全的访问，并且，自由在复制操作结束后，主机才会继续执行

`HipMemcpyAsync`是`hipMemcpy`的非阻塞版本，可以在GPU的命令处理器中指示或者发送命令来执行内存移动，并且，主机会在指示GPU执行Memcpy后立即继续执行，这意味着，在完成某种同步之前，访问这两个指针所指向的内存均不够安全，

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240206151451.png)
<center> <font face='华文宋体' size='4'> 图12 阻塞与非阻塞代码 </font> </center>

 在主机实际等待GPU完成内核代码时，内存将再转回到主机，这个就是在上面代码中的隐式同步点。

#### 任务流

##### 使用流的一种方式：分配小内核到不同的流中进行加速

 可以把流想象成一个任务队列，在这个队列中可以是内核函数、memcpys或者事件。HIP流队列中的任务会依次按顺序执行，但是，不同流队列的任务可以叠加使用和划分设备资源。

我们可以通过`hipStreamCreate`创建一个流对象，然后通过`hipStream Destroy`来进行销毁。

```cpp
hipStream_t stream;
hipStreamCreate(&stream);
hipStreamCteate(stream);
```

`hipStream_t` 是 HIP API 中的一个数据类型，用于表示 GPU 上下文中的一个流（stream）。在并行计算中，流是一种将任务异步执行的机制，可以让多个任务在 GPU 上并行执行，提高计算效率。开发者可以创建多个流，并将不同的计算任务分配给这些流，以便并行执行这些任务。

```cpp
#include <iostream>
#include <hip/hip_runtime.h>

int main() {
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1); // 创建流1
    hipStreamCreate(&stream2); // 创建流2

    // 在流1上执行计算任务
    hipLaunchKernelGGL(SomeKernel, dim3(blocks), dim3(threads), 0, stream1, args);

    // 在流2上执行另一个计算任务
    hipLaunchKernelGGL(AnotherKernel, dim3(blocks), dim3(threads), 0, stream2, args);

    // 等待流1中的计算任务完成
    hipStreamSynchronize(stream1);

    // 等待流2中的计算任务完成
    hipStreamSynchronize(stream2);

    // 销毁流
    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);

    return 0;
}
```

需要注意的是，当传入的参数为0时，这会指示API调用进入一种特殊类型的NULL流队列，在NULL流中，原则上不允许任何任务相重合，**在NULL流中排队的任务只有等待所有先前请求的任务以及其他流完成之后才开始执行**，像hipMemcpy这样的**隐式阻塞调用**始终运行在NULL流上。

使用流，可以提高程序的并行性：

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240206154518.png)
<center> <font face='华文宋体' size='4'> 图 13 使用一个流和使用多个流执行多个任务的对比 </font> </center>

注意，在这样做时，需要保证各个内核不会修改内存中的共享部分，以免产生数据冲突。

<font color='red'><b>如果一个内核实际上占用了整个GPU资源，这时GPU已经没有多余的流分配给其他流去运行了，因此此时则不应该再使用此方法进行性能的优化。</b></font>但是，对于多个小内核来说，这将会有帮助。

##### 使用流的另一种方式：用计算内核时间覆盖数据移动的时间

**Overlapping kernels with data movement**

GPU具有独立的引擎，用于执行主机和设备之间的**内存移动**、**排队**和**计算内核**。在AMD GPU上，<font color='red'><b>允许这三种操作在不分割GPU资源的情况下重复使用</b></font>（前提是：这些操作都应在各自的非NULL流队列中，并且在主机和设备之间的复制操作所涉及的**主机内存都应被固定**）。

Malloc分配的主机数据在默认情况下是可以分页的。除了使用malloc方法分配主机内存以外，也可以通过HIP api 调用在固定区域中分配主机内存。这就被称为**固定主机内存**。

固定主机内存分配和释放的例子：
```cpp
	double *h_a = NULL;
	hipHostMalloc(&h_a, Nbytes); // h_a是内存分配好之后指向那块区域的指针，Nbytes是字节数。

	hipHostFree(h_a);
```
(PS:在分配设备内存的代码为`hipMalloc(&d_a,Nbytes)`)

注意：<font color='red'><b>在使用固定内存后，主机和设备之间memcpys操作的带宽将显著增加</b></font>，因此，**如果应用程序频繁地在主机和设备之间传输数据，推荐在固定区域中分配主机内存**。

 使用流的示例：一定要记住：<font color='green'><b>允许内存移动、排队和计算内核这三种操作在不分割GPU资源的情况下重复使用</b></font>。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240206162106.png)
<center> <font face='华文宋体' size='4'> 图 14 使用Overlapping 提高并行度，进行加速操作 </font> </center>

在某些区域中，计算内核与 主机和设备 之间的数据移动在时间上相重合。

##### 通过流来协调主机和设备之间的执行

如果要协调主机和设备之间的执行，可以通过同步和阻塞机制来进行两者的协调操作（也就是常常提到的<font color='red'><b>同步点</b></font>）。

<font color='red'><b>主机与流</b></font>之间的同步有两种形式：
- `hipDeviceSynchronize()`：`hipDeviceSynchronize` 用于<font color='red'><b>在当前线程中同步设备上的所有任务</b></font>。在并行计算中，通常会在某些关键点上进行同步操作，以确保之前的任务已经完成，然后再继续执行后续的操作。`hipDeviceSynchronize` 函数就是用来实现这种同步操作的。调用 `hipDeviceSynchronize` 函数会阻塞当前线程，直到设备上的所有任务都执行完成。这样可以确保之前提交给设备的所有操作都已经完成，然后才能继续执行后续的操作。
	- Heavy-duty 同步点
	- 阻塞主机，直到**设备中所有流**中的所有工作都报告完成
	- 当主机返回到控制流时，可以保证GPU已经完成所有安排的任务
- `hipStreamSynchronize(stream)`：`hipStreamSynchronize(stream)` 用于在指定的流上同步设备上的所有任务。与 `hipDeviceSynchronize()` 同步所有流不同，`hipStreamSynchronize(stream)` 只会同步指定的流上的任务。调用 `hipStreamSynchronize(stream)` 函数会阻塞当前线程，直到流 `stream` 上的所有任务都执行完成。这样可以确保在流 `stream` 上提交的所有操作都已经完成，然后才能继续执行后续的操作。
	- 阻塞主机，直到**stream流**中所有工作都报告完成。

<font color='red'><b>流与流</b></font>之间的同步：需要引入叫做**Event**(事件)的新结构体。

与Event相关的代码：
```cpp
	hipEvent_t event;        //主机创建event结构体
	hipEventCreate(&event);  //给event结构体注入生命

	hipEventRecord(event, stream); // 将一个event放入steram流中

	hipEventDestroy(event);  //销毁event
```

 `hipEvent_t` 是 HIP API 中用于表示事件的数据类型。**事件是一种同步机制**，可以用来在 GPU 上进行任务之间的同步，或者测量计算时间等。`hipEvent_t` 类型的变量用于表示一个事件对象。通过创建、记录、等待和销毁事件对象，开发者可以实现对 GPU 上任务执行顺序和时间的精确控制。

流之间的同步----Event的同步：
- `hipEventSynchronize(event)`：`hipEventSynchronize(event)` 用于等待指定事件对象的完成。这个函数会阻塞当前线程，直到事件对象 `event` 完成。调用 `hipEventSynchronize(event)` 函数可以确保在继续执行后续操作之前，等待事件对象 `event` 完成，即直到事件对象 `event` 被记录后才会返回。
	- 设备在事件报告完成之间将始终处于阻塞状态
	- Event只是流队列中的同步点，并不会指示其他流上执行的位置。
- `hipEventElapsedTime(&time, startEvent, endEvent)`：`hipEventElapsedTime(&time, startEvent, endEvent)` 用于计算两个事件之间的时间间隔。这个函数会将从 `startEvent` 到 `endEvent` 之间的时间间隔（以毫秒为单位）存储在 `time` 变量中。
	- 获得某个操作所需要的时间
- `hipStreamWaitEvent(stream, event)`：`hipStreamWaitEvent(stream, event)` 用于在指定的流上等待指定的事件对象完成。这个函数会使得流 `stream` 中的任务等待事件对象 `event` 完成后再继续执行。即等待某个事件的完成才能执行后续的操作。
	- 主要用于在不同流之间进行强制排序
	- 注意：这个是<font color='red'><b>对主机的非阻塞调用，不是同步点</b></font>，纯粹是对GPU的指令。
	- 这会指示在特定的事件完成之前，所有提交到该流的工作都处于等待状态，并且该事件本身可以在不同的流上。

##### 流的常见用法：隐藏MPI延迟

假设要在内核中进行**局部计算**，这时将内核放到一个计算流队列（不是NULL流）上，因为此时属于非阻塞调用，因此主机可以继续执行，当内核运行时，我们期望将数据从GPU取出，并发送到MPI缓冲区，然后，返回时再将数据从MPI发送回GPU。
 
```cpp
// Queue local compute kernel
hipLaunchKernelGGL(myKernel, blocks, threads, 0, computestream, N, d_a);

// Copy halo data to host
hipMemcpyAsync(h_commBuffer, d_commBuffer, Nbytes, hipMemcpyDeviceToHost, datastream);
hipStreamSynchronize(dataStream); // Wait for data to arrive

// Exchange data with MPI
MPIDataExchange(h_commBuffer);

// Send new data back to device
hipMemcpyAsync(d_commBuffer, h_commBuffer, Nbytes, hipMemcpyHostToDevice, datastream);
 
```

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240206170250.png)
<center> <font face='华文宋体' size='4'> 图 15 隐藏MPI延迟 </font> </center>


### 设备代码区域以及Kernel 代码

这一节主要介绍设备代码区域以及内核代码的用途。

#### 三种关键字修饰的函数

在使用HIP编译器进行hipcc编译时，将会对源代码进行几次遍历，其中一次遍历用于编译主机代码，另一次遍历则用于编译设备代码。

- `_global_`functions：
	- 通过全局属性修饰的函数通常是GPU内核的入口处，该**函数由主机去进行调用**
	- 在这个区域下的函数，将会在GCN硬件的SIMD单元上进行执行
	- 在这个区域中，想要调用其他的函数，则被调用的函数需要用`_device_`进行修饰
- `_device_`functions：
	- 通过该关键字进行修饰的函数，可以被`_global_`函数或者`_device_`函数进行调用，但是无法被主机代码调用。
	- 在编译主机代码时，编译器会忽略这些函数
- `_host__device_`functions：
	- 可以同时被`_global_`函数、`_device_`函数和主机进行调用
	- 注意：当用该关键字修饰的函数被设备调用时，此函数将会在SIMD单元上进行执行

#### 线程发散

---

相关知识回顾：锁步方式

锁步方式（Lockstep）是一种并行计算模式，其中**所有线程或处理单元必须在执行下一个步骤之前同步执行当前步骤**。在锁步方式中，每个处理单元在执行完当前步骤后都必须等待所有其他处理单元完成当前步骤，然后才能进入下一个步骤。

锁步方式通常用于需要严格的同步和协调的计算任务，例如在图形处理器 (GPU) 中的某些并行计算场景。在这种情况下，锁步方式可以确保所有处理单元在每个计算步骤中执行相同的操作，并且保持数据的一致性。

锁步方式的一个缺点是可能会导致性能瓶颈，特别是在处理单元之间的通信和同步开销很高时。<font color='red'><b>由于所有处理单元都必须等待其他处理单元完成当前步骤，因此可能会导致一些处理单元处于空闲状态，从而降低整体计算效率。</b></font>

---

因为64线程宽的SIMD单元通常是以锁步方式执行的，因此分支逻辑的代价会很高。

在SIMD单元上，注意线程发散（divergence）：
- 分支逻辑（if-else）代价会很高：
	- wavefront的矢量单元遇到if语句
	- 指令会对条件进行计算求值
		- 如果条件为真，将继续执行相应的语句体
		- 如果条件为假，也会继续执行相应的语句体，但相反，某些线程可能会被搁置，仅仅显示为NoOps，
	- 这就是线程发散

目前，在硬件中的调度器有足够的能力判断wavefront的所有线程是否在计算相同的条件，并可以跳过代码块以及真或假的代码块。但是，如果一个wavefront中的一些线程判断为真，一些判断为假，这通常会导致wavefront中的线程发散，从而影响性能。

举个例子来说：
![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240208051125.png)
<center> <font face='华文宋体' size='4'> 图16 举例说明线程发散对效率的影响 </font> </center>

#### 设备代码中的内存声明

可以声明内存并确定设备区域，但是设备代码不支持Malloc和Free，因此无法动态分配设备内存。

可以在堆栈上声明变量，在设备代码中声明的堆栈变量将在设备寄存器中进行分配，并且对每个线程都是私有的。 

通常，线程可以通过设备指针访问设备RAM中的公共内存，但不能共享内存。如果想要共享内存的话，需要将内存声明为共享，即使用`_share_`关键字。
- 在堆栈上**声明为共享的任何变量只在LDS空间和共享内存的每个块上分配一次**，它们可被同一块中的所有线程共享访问。
- 注意：访问共享内存通常比访问设备内存和全局内存要快的多，但慢于寄存器！
- 通常，这些共享内存数组在编译时也会占用空间（例外：动态分配的LDS空间）

在内核中使用共享内存的例子：
```cpp
__global__ void reverse(double *d_a) {
    __shared__ double s_a[256]; // array of doubles, shared in this block
    int tid = threadIdx.x;
    s_a[tid] = d_a[tid];        // each thread fills one entry
    __syncthreads();            // synchronize threads before accessing shared memory
    d_a[tid] = s_a[255 - tid];  // write out array in reverse order
}

int main() {
    double *d_a;                // assume d_a is allocated and initialized
    hipLaunchKernelGGL(reverse, dim3(1), dim3(256), 0, 0, d_a); // Launch kernel
    // Add error checking for memory allocation and kernel launch if necessary
    return 0;
}
```
上面内核代码的作用是：获取含有256个元素的数组并进行反转。因此需要将该数组顺序读入（每个线程负责读入一个）到共享内存的数组（此时256个线程均可以进行访问）中，然后再倒序写出来（再利用256个线程对共享内存中的数组进行倒序输出到原来的数组中，从而实现数组反转的操作）。
 
### HIP API

- Device Management:
	- `hipSetDevice()`,`hipGetDevice()`,`hipGetDeviceProperties()`
- Memory Management:
	- `hipMalloc()`,`hipMemcpy()`,`hipMemcpyAsync()`,`hipFree()`
- Streams:
	- `hipstreamCreate()`,`hipSynchronize()`,`hipstreamSynchronize()`,`hipstreamFree()` 
- Events:
	- `hipEventCreate()`,`hipEventRecord()`,`hipstreamWaitEvent()`,`hipEventElapsedTime()` 
- Device Kernels:
	- `__global__`,`__device__`,`hipLaunchKernelGGL()`
- Device code:
	- `threadIdx`,`blockIdx`,`blockDim`,`__shared__`
	- 200+math functions covering entire CUDA math library. 
- Error handling:
	- `hipGetLastError()`,`hipGetErrorString()`

HIP编程文档：[https://rocm.docs.amd.com/projects/HIP/en/latest/](https://rocm.docs.amd.com/projects/HIP/en/latest/)

## GPU Programing Software

本章介绍编译器、库文件以及在编写HIP代码时应该注意的事项。

### AMD GPU 编译器

AMD 支持几个编译器来生成AMD GCN 的汇编指令
- hcc：基本上用于编译HIP代码
	- 通常不需要直接调用hcc，而是通过基于Perl语言的hipcc封装脚本调用hcc，并根据安装的ROCm版本来管理相关事项。
	- hipcc调用hcc，然后对hip代码进行编译，而hcc其实是基于clang的派生，因此，可以在hipcc中使用熟悉的clang选项和标志，然后hipcc会将它们传递给hcc，编译器将生成在设备上运行的AMD GCN指令集，然后执行多次遍历，为主机编译所有的x86指令。
	- ![image.png|left|100](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240208063449.png)
	- 注意：一旦编译完HIP代码，就得到了一个可以在设备上运行的可执行文件。
	- 如果想要检查设备上运行的指令集，可以使用ROCm中提供的一个叫做extract kernel 的软件来实现这一点，该软件将输出一个文本文件，其中包含了设备上运行的所有AMD GCN指令。
	- 指令集架构完全开源并免费：https://developer.amd.com/resources/developer-guides-manuals/
- AOMP（AMD OpenMP Compiler）
	- AOMP是一个开放的MP编译器，
- 将会支持fortran编译器

### HIPCC的使用

hipcc是clang的派生，可以接受像clang的编译命令。

`HIPCC_VERBOSE=7 hipcc dotprod.cpp -o dotprod`

前面的命令，是为了可以输出一些有用的信息。

#### 安装ROCm

为了能够运行hipcc以及HIP等，需要安装ROCm软件，在Ubuntu或Centos系统上，可以通过包管理工具进行安装。

安装说明的链接：https://github.com/RadeonOpenCompute/ROCm

---

**ROCm的介绍**

ROCm（Radeon Open Compute Platform）是由 AMD 开发的开源软件平台，旨在为 GPU 计算提供开放、可移植和高性能的解决方案。ROCm 提供了一套工具和库，使开发者能够在 AMD GPU 上进行通用目的的并行计算，包括深度学习、科学计算、机器学习、数据分析等领域。

ROCm 的主要组成部分包括：
1. **ROCm 驱动程序**：ROCm 提供了专用的 GPU 驱动程序，以便在 AMD GPU 上运行计算任务。
2. **HIP（Heterogeneous-Compute Interface for Portability）**：HIP 是 ROCm 的一个关键组件，它提供了一个类似于 CUDA 的编程接口，**使开发者能够在 AMD 和 NVIDIA GPU 上编写通用的并行计算代码**。HIP 使**开发者能够更容易地将已有的 CUDA 代码移植到 ROCm 平台上**。
3. **ROCm 数学库**：ROCm 提供了一系列针对 GPU 加速计算优化的数学库，如 rocBLAS、rocRAND、rocFFT 等，用于在 AMD GPU 上进行线性代数运算、随机数生成、傅里叶变换等操作。
4. **ROCm 编译器**：ROCm 提供了支持 GPU 加速计算的编译器，能够将 HIP、CUDA 或其他语言编写的代码转换为可在 AMD GPU 上执行的汇编代码。
5. **ROCm 开发工具**：ROCm 提供了一系列开发工具，如 ROCm Profiler 和 ROCm Debugger，用于分析和调试 GPU 加速应用程序。
6. **ROCm 容器支持**：ROCm 支持容器化技术，允许开发者在容器中部署和运行 ROCm 加速应用程序，从而更轻松地管理和部署 GPU 加速计算环境。

---

如何判断ROCm已经安装成功？

使用rocminfo软件，这个可以显示系统上所有符合HSA设备（如AMD CPU 和 AMD GPU）的设备信息。

如果安装位置标准，可以在`/opt/rocm/.info/version-dev`中查看安装的版本信息或者也可以在命令行中进行查看。


### AMD GPU 库文件

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240208065149.png)
<center> <font face='华文宋体' size='4'> 图 17 hip与roc和cu的关系 </font> </center>

- roc*: 以roc开头的库都是AMD GCN库，通常是用HIP编写的。
- cu*:以cu开头的库都是Nvidia库，
- hip*:以hip开头的库通常是位于roc或cu后段之上的接口层。
	- 例如，hipBLAS是rocBLAS和cuBLAS之间的接口层，如果使用的是Nvidia设备，则会调用cuBLAS，如果使用的是AMD设备，则会调用rocBLAS。
	- hip库可由hipcc编译并生成相应设备的调用，hip层只是一个简单的封装，用于调用正确的设备。
	- 会替换HIP调用并内联cuBLAS调用，不会在Nvidia设备上产生任何运行开销

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240208065932.png)
<center> <font face='华文宋体' size='4'> 图 18 roc 与 cu 各个库之间的对应关系 </font> </center>

注意：如果在乎代码的可移植性（既运行在AMD设备上又运行在Nvidia设备上），需要使用hipBLAS，让编译器负责管理运行时间的调用。如果不在乎移植性，只想在AMD的设备上运行，直接使用rocBLAS即可。

如果要使用对应的库，直接在编译时，链接上对应的库即可，例如：`-lrocblas`。

## Porting CUDA Applications to HIP

本章介绍如何将CUDA代码转换为HIP代码。

通常使用<font color='red'><b>hipify</b></font>来形容从CUDA到HIP的转换。

HIP的文档地址：https://github.com/ROCm-Developer-Tools/HIP，可以得到支持的CUDA版本。

### 对比
#### kernel差别

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240208071208.png)
<center> <font face='华文宋体' size='4'> 图 19 CUDA kernel 与 HIP kernel 的对比 </font> </center>
没错，确实是一毛一样的！！！

#### 调用API差别

但是API的调用存在一些差别，需要换一下名称：

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240208071347.png)
<center> <font face='华文宋体' size='4'> 图 20 API的对比 </font> </center>

变更的原因是因为：需要使用不同的前缀来区分这些等同的API。

#### kernel 启动差别

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240208071614.png)
<center> <font face='华文宋体' size='4'> 图 21 内核启动的对比 </font> </center>

通过上述的对比，可以发现一些规律，那么有规律，就可以通过自动化来进行实现。

### 实现cuda转hip

通过AMD一些公开的常用工具对代码进行移植或自动转换。**常常用到的两个工具是：Hipify-Perl 和 Hipify-clang** 。它们被设计用于自动转换现有的CUDA代码，但它们不是万能的，不能自动转换所有的代码，不能自动转换部分的CUDA代码需要程序员手动去进行处理。

#### 对 hipify-perl 的分析：

1. **实现语言**：
   - `hipify-perl` 使用 Perl 编程语言实现。
2. **转换方法**：
   - 基于规则的简单文本替换。
   - 使用预定义的规则集来识别 CUDA 代码中的函数调用、宏和特定语法结构，并将其转换为对应的 HIP 代码。
3. **准确性和可靠性**：
   - 由于是基于简单的文本替换，可能会出现一些误差和不准确的情况。
   - 对于简单的代码转换而言效果还不错，但对于复杂的代码可能不够准确。
4. **性能和效率**：
   - 处理速度可能比较快，因为是基于简单的文本替换实现的。
   
#### 对hipify-clang的分析：

1. **实现语言**：
   - `hipify-clang` 使用 C++ 编程语言实现，并依赖于 Clang 工具链中的 `clang` 和 `llvm` 库。
2. **转换方法**：
   - 通过构建语法树（AST）来分析 CUDA 代码。
   - 使用 LLVM 库中的功能来分析和修改代码。
   - **可以处理更复杂的代码结构，并生成更优化的 HIP 代码。**
3. **准确性和可靠性**：
   - 通过构建语法树进行代码分析和转换，可以更准确地识别和处理代码结构。
   - 可以生成更可靠和准确的 HIP 代码。
4. **性能和效率**：
   - 处理速度可能相对较慢，因为需要构建语法树进行代码分析和转换。
   - 但可以生成更准确和优化的 HIP 代码。

`hipify-perl` 和 `hipify-clang` 在实现方式、准确性和性能等方面存在一些差异。一般而言，`hipify-perl` 可能会更方便和快速，而 `hipify-clang` 更准确和可靠。


具体示例以后再进行补充......