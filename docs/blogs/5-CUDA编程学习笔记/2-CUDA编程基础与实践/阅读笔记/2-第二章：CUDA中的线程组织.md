## 2.1 C++语言中的Hello World 程序

```cpp
#include <stdio.h>

int main(void){
	printf("Hello world! \n");
	return 0;
}
```

经过编译、链接，即可获得可执行文件。

```shell
g++ hello.cpp -o hello
```

## 2.2 CUDA中的Hello World 程序

### 2.2.1 只有主机函数的CUDA程序

CUDA 程序的编译器驱动(compiler driver)nvcc支持编译纯粹的C++代码。

一般来说， 一个标准的CUDA程序中既有纯粹的C++代码，又有不属于C++的真正的 CUDA代码。

CUDA程序的编译器驱动nvcc在编译一个CUDA程序时，会<font color='red'><b>将纯粹的C++代码交给C++的编译器（如前面提到的g++)去处理，它自己则负责编译剩下的部分</b></font>。

CUDA程序源文件的扩展名是.cu，所以可以先将上面写好的源文件更名为hello.cu，然后用nvcc编译：

```shell
nvcc hello.cu -o hello
```

### 2.2.2 使用核函数的CUDA程序

```cpp
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

- 调用核函数的格式：`hello_from_gpu<<<1,1>>>();`
	- 核函数中的线程常组织为若干线程块
		- 三括号中的第一个数字可以看作线程块的个数。可以记作：网格大小（grid size）
		- 第二个数字可以看作每个线程块中的线程数。可以记作：线程块大小（block size）
	- 因此，`<<<网格大小，线程块大小>>>`
	- 核函数中总的线程数就等于网格大小乘以线程块大小。
- 在核函数中使用`printf()`函数时，也需要包含头文件`<stdio.h>`。需要注意的是，<font color='red'><b>核函数中不支持C++的iostream</b></font>。
- 在调用核函数之后，有这句代码：`cudaDeviceSynchronize();`
	- 该代码调用了一个CUDA的运行时API函数，<font color='red'><b>去掉这个函数将不能输出字符串</b></font>。
	- 这是因为调用输出函数时，**输出流是先存放在缓冲区的，而缓冲区不会自动刷新。只有程序遇到某种同步操作时，缓冲区才会刷新**。
	- 上面函数的作用是，同步主机与设备，所以能够促使缓冲区刷新。


## 2.3 CUDA中的线程组织

### 2.3.1 使用多个线程的核函数

每个线程块的计算是相互独立的，无论完成计算的次序如何，每个线程块中的每个线程都进行一次计算。

### 2.3.2 使用线程索引

一个核函数允许指派的线程数目是巨大的，能够满足绝大多数应用程序的要求。需要指出的是，一个核函数中虽然可以指派如此巨大数目的线程数，但在执行时能够同时活跃（不活跃的线程处于等待状态）的线程数是由硬件（主要是CUDA核心数）和软件（核函数中的代码）决定的。

每个线程在核函数中都有一个唯一的身份标识。每个线程的身份标识可由两个参数确定。


### 2.3.3 推广至多维网格

`blockIdx`和`threadIdx`是类型为`uint3`的变量。这个类型是一个结构体，具体有x、y、z这3个成员。

该结构体的定义如下：
```cpp
struct __device_builtin__ uint3
{
    unsigned int x, y, z;
};
typedef __device_builtin__ struct uint3 uint3;
```

`gridDim` 和 `blockDim` 是类型为dim3的变量。该类型是一个结构体，具体有x、y、z这3个成员。

因此，可以用结构体dim3来定义多维的网格和线程块：
```cpp
dim3 grid_size(Gx, Gy, Gz);
dim3 block_size(Bx, By, Bz);
```

![image.png|center|300](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240922223142.png)

一个线程块中的线程还可以细分为不同的线程束，一个线程束是同一个线程块中相邻的warpSize个线程。warpSize是一个内建变量，表示线程束大小，在Nvidia GPU中，每个架构的线程束大小都为32。

### 2.3.4 网格与线程块大小的限制

在 NVIDIA GPU 的 CUDA 编程模型中，网格大小和线程块大小的限制如下：

- **线程块大小限制（Thread Block Size Limits）**
   每个线程块中的线程数量限制为：
   - **最大线程数**: 每个线程块最多可以有 **1024 个线程**。
   - **线程块的维度限制**:
     - 每个线程块可以是 1D、2D 或 3D 结构。
     - 在 3D 线程块中，每个维度的最大尺寸为：
       - `x` 维度：**1024**
       - `y` 维度：**1024**
       - `z` 维度：**64**
     - 但是，线程块内的线程总数（`x * y * z`）不能超过 **1024**。
- **网格大小限制（Grid Size Limits）**
   网格大小表示线程块在 1D、2D 或 3D 网格中的数量。每个维度的网格大小限制如下：
   - **x 维度**：`2^31 - 1`（即 2147483647）
   - **y 维度**：`65535`
   - **z 维度**：`65535`

在实际编程时最好查看特定 GPU 架构的文档（例如，使用 `cudaDeviceProp` 查询设备属性）。

## 2.4 CUDA中的头文件

CUDA程序也需要头文件，但是在使用nvcc编译器驱动编译.cu文件时，将自动包含必要的CUDA头文件，比如`<cuda.h>`和`<cuda_runtime.h>`。但为了程序的更加直观，还是要直接写明包含相应的头文件才好。

## 2.5 用nvcc编译CUDA程序

CUDA的编译器驱动(compiler driver)nvcc先**将全部源代码分离为主机代码和设备代码**。

主机代码完整地支持C++语法，但设备代码只部分地支持C++。

nvcc 先将设备代码编译为PTX(parallel thread execution)伪汇编代码，再将PTX代码编译为二进制的cubin目标代码。
- 在**将源代码编译为PTX代码**时，需要用选项`-arch=compute_XY`指定一个<font color='red'><b>虚拟架构</b></font>的计算能力，用以确定代码中能够使用的 CUDA功能。
- 在**将PTX代码编译为cubin代码**时，需要用选项`-code=sm_ZW`指定一个<font color='red'><b>真实架构</b></font>的计算能力，用以确定可执行文件能够使用的GPU。
- 才用根据设备的型号，进行动态编译的方式，需要用到选项`-code=compute_XY`，不会为特定的架构生成机器代码，而是在运行时，由GPU驱动程序动态编译成目标机器代码。

为了程序的正确执行，真实架构的计算能力必须等于或者大于虚拟架构的计算能力。

如果仅仅针对一个GPU编译程序，一般情况下，建议将以上两个计算能力都选为所用GPU的计算能力。

1. `-gencode arch=compute_XY,code=sm_XY`:
	- 这会为指定的 **XY** 架构生成 PTX 代码，并<font color='red'><b>直接编译成适用于 XY 架构的目标代码（SASS）</b></font>
	- 供在具有 **XY** 计算能力的 GPU 上直接运行。
	- 比如：`-gencode arch=compute_75,code=sm_75` 会为 **Compute Capability 7.5** 的 GPU 生成机器代码。
2. `-gencode arch=compute_XY,code=compute_XY`:
	- 这会生成 **XY** 架构的 PTX 代码，但<font color='red'><b>不会为特定的架构生成机器代码</b></font>。**生成的 PTX 代码会在运行时由 GPU 驱动程序动态编译成目标机器代码**。
	- 这种方式更加灵活，适用于多个 GPU 计算能力版本，具体生成的机器代码取决于运行时的 GPU。
	- 例如：`-gencode arch=compute_75,code=compute_75` 会生成 **Compute Capability 7.5** 的 PTX 代码，并在运行时进行动态编译。

如果两个命令一起使用：
```bash
-gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75
```
这样，编译器会生成针对 **sm_75** 架构的机器代码，同时保留 **compute_75** 的 PTX 代码，确保它可以在运行时动态编译以适配其它兼容的架构。

上面的两个命令，也可以缩减为一条指令：
```bash
-arch=sm_XY
```

❗️❗️如果在编译时，不指定计算能力，则编译会使用默认的计算能力：
- CUDA 6.0 以及更早的：默认的计算能力是1.0
- CUDA 6.5～CUDA 8.0：默认的计算能力是2.0
- CUDA 9.0～CUDA 10.2：默认的计算能力是3.0
- 目前最新的CUDA版本，默认的计算能力是5.0

最好在编译程序时，为其指定对应的计算能力和对应的设备编译代码，从而发挥出设备的最大性能。

















