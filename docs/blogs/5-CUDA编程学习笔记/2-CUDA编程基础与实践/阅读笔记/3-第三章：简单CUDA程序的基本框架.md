## 3.1 例子：数组相加

略

注意：在判断两个浮点数是否相等时，不能运用运算符`==`，而是要将这两个数的差的绝对值与一个很小的数进行比较。

## 3.2 CUDA程序的基本框架

基本框架如下：

```cpp
// 头文件包含
#include <cuda_runtime.h>
#include <iostream>

// 常量定义（或者宏定义）
#define N 1024

// C++ 自定义函数和 CUDA 核函数的声明（原型）
__global__ void kernel_function();

// int main(void)
int main(void)
{
    // 分配主机与设备内存
    int* host_memory;
    int* device_memory;
    cudaMalloc((void**)&device_memory, N * sizeof(int));
    host_memory = (int*)malloc(N * sizeof(int));

    // 初始化主机中的数据
    for (int i = 0; i < N; i++) {
        host_memory[i] = i;
    }

    // 将某些数据从主机复制到设备
    cudaMemcpy(device_memory, host_memory, N * sizeof(int), cudaMemcpyHostToDevice);

    // 调用核函数在设备中进行计算
    kernel_function<<<1, N>>>();

    // 将某些数据从设备复制到主机
    cudaMemcpy(host_memory, device_memory, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 释放主机与设备内存
    cudaFree(device_memory);
    free(host_memory);

    return 0;
}

// C++ 自定义函数和 CUDA 核函数的定义（实现）
__global__ void kernel_function() {
    // 核函数的内容
}
```


### 3.2.1 隐形的设备初始化

在CUDA运行时API中，没有明显地初始化设备（即GPU)的函数。<font color='red'><b>在第一次调用一个和设备管理及版本查询功能无关的运行时API函数时，设备将自动初始化</b></font>。

1. **自动初始化设备**：当在 CUDA 程序中第一次调用一个与设备无关的函数时（例如：内存分配函数 `cudaMalloc()` 或核函数调用` kernel<<<>>>`），CUDA 运行时会自动进行设备的初始化。这意味着你无需手动调用某个初始化函数，CUDA 会在需要时自动选择和初始化 GPU。
2. **与设备管理及版本查询无关的函数**：这里的“与设备管理及版本查询无关”是指一些 CUDA API 函数，比如 `cudaGetDevice()` 或 `cudaDeviceProp`，这些函数是专门用于查询或管理设备信息的，它们本身并不会触发 GPU 的自动初始化。只有当你调用真正涉及到 GPU 计算或内存分配的函数时，GPU 才会被初始化。

所以，在进行函数的测速之前，一定要先进行<font color='red'><b>warm up</b></font>的操作，避免测速的误差。


### 3.2.2 设备内存的分配与释放

略

### 3.2.3 主机与设备之间数据的传递

略

### 3.2.4 核函数中数据与线程的对应

略

### 3.2.5 核函数的要求

这里值得注意的有几点：
- 可以向核函数传递非指针变量，其内容对每个线程都可见。
- 除非使用统一内存编程机制，否则传给和函数的数组（指针）必须指向设备内存。

### 3.2.6 核函数中if语句的必要性

需要通过条件语句规避不需要的线程操作。

## 3.3 自定义设备函数

核函数可以调用不带执行配置的自定义函数，这样的自定义函数称为设备函数(device function)。它是在设备中执行，并在设备中被调用的。与之相比，核函数是在设备中执行，但在主机端被调用的。

### 3.3.1 函数执行空间标识符

1. 用 `__global__` 修饰的函数称为核函数，一般由主机调用，在设备中执行。如果使用动态并行，则也可以在核函数中调用自己或其他核函数。
2. 用 `__device__` 修饰的函数称为设备函数，只能被核函数或其他设备函数调用，在设备中执行。
3. 用 `__host__` 修饰的函数就是主机端的普通 C++ 函数，在主机中被调用，在主机中执行。对于主机端的函数，该修饰符可省略。之所以提供这样一个修饰符，是因为有时可以用 `__host__` 和 `__device__` 同时修饰一个函数，使得该函数既是一个 C++ 中的普通函数，又是一个设备函数。这样做可以减少冗余代码。编译器将针对主机和设备分别编译该函数。

注意：编译器决定把设备函数当作内联函数或者非内联函数。但是：
- 可以用修饰符`__noinline__`建议把一个设备函数作为非内联函数（编译器不一定接受）
- 也可以用修饰符`__forceinline__`建议一个设备函数作为内联函数。















