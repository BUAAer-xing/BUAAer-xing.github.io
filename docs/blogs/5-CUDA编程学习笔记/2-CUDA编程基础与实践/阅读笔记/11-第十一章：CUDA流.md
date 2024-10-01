
CUDA程序的并行层次主要有两个，一个是核函数内部的并行，一个是核函数外部的并行。

核函数外部的并行主要指：
- 核函数计算与数据传输之间的并行。
- 主机计算与数据传输之间的并行。
- 不同的数据传输之间的并行。
- 核函数计算与主机计算之间的并行
- 不同核函数之间的并行

要获得较高的加速比，需要尽量减少主机与设备之间的数据传输及主机中的计算。

## 11.1 CUDA 流概述

一个CUDA流指的是**由主机发出的在一个设备中执行的CUDA操作序列**。（即和CUDA有关的操作。）

一个CUDA流中各个操作的次序是由主机控制的，按照主机发布的次序进行执行。然而，来自于两个不同CUDA流中的操作不一定按照某个次序执行，而有可能并发或者交错的进行执行。

任何CUDA操作都存在于某个CUDA流中，要么是默认流（default stream）也可以称之为空流（null stream），要么是明确指定的非空流。

CUDA流的定义、产生与销毁如下：
```cpp
cudaStream_t stream_1;
cudaStreamCreate(&stream_1); //注意要传流的地址 
cudaStreamDestroy(stream_1);
```

为了实现不同CUDA流之间的并发，主机在向某个CUDA流中发布一系列命令之后必须马上获得程序的控制权，不用等待该CUDA流中的命令在设备中执行完毕。这样，就可以通过主机产生多个相互独立的CUDA流。

为了检查一个CUDA流中的所有操作是否都在设备中执行完毕，CUDA运行 时API提供了如下两个函数： 
```cpp
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
```
 - 函数`cudaStreamSynchronize()`会强制阻塞主机，直到CUDA流stream中的所有操作都执行完毕。
 - 函数`cudaStreamQuery()`不会阻塞主机，只是检查CUDA 流stream中的所有操作是否都执行完毕。若是，返回cudaSuccess，否则返回 cudaErrorNotReady。


## 11.2 在默认流中重叠主机和设备计算

平常使用的cudaMemcpy是同步的，在使用时，会阻塞掉主机，使得主机空闲。

而调用的核函数是异步的，不会对主机具有阻塞的效果。

当主机和设备的计算量相当时，将主机函数放在设备函数之后可以达到主机函数与设备函数并发执行的效果。所以，当一个主机函数与一个设备函数的计算相互独立时，应该将主机函数的调用放置在核函数的调用之后。从而**充分利用核函数的执行是异步的特点**。

## 11.3 用非默认CUDA流重叠多个核函数的执行

虽然在一个默认流中就可以实现主机计算和设备计算的并行，但是<font color='red'><b>要实现多个核函数之间的并行必须使用多个CUDA流</b></font>。这是因为，同一个CUDA流中的 CUDA操作在设备中是顺序执行的，故**同一个CUDA流中的核函数也必须在设备中顺序执行**，虽然主机在发出每一个核函数调用的命令后都立刻重新获得程序控制权。


### 11.3.1 核函数执行配置中的流参数

如果要想使用非空流但不使用动态共享内存的情况下，必须使用下面的方式：

```cpp
my_kernel<<<N_grid, N_block, 0, stream_id>>>(函数参数);
```

利用CUDA流并发多个核函数可以提升GPU硬件的利用率，减少闲置的SM，从而从整体上获得性能提升。


## 11.4 用非默认CUDA流重叠核函数的执行与数据传递

### 11.4.1 不可分页主机内存与异步的数据传输函数

要实现核函数执行与数据传输的并发（重叠），必须让这两个操作处于不同的非默认流，而且数据传输必须使用`cudaMemcpy()`函数的异步版本，即`cudaMemcpyAsync()`函数。
- <font color='red'><b>异步传输由GPU中的DMA(direct memory access)直接实现，不需要主机参与</b></font>。

在使用异步的数据传输函数时，需要将主机内存定义为**不可分页内存**（nonpageable memory）或者**固定内存**（pinned memory），若主机中的内存声明为不可分页内存，则在程序运行期间，其**物理地址将保持不变**。<font color='red'><b>如果将可分页内存传给cudaMemcpyAsync()函数，则会导致同步传输</b></font>，达不到重叠核函数执行与数据传输的效果。
- 这是因为：主机内存为可分页内存时，数据传输过程**在使用GPU中的DMA之前必须先将数据从可分页内存移动到不可分页内存，从而必须与主机同步**。主机无法在发出数据传输的命令后立刻获得程序的控制权，从而无法实现不同CUDA流之间的并发。

不可分页主机内存的申请和释放：
```cpp
// 申请不可分页内存
cudaError_t cudaMallocHost(void**ptr,size_t size);
cudaError_t cudaHostAlloc(void**ptr,size_t size,size_t flags);

// 释放不可分页内存
cudaError_t cudaFreeHost(void*ptr);
```
