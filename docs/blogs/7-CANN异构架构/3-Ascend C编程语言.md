
---
大致看了一下，感觉需要记住硬件的特性：

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250425151809.png)

针对AI Core中的<font color='red'><b>矩阵乘法</b></font>：
- 对于`FP16`类型的数据：A、B、C的块大小都是$16\times16$，结果会在一拍中给出。
- 对于`int8`类型的数据：A的块大小是$32\times16$、B的块大小是$16\times32$、C的块大小是$32\times32$。

针对AI Core中的<font color='red'><b>向量运算</b></font>：
- 针对`FP16`类型的数据：一拍可以完成两个**128长度**的向量乘加操作。
- 针对`FP32`类型的数据：一拍可以完成两个**64长度**的向量乘加操作。

---

## 1-Ascend C 简介

Ascend C是CANN针对算子开发场景推出的编程语言，原生支持C和C++标准规范，兼具开发效率和运行性能。基于Ascend C编写的算子程序，通过编译器编译和运行时调度，运行在昇腾AI处理器上。使用Ascend C，开发者可以基于昇腾AI硬件，高效的实现自定义的创新算法。

📒：<font color='red'><b>可以将Ascend C编程类比于CUDA编程</b></font>

![image.png|center|300](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427101836.png)

## 2-使用Ascend C的Demo

### 2.1-核函数输出Hello World

下面是一个简单的Ascend C的"Hello World"样例，展示了一个Ascend C核函数（设备侧实现的入口函数）的基本写法，及其如何被调用的流程。

```cpp
/*
包含核函数的实现文件hello_world.cpp代码如下：核函数hello_world的核心逻辑为打印"Hello World!!!"字符串。hello_world_do封装了核函数的调用程序，通过<<<>>>内核调用符对核函数进行调用。
*/
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void hello_world()
{
    AscendC::printf("Hello World!!!\n");
}

void hello_world_do(uint32_t blockDim, void* stream)
{
    hello_world<<<blockDim, nullptr, stream>>>();
}

#include "acl/acl.h"
extern void hello_world_do(uint32_t coreDim, void* stream);

int32_t main(int argc, char const *argv[])
{
    // AscendCL初始化
    aclInit(nullptr);
    // 运行管理资源申请
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 设置参与运算的核数为8（核数可根据实际需求设置）
    constexpr uint32_t blockDim = 8;
    // 用内核调用符<<<>>>调用核函数，hello_world_do中封装了<<<>>>调用
    hello_world_do(blockDim, stream);
    aclrtSynchronizeStream(stream);
    // 资源释放和AscendCL去初始化
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```

### 2.2-核函数计算两个向量加法Demo

#### 算子分析

Add算子的逻辑表达式是：$\vec{z} = \vec{x} + \vec{y}$，计算逻辑是：从外部存储Global Memory搬运数据至内部存储Local Memory，然后使用Ascend C计算接口完成两个输入参数相加，得到最终结果，再搬运到Global Memory上。

本次demo的数据输入输出为：
- Add算子有两个输入：x与y，输出为z。
- 本样例中算子输入支持的数据类型为half（float16），算子输出的数据类型与输入数据类型相同。
- 算子输入支持的shape为（8，2048），输出shape与输入shape相同。
- 算子输入支持的[format](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0099.html)为：ND。

<center> <font face='华文宋体' size='4'> Ascend C Add算子设计规格 </font> </center>
![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427104544.png)


#### 核函数开发

本样例中使用多核并行计算，即把<font color='red'><b>数据进行分片，分配到多个核上进行处理</b></font>。Ascend C核函数是在一个核上的处理函数，所以只处理部分数据。

分配方案是：假设共启用8个核，数据整体长度`TOTAL_LENGTH`为8 * 2048个元素，平均分配到8个核上运行，每个核上处理的数据大小`BLOCK_LENGTH`为2048个元素。下文的核函数，只关注长度为BLOCK_LENGTH的数据应该如何处理。

核函数的定义如下：

指针入参变量需要增加变量类型限定符`__gm__`，表明该指针变量指向**Global Memory**上某处内存地址。为了统一表达，使用GM_ADDR宏来修饰入参，GM_ADDR宏定义如下：
```cpp
#define GM_ADDR __gm__ uint8_t*
```

使用`__global__`函数类型限定符来标识它是一个核函数，可以被`<<<>>>`调用；使用`__aicore__`函数类型限定符来标识该核函数在设备端**AI Core**上执行。
```cpp
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KernelAdd op;     // 算子类
    op.Init(x, y, z);
    op.Process();
}
```

<font color='red' size='5'><b>算子类的Init函数，完成内存初始化相关工作，Process函数完成算子实现的核心逻辑。</b></font>

##### Demo中算子类的整体框架

```cpp
class KernelAdd {
public:
    __aicore__ inline KernelAdd(){}
    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z){}
    // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Compute、CopyOut完成矢量算子的三级流水操作
    __aicore__ inline void Process(){}

private:
    // 搬入函数，从Global Memory搬运数据至Local Memory，被核心Process函数调用
    __aicore__ inline void CopyIn(int32_t progress){}
    // 计算函数，完成两个输入参数相加，得到最终结果，被核心Process函数调用
    __aicore__ inline void Compute(int32_t progress){}
    // 搬出函数，将最终结果从Local Memory搬运到Global Memory上，被核心Process函数调用
    __aicore__ inline void CopyOut(int32_t progress){}

private:
    AscendC::TPipe pipe;  //TPipe内存管理对象
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;  //输入数据Queue队列管理对象，TPosition为VECIN
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;  //输出数据Queue队列管理对象，TPosition为VECOUT
    AscendC::GlobalTensor<half> xGm;  //管理输入输出Global Memory内存地址的对象，其中xGm, yGm为输入，zGm为输出
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
};

```

##### 算子类中的Init函数

初始化函数**Init**主要完成以下内容：设置输入输出Global Tensor的Global Memory内存地址，通过TPipe内存管理对象为输入输出Queue分配内存。

<font color='red'><b>这里的Global Tensor作用就是将Global Memory中的数据进行切分，计算出每个Core应该负责的Global Memory的内存地址。 </b></font>

<font color='red'><b>GmAlloc和GmFree，而没有Gmmemcpy函数，因为通过这种方式申请的内存是在系统/tmp目录下生成的临时文件，并不是在卡里面的，是存储在硬盘上的。</b></font>

<font color='red'><b>aclrtMalloc、aclrtMemcpy等函数，也是利用__gm__指针，将主机内存中的数据拷贝到设备内存上去</b></font>


**本样例将数据切分成8块，平均分配到8个核上运行，每个核上处理的数据大小BLOCK_LENGTH为2048个元素。数据切分主要通过地址偏移来进行实现：** 每个核上处理的数据地址需要在起始地址上增加`GetBlockIdx() * BLOCK_LENGTH`（每个block处理的数据长度）的<font color='red'><b>偏移</b></font>来获取。这样也就实现了多核并行计算的数据切分。

以输入x为例，`x + BLOCK_LENGTH * GetBlockIdx()`即为单核处理程序中x在Global Memory上的内存偏移地址，获取偏移地址后，使用GlobalTensor类的[SetGlobalBuffer](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0007.html)接口设定该核上Global Memory的起始地址以及长度。具体示意图如下。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427105727.png)

实现了在多核上的数据切分后，还需要在单核上进行进一步的数据切分（这是为什么？**猜测可能是因为每个AI Core中的矩阵计算单元和向量计算单元都是固定大小的，每次切分出来都都得是符合这些硬件单元大小的切分**）

对于单核上的处理数据，可以进行数据切块（Tiling），在本示例中，仅作为参考，将数据切分成8块（并不意味着8块就是性能最优）。切分后的每个数据块再次切分成2块，即可开启[double buffer](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0090.html)，实现流水线之间的并行。<font color='red'><b>可以将数据传入传出和计算过程进行重叠起来～</b></font>

![image.png|center|300](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427110311.png)
<center> <font face='华文宋体' size='4'>双缓冲策略</font> </center>

这样单核上的数据（2048个数）被切分成16块，每块TILE_LENGTH（128）个数据。TPipe为inQueueX分配了两块大小为TILE_LENGTH * sizeof(half)个字节的内存块，**每个内存块能容纳TILE_LENGTH（128）个half类型数据**。数据切分示意图如下：

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427110118.png)

```cpp
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // separate to 2 parts, due to double buffer

// 2048 / 16 = 128 符合了向量计算单元的大小，同时开启了双缓冲策略。

__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    // get start index for current core, core parallel
    xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    yGm.SetGlobalBuffer((__gm__ half*)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    zGm.SetGlobalBuffer((__gm__ half*)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
}
```


##### 算子类中的Process函数

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427112725.png)

基于矢量编程范式，将核函数的实现分为3个基本任务：CopyIn，Compute，CopyOut。**Process**函数中通过如下方式调用这三个函数：

```cpp
__aicore__ inline void Process()
{
    // loop count need to be doubled, due to double buffer
    constexpr int32_t loopCount = TILE_NUM * BUFFER_NUM;
    // 这里之所以loopCount长这样，是因为每次loop都在一个AI Core上计算一次128位的向量加法。
    // tiling strategy, pipeline parallel
    for (int32_t i = 0; i < loopCount; i++) {
	    // 相当于对每个AI Core中的数据通路进行了更加精细的控制
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}
```

##### CopyIn函数
1. 使用[DataCopy](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0101.html)接口将GlobalTensor数据拷贝到LocalTensor。
2. 使用[EnQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0140.html)将LocalTensor放入VecIn的Queue中。

```cpp
__aicore__ inline void CopyIn( int32_t progress)
{
    // alloc tensor from queue memory
    AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    // copy progress_th tile from global tensor to local tensor
    AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
    AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
    // enque input tensors to VECIN queue
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}
```

##### Compute函数
1. 使用[DeQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0141.html)从VecIn中取出LocalTensor。
2. 使用Ascend C接口[Add](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0035.html)完成矢量计算。
3. 使用[EnQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0140.html)将计算结果LocalTensor放入到VecOut的Queue中。
4. 使用[FreeTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0139.html)将释放不再使用的LocalTensor。

```cpp
__aicore__ inline void Compute(int32_t progress)
{
    // deque input tensors from VECIN queue
    AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
    // call Add instr for computation
    AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
    // enque the output tensor to VECOUT queue
    outQueueZ.EnQue<half>(zLocal);
    // free input tensors for reuse
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
}
```

##### CopyOut函数

1. 使用[DeQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0141.html)接口从VecOut的Queue中取出LocalTensor。
2. 使用[DataCopy](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0101.html)接口将LocalTensor拷贝到GlobalTensor上。
3. 使用[FreeTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0139.html)将不再使用的LocalTensor进行回收。

```cpp
 __aicore__ inline void CopyOut(int32_t progress)
{
    // deque output tensor from VECOUT queue
    AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
    // copy progress_th tile from local tensor to global tensor
    AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
    // free output tensor for reuse
    outQueueZ.FreeTensor(zLocal);
}
```


#### 核函数运行验证

异构计算架构中，NPU（kernel侧）与CPU（host侧）是协同工作的，完成了kernel侧核函数开发后，即可编写host侧的核函数调用程序，实现从host侧的APP程序调用算子，执行计算过程。

##### 异构计算框架

内置宏`ASCENDC_CPU_DEBUG`是区分运行CPU模式或NPU模式逻辑的标志，在同一个main函数中通过对`ASCENDC_CPU_DEBUG`宏定义的判断来区分CPU模式和NPU模式的运行程序。

```cpp
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void add_custom_do(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z);
#endif

int32_t main(int32_t argc, char* argv[])
{
    size_t inputByteSize = 8 * 2048 * sizeof(uint16_t);  // uint16_t represent half
    size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);  // uint16_t represent half
    uint32_t blockDim = 8;

#ifdef ASCENDC_CPU_DEBUG
    // 用于CPU模式调试的调用程序
    
#else
    // NPU模式运行算子的调用程序

#endif
    return 0;
}
```

##### CPU调试用的调试程序

![image.png|center|200](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427114644.png)


```cpp
    // 使用GmAlloc分配共享内存，并进行数据初始化
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, y, inputByteSize);
    // 调用ICPU_RUN_KF调测宏，完成核函数CPU模式调用
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_custom, blockDim, x, y, z); // use this macro for cpu debug
    // 输出数据写出
    WriteFile("./output/output_z.bin", z, outputByteSize);
    // 调用GmFree释放申请的资源
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
```


##### NPU模式运行算子调试程序

![image.png|center|150](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427114807.png)


```cpp
    // AscendCL初始化
    CHECK_ACL(aclInit(nullptr));
    // 运行管理资源申请
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    // 分配Host内存
    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;
    CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputByteSize));
    // 分配Device内存
    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // Host内存初始化
    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 用内核调用符<<<>>>调用核函数完成指定的运算,add_custom_do中封装了<<<>>>调用
    add_custom_do(blockDim, nullptr, stream, xDevice, yDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // 将Device上的运算结果拷贝回Host
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);
    // 释放申请的资源
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    // AscendCL去初始化
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
```


## 3-硬件架构

### 3.1-基本架构

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427120229.png)

AI Core负责执行标量、向量和张量相关的计算密集型算子，包括三种基础**计算单元**：Cube（矩阵）计算单元、Vector（向量）计算单元和Scalar（标量）计算单元，同时还包含**存储单元**（包括硬件存储和用于数据搬运的搬运单元）和**控制单元**。硬件架构根据Cube计算单元和Vector计算单元是否同核部署分为**耦合架构**和**分离架构**两种。

#### 耦合架构

耦合架构是指Cube计算单元和Vector计算单元同核部署，架构图如下图所示，耦合架构中Cube计算单元和Vector计算单元共享同一个Scalar单元，统一加载所有的代码段。
![image.png|center|500](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427120321.png)


#### 分离架构

分离架构将AI Core拆成矩阵计算（AI Cube，AIC）和向量计算（AI Vector，AIV）两个独立的核，每个核都有自己的Scalar单元，能独立加载自己的代码段，从而实现矩阵计算与向量计算的解耦，在系统软件的统一调度下互相配合达到计算效率优化的效果。AIV与AIC之间通过Global Memory进行数据传递。另外分离架构相比耦合架构，增加了两个Buffer：BT Buffer(BiasTable Buffer，存放Bias)和FP Buffer(Fixpipe Buffer，存放量化参数、Relu参数等)。

![image.png|center|500](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427120344.png)



### 3.2-计算单元

计算单元是AI Core中提供强大算力的核心单元，包括三种基础**计算单元**：Cube（矩阵）计算单元、Vector（向量）计算单元和Scalar（标量）计算单元，完成AI Core中不同类型的数据计算。

#### 标量计算单元

在AI Core架构中，**Scalar**主要负责各类标量数据运算与程序流程控制，功能上相当于一个简化版的小型CPU。<font color='red'><b>它承担循环控制、分支判断、Cube/Vector指令地址和参数计算、基本算术运算</b></font>等任务，同时能够通过事件同步模块插入同步符，控制AI Core内部其他执行单元的流水操作。需要注意的是，相比Host CPU，AI Core中的Scalar计算能力较弱，主要定位在指令发射层面，因此在实际应用中，应尽量减少Scalar上的计算负担，尤其是在性能调优时，要避免频繁的`if/else`分支和变量计算。

![image.png|center|300](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427120712.png)

从执行机制上看，Scalar在执行标量运算指令时，依赖标准的**ALU（Arithmetic Logic Unit）** 执行具体操作。其所需的代码段和数据段（栈空间）均来源于**GM（Global Memory）** 。为了提高效率，Scalar配备了**ICache（Instruction Cache）** 用于缓存代码段和**DCache（Data Cache）** 用于缓存数据段。ICache的大小通常与硬件规格有关，如16K或32K字节，按2K字节为单位加载；DCache大小同样与硬件规格相关，典型值为16K字节，按**cacheline（64字节）** 为单位加载。为最大化核内访问效率，应尽可能确保程序的代码段和数据段能够被完全缓存至ICache和DCache中，减少核外访问带来的延迟。此外，在DCache加载数据时，如果<font color='red'><b>内存首地址与cacheline对齐</b></font>（即64字节对齐），则数据加载效率最高。因此在程序设计和内存布局时，应特别关注对齐优化，以提升整体数据加载效率和程序执行性能。

#### 向量计算单元

Vector负责执行向量运算。向量计算单元执行向量指令，类似于传统的单指令多数据（Single Instruction Multiple Data，SIMD）指令，每个向量指令可以完成多个操作数的同一类型运算。向量计算单元可以快速完成两个FP16类型的向量相加或者相乘。向量指令支持多次迭代执行，也支持对带有间隔的向量直接进行运算。

![image.png|center|300](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427120758.png)

Vector所有计算的源数据以及目标数据都要求存储在[Unified Buffer](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0010.html#ZH-CN_TOPIC_0000002263815169__table1692510612218)中，Vector指令的首地址和操作长度有对齐要求，通常要求32B对齐，具体对齐要求参考API的约束描述。


#### 矩阵计算单元

Cube计算单元负责执行矩阵运算，一次执行即可完成A矩阵（M * K）与B矩阵（K * N）的矩阵乘。如下图所示红色虚线框划出了Cube计算单元及其访问的存储单元，其中L0A存储左矩阵，L0B存储右矩阵，L0C存储矩阵乘的结果和中间结果。

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427120822.png)

### 3.3-存储单元

AI Core中包含多级**内部存储**，AI Core需要把**外部存储**中的数据加载到内部存储中，才能完成相应的计算。AI Core的主要内部存储包括：L1 Buffer（L1缓冲区），L0 Buffer（L0缓冲区），Unified Buffer（统一缓冲区）等。

为了配合AI Core中的数据传输和搬运，AI Core中还包含<font color='red'><b>MTE</b></font>（Memory Transfer Engine，存储转换引擎）搬运单元，在搬运过程中可执行随路数据格式/类型转换。



### 3.4-控制单元

控制单元为整个计算过程提供了指令控制，负责整个AI Core的运行。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427121007.png)

多条指令通过总线接口从系统内存进入到指令缓存（ICache）。
在后续指令执行过程中：
- 若为Scalar指令，直接由Scalar单元执行；
- 若为其他类型指令，则由Scalar单元调度至五个独立分类指令队列（Vector指令队列、Cube指令队列、MTE1/MTE2/MTE3指令队列），由对应的执行单元处理。<font color='red'><b>单个指令队列内部严格按照指令进入顺序执行，不同指令队列之间可以并行执行，以提高整体执行效率</b></font>。
为了处理并行执行中潜在的数据依赖问题，**系统通过事件同步模块插入同步指令**，提供PipeBarrier与SetFlag/WaitFlag两种API实现同步控制。

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427121050.png)

PipeBarrier作为一条同步指令，用于约束单个指令队列内的执行顺序，确保所有前序指令的数据读写操作完成后，后序指令方可执行。SetFlag和WaitFlag则实现不同指令队列之间的同步关系，其中SetFlag指令在完成前序指令所有读写后设置同步标志位为1；WaitFlag指令在检测到标志位为0时阻塞后续指令，待标志位为1后自动清零并继续执行。

Ascend C提供了同步控制API，通常开发者无需手动管理同步，编程模型及范式会自动处理同步控制，这是推荐的开发方式。但了解同步机制的原理仍然重要，尤其是在特殊情况下需要手动插入同步指令，以确保并行程序的正确性和高效性。



## 4-编程模型

### 4.1-SPMD模型（AI Core之间的并行性来源）

假设，从输入数据到输出数据需要经过3个阶段任务的处理（T1、T2、T3）。如下图所示，SPMD模式下，系统会启动一组进程，并行处理待处理的数据：**首先待处理数据会被切分成多个数据分片，切分后的数据分片随后被分发给不同进程处理，每个进程接收到一个或多个数据分片，并独立地对这些分片进行3个任务的处理**。

具体到Ascend C编程模型中的应用，是<font color='red'><b>将需要处理的数据拆分并同时在多个计算核心（类比于上文介绍中的多个进程）上运行</b></font>，从而获取更高的性能。多个AI Core共享相同的指令代码，**每个核上的运行实例唯一的区别是block_idx不同，每个核通过不同的block_idx来识别自己的身份**。block的概念类似于上文中进程的概念，block_idx就是标识进程唯一性的进程ID。

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427121531.png)

算子被调用时，所有的计算核心都执行相同的实现代码，入口函数的入参也是相同的。每个核上处理的数据地址需要在起始地址上增加`GetBlockIdx()*BLOCK_LENGTH`（每个核处理的数据长度）的<font color='red'><b>偏移</b></font>来获取。这样也就实现了多核并行计算的数据切分。

```cpp
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        // 不同核根据各自的block_idx设置数据地址
        xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half*)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half*)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        // Queue初始化，单位为字节
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }
    ...
}

// 实现核函数
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    // 初始化算子类，算子类提供算子初始化和核心处理等方法
    KernelAdd op;
    // 初始化函数，获取该核函数需要处理的输入输出地址，同时完成必要的内存初始化工作
    op.Init(x, y, z);
    // 核心处理函数，完成算子的数据搬运与计算等核心逻辑
    op.Process();
}
```

### 4.2-核函数

和cuda的调用方式类似，只不过启用的不再是线程，而是一个block对应一个AI Core，类似于一个进程。

```cpp
// 实现核函数
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z)
{
    // 初始化算子类，算子类提供算子初始化和核心处理等方法
    KernelAdd op;
    // 初始化函数，获取该核函数需要处理的输入输出地址，同时完成必要的内存初始化工作
    op.Init(x, y, z);
    // 核心处理函数，完成算子的数据搬运与计算等核心逻辑
    op.Process();
}

// 调用核函数
void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z);
}
```

注意：**blockDim**，规定了核函数将会在几个核上执行。每个执行该核函数的核会被分配一个逻辑ID，即block_idx，可以在核函数的实现中调用[GetBlockIdx](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0185.html)来获取block_idx；blockDim是逻辑核的概念，取值范围为[1,65535]。为了充分利用硬件资源，**一般设置为物理核的核数或其倍数**。对于耦合架构和分离架构，blockDim在运行时的意义和设置规则有一些区别，具体说明如下：
- 耦合架构：由于其Vector、Cube单元是集成在一起的，blockDim用于设置启动多个AICore核实例执行，不区分Vector、Cube。AI Core的核数可以通过[GetCoreNumAiv](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_1031.html)或者[GetCoreNumAic](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_1030.html)获取。
- 分离架构
	- 针对仅包含Vector计算的算子，blockDim用于设置启动多少个Vector（AIV）实例执行，比如某款AI处理器上有40个Vector核，建议设置为40。
	- 针对仅包含Cube计算的算子，blockDim用于设置启动多少个Cube（AIC）实例执行，比如某款AI处理器上有20个Cube核，建议设置为20。
	- 针对Vector/Cube融合计算的算子，启动时，按照AIV和AIC组合启动，blockDim用于设置启动多少个组合执行，比如某款AI处理器上有40个Vector核和20个Cube核，一个组合是2个Vector核和1个Cube核，建议设置为20，此时会启动20个组合，即40个Vector核和20个Cube核。**注意：该场景下，设置的blockDim逻辑核的核数不能超过物理核（2个Vector核和1个Cube核组合为1个物理核）的核数。**
	- AIC/AIV的核数分别通过[GetCoreNumAic](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_1030.html)和[GetCoreNumAiv](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_1031.html)接口获取。

### 4.3-硬件架构的抽象

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427122218.png)

AI Core中包含**计算单元、存储单元、搬运单元**等核心组件。
- 计算单元包括了三种基础计算资源：Cube计算单元、Vector计算单元和Scalar计算单元。
- 存储单元包括内部存储和外部存储：
    - AI Core的内部存储，统称为Local Memory，对应的数据类型为LocalTensor。由于不同芯片间硬件资源不固定，可以为UB、L1、L0A、L0B等。
    - AI Core能够访问的外部存储称之为Global Memory，对应的数据类型为GlobalTensor。
- DMA（Direct Memory Access）搬运单元：负责数据搬运，包括Global Memory和Local Memory之间的数据搬运，以及不同层级Local Memory之间的数据搬运。

### 4.4-编程范式

**编程范式描述了算子实现的固定流程，基于编程范式进行编程，可以快速搭建算子实现的代码框架。**

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427142122.png)

Ascend C编程范式就是一种流水线式的编程范式，把算子核内的处理程序，分成多个**流水任务**，通过队列（Queue）完成**任务间通信和同步**，并通过统一的**资源管理**模块（Pipe）来统一管理内存、事件等资源。


#### Vector 编程范式

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427142355.png)

如上图所示，Vector编程范式把算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。
- **CopyIn**负责搬入操作：将输入数据从Global Memory搬运到Local Memory（VECIN用于表达矢量计算搬入数据的存放位置），完成搬运后执行入队列操作；
- **Compute**负责矢量指令计算操作：完成队列出队后，从Local Memory获取数据并计算，计算完成后执行入队操作；
- **CopyOut**负责搬出操作：完成队列出队后，将计算结果从Local Memory（VECOUT用于表达矢量计算搬出数据的存放位置）搬运到GM。

上文中提到的VECIN/VECOUT是[TPosition](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0174.html)的概念。Ascend C管理不同层级的物理内存时，**用一种抽象的逻辑位置（TPosition）来表达各级别的存储，代替了片上物理存储的概念**，达到隐藏硬件架构的目的。除了VECIN/VECOUT，矢量编程中还会使用到VECCALC，一般在定义临时变量时使用此位置。这些TPosition与物理内存的映射关系如下表。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427142451.png)

详细的流程为：

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427142613.png)

vector编程范式的指令流程为：

![image.png|center|500](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427143409.png)


任务间数据传递使用到的内存、事件等资源统一由管理模块Pipe进行管理。如下所示的内存管理示意图，TPipe通过[InitBuffer](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0110.html)接口对外提供Queue内存初始化功能，开发者可以通过该接口为指定的Queue分配内存。

Queue队列内存初始化完成后，需要使用内存时，通过调用[AllocTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0138.html)来为LocalTensor分配内存，当创建的LocalTensor完成相关计算无需再使用时，再调用[FreeTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0139.html)来回收LocalTensor的内存。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427142643.png)


从运行图中可以看出，对于同一片数据，Stage1、Stage2、Stage3之间的处理具有依赖关系，需要串行处理；不同的数据切片，同一时间点，可以有多个任务在并行处理，由此达到任务并行、提升性能的目的。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427142719.png)


#### Cube 编程范式

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427142819.png)

和矢量编程范式一样，同样也使用逻辑位置（TPosition）来表达数据流，Cube编程范式中主要使用的逻辑位置定义如下：
- 搬入数据的存放位置：A1，用于存放整块A矩阵，可类比CPU多级缓存中的二级缓存；
- 搬入数据的存放位置：B1，用于存放整块B矩阵，可类比CPU多级缓存中的二级缓存；
- 搬入数据的存放位置：C1，用于存放整块的矩阵乘偏置Bias矩阵，可类比CPU多级缓存中的二级缓存；
- 搬入数据的存放位置：A2，用于存放切分后的小块A矩阵，可类比CPU多级缓存中的一级缓存；
- 搬入数据的存放位置：B2，用于存放切分后的小块B矩阵，可类比CPU多级缓存中的一级缓存；
- 搬入数据的存放位置：C2，用于存放切分后的小块矩阵乘偏置Bias矩阵，可类比CPU多级缓存中的一级缓存；
- 结果数据的存放位置：CO1，用于存放小块结果C矩阵，可理解为Cube Out；
- 结果数据的存放位置：CO2，用于存放整块结果C矩阵，可理解为Cube Out；
- 搬入数据的存放位置：VECIN，用于矢量计算，实际业务在数据搬入Vector计算单元时使用此位置；
- 搬入数据的存放位置：VECCALC，用于矢量计算，实际业务一般在计算需要临时变量时使用此位置；
- 搬出数据的存放位置：VECOUT，用于矢量计算，实际业务在将Vector计算单元结果搬出时使用此位置。

Cube计算流程同样也可以理解为CopyIn、Compute、CopyOut这几个阶段，因为流程相对复杂，<font color='red'><b>Matmul高阶API提供对此的高阶封装，简化了编程范式</b></font>。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427142912.png)

#### 融合算子编程范式

支持Vector与Cube混合计算的算子称之为融合算子。Ascend C提供**融合算子的编程范式**，方便开发者基于该范式表达融合算子的数据流，快速实现自己的融合算子。**融合算子数据流**指融合算子的输入输出在各存储位置间的流向。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427143130.png)

基于Matmul高阶API的融合算子编程范式，对上述数据流简化表达如下：

![center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427143147.png)

流程如下：
1. 初始化一个MatMul对象，将输入数据从Global Memory搬运到Cube核上。
2. 进行MatMul内部的计算。
3. 将MatMul的计算结果搬运到Vector核上。
4. 进行Vector矢量计算。
5. 将输出结果搬运到Global Memory上。


## 5-编程API

Ascend C提供一组类库API，开发者使用标准C++语法和类库API进行编程。Ascend C编程类库API示意图如下所示，分为：
- **Kernel API**：用于实现算子核函数的API接口。包括：
    - **基本数据结构：** kernel API中使用到的基本数据结构，比如GlobalTensor和LocalTensor。
    - **基础API：** 实现对硬件能力的抽象，开放芯片的能力，保证完备性和兼容性。标注为ISASI（Instruction Set Architecture Special Interface，硬件体系结构相关的接口）类别的API，不能保证跨硬件版本兼容。
    - **高阶API：** 实现一些常用的计算算法，用于提高编程开发效率，通常会调用多种基础API实现。高阶API包括数学库、Matmul、Softmax等API。高阶API可以保证兼容性。
- **Host API**：
    - 高阶API配套的Tiling API：kernel侧高阶API配套的Tiling API，方便开发者获取kernel计算时所需的Tiling参数。
    - Ascend C算子原型注册与管理API：用于Ascend C算子原型定义和注册的API。
    - Tiling数据结构注册API：用于Ascend C算子TilingData数据结构定义和注册的API。
    - 平台信息获取API：在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。平台信息获取API提供获取这些平台信息的功能。
- **算子调测API**：用于算子调测的API，包括孪生调试，性能调测等。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427143710.png)

**对于基础API，主要分为以下几类：**
- **[标量计算API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0003.html#ZH-CN_TOPIC_0000002263727309__table339023582010)**，实现调用Scalar计算单元执行计算的功能。
- **[矢量计算API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0003.html#ZH-CN_TOPIC_0000002263727309__table107281858237)**，实现调用Vector计算单元执行计算的功能。
- **[矩阵计算API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0236.html)**，实现调用Cube计算单元执行计算的功能。
- **[数据搬运API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0003.html#ZH-CN_TOPIC_0000002263727309__table1199372172410)**，计算API基于Local Memory数据进行计算，所以数据需要先从Global Memory搬运至Local Memory，再使用计算API完成计算，最后从Local Memory搬出至Global Memory。执行搬运过程的接口称之为数据搬运API，比如DataCopy接口。
- **[内存管理与同步控制API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0003.html#ZH-CN_TOPIC_0000002263727309__table1267664316264)**
    - 内存管理API，用于分配管理内存，比如AllocTensor、FreeTensor接口;
    - 同步控制API，完成任务间的通信和同步，比如EnQue、DeQue接口。不同的API指令间有可能存在依赖关系，从[硬件架构抽象](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0015.html)可知，不同的指令异步并行执行，为了保证不同指令队列间的指令按照正确的逻辑关系执行，需要向不同的组件发送同步指令。同步控制API内部即完成这个发送同步指令的过程，开发者无需关注内部实现逻辑，使用简单的API接口即可完成。




















