## 1.1 GPU硬件简介

CPU和GPU的显著区别如下： 一块典型的CPU拥有少数几个快速的计算核心，而一块典型的GPU拥有几百到几千个不那么快速的计算核心。CPU中有更多的晶体管用于数据缓存和流程控制， 但**GPU中有更多的晶体管用于算术逻辑单元**。所以，GPU是靠众多的计算核心来获得相对比较高的计算性能的。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240922152913.png)

一块单独的GPU是无法独立完成所有计算任务的，它必须在CPU的调度下才能完成特定任务。在CPU和GPU构成的异构计算平台中，通常将起控制作用的CPU称为主机（host），将起加速作用的GPU称为设备（device）。主机和（非集成）设备都有自己的DRAM，它们之间一般由<font color='red'><b>PCIe总线</b></font>连接。

Nvidia公司提供了多种系列的GPU，主要有以下几个系列：
- **Tesla**系列：其中的内存为纠错内存（ECC），稳定性好，主要用于高性能、高强度的科学计算。
- **Quadro**系列：支持高速OpenGL渲染，主要用于专业的绘图设计。
- **GeForce**系列：主要用于游戏和娱乐，但也常用于科学计算。GeForce系列的GPU没有纠错内存，用于科学计算具有一定的风险。
- **Jetson**系列：嵌入式设备中的GPU。

每一款GPU都有一个用以表示其计算能力的版本号。该版本号可以写为$X.Y$的形式。其中，X表示主版本号，Y表示次版本号。版本号决定了GPU硬件所支持的功能，可为应用程序在运行时判断硬件特征提供依据。这里需要注意的是：<font color='red'><b>计算能力和性能没有简单的正比关系</b></font>。

|      **架构**      | **年份** |                        **示例GPU**                         |            **计算能力**             |                      **主要特性**                       |
| :--------------: | :----: | :------------------------------------------------------: | :-----------------------------: | :-------------------------------------------------: |
|    **Hopper**    |  2022  |           <font color='red'><b>H100</b></font>           |               9.0               |  Transformer引擎，增强的Tensor Cores，改进的多实例GPU，NVLink增强   |
| **Ada Lovelace** |  2022  |                    RTX 4090, RTX 4080                    |               8.9               |       第三代光线追踪核心，第四代Tensor Cores，能效提升，CUDA核心升级       |
|    **Ampere**    |  2020  | <font color='red'><b>A100</b></font>, RTX 3090, RTX 3080 | 8.0 (A100), 8.6 (RTX 3090/3080) |  第三代Tensor Cores，支持稀疏矩阵，增强的多实例GPU (MIG)，改进的CUDA核心   |
|    **Turing**    |  2018  |                  RTX 2080 Ti, RTX 2060                   |               7.5               | **首次引入实时光线追踪，Tensor Cores**，支持NVLink，多GPU配置的高级SLI支持 |
|    **Volta**     |  2017  |        Tesla <font color='red'><b>V100</b></font>        |               7.0               |  首次引入Tensor Cores，专注于AI、高性能计算 (HPC) 和科学计算，FP16性能提升  |
|    **Pascal**    |  2016  |                   GTX 1080, Tesla P100                   |            6.0 - 6.2            |            引入FP16计算能力，相较于之前架构，性能/功耗比大幅提升            |

计算能力并不等价于计算性能。
- 表征计算性能的一个重要参数是**浮点数运算峰值**，即每秒最多能执行的浮点数运算次数。浮点数运算峰值有单精度和双精度之分。
	- 对于Tesla系列的GPU来说，双精度浮点数运算峰值一般是单精度浮点数运算峰值的1/2左右。
	- 对GeForce系列的GPU来说，双精度浮点数运算峰值一般是单精度浮点数运算峰值的1/32左右。
- 另一个影响计算性能的参数是**GPU中的内存带宽**。GPU中的内存常称为显存。
	- **显存容量**也是制约应用程序性能的一个因素。


## 1.2 CUDA程序开发工具

以下几种软件开发工具都可以用来进行GPU编程。
- CUDA：主要使用！！
- OpenCL：这是一个更为通用的为各种异构平台编写并行程序的框架。
- OpenACC：这是一个由多个公司共同开发的异构并行编程标准。

CUDA编程语言最初主要是基于C语言的，但目前越来越多地支持C++语言。

CUDA提供了两层API（application programming interface，应用程序编程接口）供程序员使用。也就是<font color='red'><b>CUDA驱动（driver）API</b></font>和<font color='red'><b>CUDA运行时（runtime）API</b></font>。
- CUDA驱动API是更加底层的API，它为程序员提供了更为灵活的编程接口；
- CUDA运行时API是在CUDA驱动API的基础上构建的一个更为高级的API，更容易使用。
**这两种API在性能上几乎没有差别**。从程序的可读性来看，使用CUDA运行时API是更好的选择。

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240922161056.png)

CUDA版本也由形如$X.Y$的两个数字表示，但它并不等同于GPU的计算能力。

可以这样理解：**CUDA版本是GPU软件开发平台的版本**，而**计算能力对应着 GPU硬件架构的版本**。

## 1.3 CUDA开发环境搭建

略

## 1.4 用nvidia-smi检查与设置设备

![image.png|center|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240922173430.png)


- 第一行是驱动版本和cuda版本
- 该GPU的型号为A100，设备号为0，该计算机仅有一个GPU，如果有多个GPU，会将各个GPU从0开始编号。
	- 如果系统中有多个GPU，而且只需要使用某个特定的GPU，则可以通过设置环境变量`CUDA_VISIBLE_DEVICES`的值在运行CUDA程序之前选定一个GPU。
	- 比如：系统中有两个GPU，想在第二个GPU上运行CUDA程序，则可以设置环境变量为：`export CUDA_VISIBLE_DEVICES=1`
	- 这样设置的<font color='red'><b>环境变量在当前shell session以及子进程中有效</b></font>。
- Compute M. 指的是计算模式。该GPU的计算模式是Default。
	- 在默认模式中，同一个GPU中允许存在多个计算进程，但每个计算进程对应程序的运行速度一般会降低。
	- 还有一种模式为：E. Process，指的是独占进程模式。在该模式下，只能运行一个计算进程独占该GPU。
	- 设置计算模式的命令为：
		- `sudo nvidia-smi -i GPU_ID -c 0 # 默认模式`
		- `sudo nvidia-smi -i GPU_ID -c 1 # 独占进程模式`
		- 其中，`-i GPU_ID`的意思是希望该设置仅仅作用于编号为GPU_ID的GPU。如果没有该项，该设置将会作用于系统中所有的GPU。

## 1.5 其他学习资料

必备网站：https://docs.nvidia.com/cuda/

快速了解一个新的GPU架构，如下图所示：

![image.png|center|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240922174947.png)


















