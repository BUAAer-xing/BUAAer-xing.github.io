
异构计算架构CANN（Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，向上支持多种AI框架，包括MindSpore、PyTorch、TensorFlow等，向下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台。同时针对多样化应用场景，提供多层次编程接口，支持用户快速构建基于昇腾平台的AI应用和业务。

## 1-总体架构

CANN提供了功能强大、适配性好、可自定义开发的AI异构计算架构。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250425144442.png)

- GE图引擎（ Graph Engine）
	- 是计算图编译和运行的控制中心，提供图优化、图编译管理以及图执行控制等功能。GE通过统一的图开发接口提供多种AI框架的支持，不同AI框架的计算图可以实现到Ascend图的转换。
- <font color='red'><b>Ascend C算子开发语言</b></font>
	- 是CANN针对算子开发场景推出的编程语言，原生支持C和C++标准规范，最大化匹配用户开发习惯；通过多层接口抽象、自动并行计算、孪生调试等关键技术，极大提高算子开发效率，助力AI开发者低成本完成算子开发和模型调优部署。
- <font color='red'><b>AOL算子加速库</b></font>（Ascend Operator Library）
	- 提供了丰富的深度优化、硬件亲和的高性能算子，包括神经网络（Neural Network，NN）库、线性代数计算库（Basic Linear Algebra Subprograms，BLAS）等，为神经网络在昇腾硬件上加速计算奠定了基础。
- <font color='red'><b>HCCL集合通信库</b></font>（Huawei Collective Communication Library）
	- 是基于昇腾硬件的高性能集合通信库，提供单机多卡以及多机多卡间的数据并行、模型并行集合通信方案。HCCL支持AllReduce、Broadcast、Allgather、ReduceScatter、AlltoAll等通信原语，Ring、Mesh、HD等通信算法，在HCCS、RoCE和PCIe高速链路实现集合通信。
- BiSheng Compiler毕昇编译器
	- 提供Host-Device异构编程编译能力，利用微架构精准编译优化释放昇腾AI处理器性能，提供完备二进制调试信息与二进制工具链，支撑AI开发者自主调试调优。
- Runtime运行时
	- 提供了高效的硬件资源管理、媒体数据预处理、单算子加载执行、模型推理等开发接口，供开发者轻松构建高性能人工智能应用。

## 2-关键功能特性

- **推理应用开发**
    CANN提供了在昇腾平台上开发神经网络应用的昇腾计算语言AscendCL（Ascend Computing Language），提供运行资源管理、内存管理、模型加载与执行、算子加载与执行、媒体数据处理等API，实现利用昇腾硬件计算资源、在昇腾CANN平台上进行深度学习推理计算、图形图像预处理、单算子加速计算等能力。简单来说，就是统一的API框架，实现对所有资源的调用。
- **模型训练**
    CANN针对训练任务提供了完备的支持，针对PyTorch、TensorFlow等开源框架网络模型，CANN提供了模型迁移工具，支持将其快速迁移到昇腾平台。此外，CANN还提供了多种自动化调测工具，支持数据异常检测、融合异常检测、整网数据比对等，帮助开发者高效问题定位。
- **算子开发**
    CANN提供了超过1400个硬件亲和的高性能算子，可覆盖主流AI框架的算子加速需求，同时，为满足开发者的算法创新需求，CANN开放了自定义算子开发的能力，开发者可根据自身需求选择不同的算子开发方式。














