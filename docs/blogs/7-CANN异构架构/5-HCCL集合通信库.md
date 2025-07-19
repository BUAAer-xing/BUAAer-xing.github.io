## 0-什么是HCCL？

### 简介

HCCL(Huawei Collective Communication Library)是基于昇腾AI处理器的高性能集合通信库， 提供**单机多卡**以及**多机多卡**间的集合通信能力，支持大模型的数据并行、模型并行、专家并行、 pipeline并行、序列并行等多种加速方案。 HCCL支持AllReduce、Broadcast、Allgather、ReduceScatter、AlltoAll等通信原语，支持Ring、 Mesh、Halving-Doubling(HD)等通信算法，支持基于HCCS、RoCE和PCle等链路协议实现集合通信，未来还将支持更多链路协议。 HCCL集合通信库作为异腾AI异构计算架构CANN的重要组成部分，为异腾超大集群训练/推理 提供了基础通信能力

### 特点

HCCL主要用于需要多个NPU协同工作的高性能计算任务，例如深度学习训练、大规模数据分析和科学计 算等。通过使用HCCL，这些应用可以更有效地利用NPU资源，缩短计算时间，提高工作效率。 
1. **高性能集合通信算法，提升大规模并行计算通信效率**
	- HCCL会根据服务器内、服务器间的基础拓扑，自动选择合适的通信算法，包括常见的标准算法以及多种 自研算法，也支持接入用户开发实现的通信算法。
2. **计算通信统一硬化调度，降低调度开销，优化硬件资源利用率**
	- 专用硬件调度引擎和硬件通信原语，实现计算任务与通信任务全硬化调度，降低调度开销，精准控制系统抖动。 
3. 计算通信高性能并发，**计算与通信并发流水执行**，系统性能进一步提升 
	- “归约“类集合通信操作（AllReduce、ReduceScatter、Reduce)通过随路方式实现，不占用计算资源。 计算通信任务并发执行，总执行时长大幅降低。


## 1-HCCL接口简介

HCCL提供了C与Python两种语言的接口，用于实现分布式能力。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427150943.png)

- C语言接口用于实现单算子模式下的框架适配。
    针对PyTorch框架网络，HCCL单算子API已嵌入到Ascend Extension for PyTorch后端代码中，开发者直接使用PyTorch原生集合通信API，即可实现分布式能力。
- Python语言的接口用于实现图模式下的框架适配，当前仅用于TensorFlow网络在昇腾AI处理器执行分布式优化。

## 2-HCCL的术语以及相关概念

| 缩写      | 全称及说明                                                                                                |
| :------ | :--------------------------------------------------------------------------------------------------- |
| NPU     | Neural Network Processing Unit，神经网络处理单元。采用“数据驱动并行计算”的架构，特别擅长处理视频、图像类的海量多媒体业务数据，专门用于处理人工智能应用中的大量计算任务。 |
| HCCL    | Huawei Collective Communication Library，华为集合通信库。提供单机多卡以及多机多卡间的数据并行、模型并行集合通信方案。                       |
| HCCS    | Huawei Cache Coherence System，华为缓存一致性系统。HCCS用于CPU/NPU之间的高速互联。                                        |
| HCCP    | Huawei Collective Communication adaptive Protocol，集合通信适配协议。提供跨NPU设备通信能力，向上屏蔽具体通讯协议差异。                |
| TOPO    | 拓扑、拓扑结构。一个局域网内或者多个局域网之间的设备连接所构成的网络配置或者布局。                                                            |
| PCIe    | Peripheral Component Interconnect Express，一种串行外设扩展总线标准，通常用于计算机系统中外设扩展使用。                             |
| PCIe-SW | PCIe Switch，符合PCIe总线扩展的交换设备。                                                                         |
| AI节点    | 昇腾AI节点，又称昇腾AI Server，通常是8卡或16卡的昇腾NPU设备组成的服务器形态的统称。                                                   |
| AI集群    | 多个AI节点通过交换机（Switch）互联后用于分布式训练或推理的系统。                                                                 |
| 通信域     | 包含了一组参与通信的NPU设备以及设备对应的通信过程。                                                                          |

| 缩写        | 全称及说明                                                                                                                                                                                                |
| :-------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SDMA**  | SDMA（System Direct Memory Access），系统直接内存访问，是一种异步数据传输技术，可以在不占用CPU资源的情况下，实现数据在内存和外设间的告诉传输。在昇腾AI处理器的多个异构计算单元中，**SDMA通常用于在多个不同计算单元之间传输数据**，以提高系统整体的计算效率和性能。                                              |
| **RDMA**  | RDMA（Remote Direct Memory Access）技术全流程直接数据存取，就是为了减少网络传输中服务器端数据处理的延迟而产生的。RDMA通过网络把资料直接传入计算机的存储区，将数据从一个系统快速移动到远程系统存储器中，而不对操作系统造成任何影响，这样就不需要用到多少计算机的处理功能。它消除了外部存储复制和上下文切换的开销，因而能解放内存带宽和CPU周期用于改进应用系统性能。 |
| P（作为单位）   | 指NPU数量，如8P即指代8个NPU。                                                                                                                                                                                  |
| RANK      | 卡、NPU，集合通信进程实体，在整个通信域内全局唯一。                                                                                                                                                                          |
| DEVICE_ID | 物理的卡编号，在一个节点（一个服务器）内唯一。                                                                                                                                                                              |
| **RoCE**  | RoCE全称RDMA over Converged Ethernet，是一种允许**在以太网上实现远程内存直接访问的网络协议**。基于以太网速率发展优势，利用RDMA（Remote Direct Memory Access）技术，可以在极少占用CPU资源的情况下，实现服务器之间的高速数据访问，提供大带宽、低时延的远程内存访问能力，适用于AI计算、高性能计算、高速存储型业务场景需求。     |


### 2.1-集合通信中的存储

存储是集合通信执行所需要的各种Buffer资源，集合通信执行中涉及到以下几种Buffer： 
- **Input Buffer**：集合通信算子输入数据缓冲区。 
- **Output Buffer**：集合通信算子输出数据缓冲区。 
- **CCL Buffer**：<font color='red'><b>一组地址固定的Buffer</b></font>，单算子模式下，通信实体通过CCL Buffer来实现Rank间的数据交换。 
	- CCL Bufferi和通信域绑定，通信域初始化的时候创建两块CCL Buffer，分别称为CCL In和CCL Out。 CCL_Ini和CCL Out默认大小是200M Byte，可以通过环境变量HCCL_BUFFSIZE进行修改。同一个通信域内执行的集合通信算子都复用相同的CCL Buffer。
	
![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428155950.png)

- **Scratch Buffer**：有些算法计算过程中需要额外的存储空间，这部分额外的存储空间，称为Scratch Buffer。。


### 2.2-流

流(Stream)是NPU上的一种硬件资源，承载了待执行的Task序列。Task可以是一个DMA操作、一个同步操作或者一个NPU算子等。<font color='red'><b>同一个流上的Task序列按顺序执行，不同流上的Task序列可并发执行</b></font>。 

由Al框架（例如PyTorch等）调用集合通信API时传入的Stream对象称为主流，为实现集合通信的并行性而额外申请的Stream对象称为从流。主从流之间通过Post/Waiti这一组Task进行同步。主从流之间没有依赖关系时，Task可并行执行，如下图中主流的TaskA、TaskB和从流的TaskX、TaskY可并行执行。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428160131.png)

```cpp
HcclResult HcclAllReduce(void *sendBuf,void *recvBuf,uint64_t count,HcclDataType dataType, HcclReduceOp op,HcclComm comm,aclrtStream stream)
```

### 2.3-Notify

Notify是NPU上的硬件资源，用来做同步。在集合通信中主要有两种作用： 
1. 进行主从流之间的同步进行Rank间数据收发的同步。
	- ![image.png|left|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428160446.png)
	- 和Notify有关的Task有两种：**Post**和**Wait**(即流/主从流中的Post/Waiti这组Task)
		- Post是非阻塞操作， 作用是给对应的Notify寄存器置1，如果对应的Notify值已经是1，则不产生变化；
		- Wait是阻塞操作，会等待对应的Notify值变为1。当预先设置的条件满足后，会将对应的Notify值复位为0，并继续执行后续的Task。
2. Rank间的数据收发也需要同步，比如向远端Rank写数据前，得知道远端是否准备好接受数据的 Buffer。

### 2.4-Transport链路

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428162439.png)

要完成Rank间的数据通信需要先建立transport链路，transport链路分两种：
- **SDMA transport链路**（对应到 HCCS/PCIE硬件连接)和RDMA链路（对应到RoCE硬件连接）。 SDMA transport链路的两端各有2种类型的Notify，分别称为Ack、DataSignal。
- **RDMA transport链路**的两端各有3种类型的notify，分别称为Ack、DataSiganl、DataAck。 每条transport链路会申请各自的Notify资源，不同的transport之间不会复用Notify。所以SDMA链路会申请4个Notify，每端各2个；RDMA链路会申请6个Notify，每端各有3个。

#### SDMA收发数据

一次SDMA数据收发需要两组同步，如下图所示，分别使用了Ack和DataSignal两个Notify。为了避免同一条链路上多次收发数据相互影响，同步需以Ack开始，以DataSignal结束。

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428162627.png)

#### RDMA 收发数据

一次RDMA数据收发需要三组同步信号，如下图所示。这是因为**RDMA操作在流上是异步执行**的，所以Rank 0执行完Write和Post DataSignal之后，并不知道数据什么时候写完，因此需要Rank1 Wait DataSignali满足条件后，再给Rank0发送一个DataAck同步信号，通知Rank0数据已经写完了。 为了避免同一条链路上多次收发数据相互影响，RDMA数据同步需以Ack开始，以DataAck结束。

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428163108.png)

### 2.5-通信域/子通信域/算法分层

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428163333.png)

### 2.6-通信算子(原语/框架)与算法

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428163548.png)

**通信原语（框架）**：定义的是从怎样的初始状态达成怎样的终止状态，而没有规定如何达成
- 典型原语：AllGather、AllReduce....
**通信算法**：定义的是从如何从初始状态达成终止状态
- 典型算法：Mesh、Ring、RND、NHR

## 3-HCCL的软硬件架构

### 3.1-HCCL的软件架构

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428163653.png)

集合通信库软件架构分为三层：适配层，图引擎与单算子适配，进行通信切分寻优等操作。 
- 集合通信业务层，包含通信框架与通信算法两个模块： 
	- **通信框架**：负责通信域管理，通信算子的业务串联，协同通信算法模块完成算法选择，协同通信平台模块完成资源申请并实现集合通信任务的下发。
	- **通信算法**：作为集合通信算法的承载模块，提供特定集合通信操作的资源计算，并根据通信域信息完成通信任务编排。 
- 集合通信平台层，提供NPU之上与集合通信关联的资源管理，并提供集合通信的相关维护、测试能力。

### 3.2-HCCL的硬件架构

#### 8P架构

集合通信库需要承载在特定的硬件之上，这里说的硬件，主要是多个昇腾AI处理器（即多个PU，又称多张卡） 组成的硬件形态。实际实现中，**一般一个服务器组成一个AI节点**，如果需要使用更大规模的昇腾组网，通常使用**以太交换机将多个AI节点互联为一个AI集群**。 AI节点硬件是由NPU、CPU、内存、硬盘等组成的硬件实体，本节主要体现集合通信相关的硬件TOPO结构。
![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428164002.png)

该硬件形态中，8P昇腾NPU之间通过HCCS总线完成两两互联，昇腾NPU与CPU之间通过PCIe总线互联。

#### 16P架构

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428164057.png)

该硬件形态中，NPU分为两个8P Fullmesh组，每个8P Fullmesh互连组内，8P昇腾NPU之间通过HCCS总线完成两两互联(Fullmesh)，两个8P Fullmesh：组之间通过PCIe-SW完成互连。CPU与NPU之间也是通过PCIe-SW互联。


## 4-HCCL常见的集合通信原语（算子/框架）

集合通信(Collective Communications)是一个进程组的所有进程都参与的全局通信操作，其最为基础的操作有发送send、接收receive、复制copy、组内进程栅障同步Barrierl以及节点间进程同步(signal+wait)，**这几个最基本的操作经过组合构成了一组<font color='red'><b>通信模板</b></font>也叫<font color='red'><b>通信原语</b></font>**，比如：1对多的广播broadcast、 多对1的收集gather、多对多的收集all-gather、1对多的发散scatter、多对1的规约reduce、多对多的规约all-reduce、组合的规约与发散reduce-scatter、多对多的all-to-all等，集合通信的难点在于通信效率以及网络硬件连接拓扑结构的最佳适用。


![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427160011.png)



![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427160119.png)


## 5-集合通信常见算法

集合通信的难点在于需要在<font color='red'><b>固定的网络互联结构的约束</b></font>下进行高效的<font color='red'><b>通信</b></font>，集合通信拓扑算法与物理网络互联结构强相关，为了发挥网络通信的效率，也不是说就能随意发挥通信拓扑算法，更多的是在效率与成本、 带宽与时延、客户要求与质量、创新与产品化等之间进行合理取舍。


### 5.1-典型的网络拓扑

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428165015.png)

### 5.2-集合通信算法

- **网络拓扑**：多机/多卡是怎么连接到一起的
- **通信原语**：数据当前的**起点**状态和**最终**想达到的状态
- **通信算法**：如何基于确定的**网络拓扑**达到**通信原语**想达到的效果

#### 性能评估理论
HCCL采用α-β模型（Hockney）进行性能评估，算法耗时计算用到的变量定义如下：
- **a**：节点间的固定时延。
- **β**：每byte数据传输耗时。
- **n**：节点间通信的数据大小，单位为byte。
- **γ**：每byte数据规约计算耗时。
- **p**：通信域节点个数，影响通信步数。
其中单步传输并规约计算n byte数据的耗时为：
$$
D = a + nβ + nγ
$$


#### Mesh算法（Server内的算法）

Mesh是FullMesh互联拓扑内的基础算法，是NPU之间的全连接，<font color='red'><b>任意两个NPU之间可以直接进行数据收发</b></font>。
![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428165631.png)
Mesh算法实现`AllReduce`算子的流程如下图所示，每个NPU并发的使用多路 HCCS链路从对端读取或者写入数据，使**双工互联链路的双向带宽同时得到利用**。 Mesh算法的时间复杂度是O(1)。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428165727.png)

#### Ring算法（server内和server间算法）

Ring算法，所有的NPU以环形相连，每张卡都有左手卡与右手卡，一个负责数据接收，一个负责数据发送，<font color='red'><b>循环完成梯度累加，再循环做参数同步</b></font>。 

适用于小规模节点数(小于32机，且非2幂)和中大规模通信数据量(大于等于256M)的场景。

![image.png|center|200](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428165902.png)

Ring算法适用于“星型”或“胖树”拓扑互联，其特点是通过**Ring环**将所有 NPU设备的单端口双工链路串联起来。 Ring算法实现AllReduce算子的流程如下图所示，每一步依次给下游发送对应的数据块，沿着环转一圈之后完成ReduceScatter阶段，再沿环转一圈完成 AllGather阶段。 Ring算法的时间复杂度是O(n-1)，n为Ring环上的NPU设备个数。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428170004.png)

#### RHD算法（server间/节点间）

**递归二分和倍增算法**，当通信域内Server个数为2的整数次幂时，此算法具有较好的亲和性。

当组网增大时，例如增大至4K个rank的场景，Mesh很难组成4K个rank的全连接网络（全连接一个时钟周期就可以完成操作），且资源开销 (链路资源，交换资源，同步资源)太大，还可能存在算力和资源开销不匹配的问题。Ring在这种情况下虽然节省资源（只用左手卡和右手卡进行一次收发)，但是环内要做太多次，流转太慢。大型规模集群运算有服务器内数据量庞大、Ring环极长的特点，Ring的这种切分数据块的方式就不再占优势。

RHD(Recursive Halving-Doubling)算法通过<font color='red'><b>递归加倍及递归折半方式</b></font>完成NPU间的数据交换，相对Mesh资源消耗较小，相对Ring效率会更高。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428170308.png)

RHD算法的实现流程如图所示，假设有$5\times(2^2+1)$ 个rank，首先将rank1的数据合并到rank0，变成$4\times(2^2)$个rank，然后将这4个rank的数据两两对半交换数据并求和，即ReduceScatter操作。下一阶段，将这4个rank的数据两两拼接，即AllGather操作。最后 ，将rank0的数据复制到rank1，至此每个rank都具有所有rank的全量数据之和。 

RHD算法同样适用于“星型”或“胖树”拓扑互联 算法的时间复杂度是$O(log_2N)$。


#### PairWise算法（Server间）

比较算法，仅用于AllToAlL与AlltoAllV算子，适用于数据量较小( $<=1M * RankSize$) 的场景。 通常每个节点只有一个RDMA网口，如果在RDMA链路上使用Mesh算法完成AllToAll，存在同时从多个节点接收数据、向多个节点发送数据的 “多打多”问题，多个流在同一条链路上肆意争抢资源，可能反而导致整体性能下降。 

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428170600.png)

PairWise算法是**Mesh算法的分步执行版本**，通过合理的规划，将通信分解成多个步骤，每一步只从一个节点接收数据、向一个节点发送数据。 比如对于rankid为i的节点，第一步从（i-1）节点接收数据，向(i+1)节点发送数据；第二步从(i-2)节点接收数据，向(i+2)节点发送数据。以此类推。


#### Star 算法（Server内）

Star算法适用于有根节点的通信操作（如Broadcast、Reduce、Gather、Scatter等），利用星型拓扑或全连接拓扑一步完成通信操作。以 Broadcast算子为例，Star算法实现如下图所示，根节点Root利用星型拓扑从其他各节点收集数据。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428170749.png)

## 6-集合通信业务流

### 6.1-概述

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428170920.png)


1. 首先进行集群信息配置，创建通信域句柄，并初始化HCCL通信域。 
2. 实现集合通信操作，例如点对点通信、调用集合通信算子。 
3. 集合通信操作完成后，需要释放相关资源，销毁通信域。

### 6.2-通信域初始化

#### 单机多卡（多线程管理）

📒：启动一个进程进行管理多个卡，其中，每张卡对应一个线程来进行管理！！！！！！！！！

单机通信场景中，通过<font color='red'><b>一个进程统一创建多张卡的通信域（其中一张卡对应一个线程）</b></font>。在初始化通信域的过程中，**devices[0]** 作为root_rank自动收集集群信息。

函数原型：
```cpp
HcclResult HcclCommInitAll(uint32 t ndev,int32 t*devices,HcclComm* comms)
```
参数说明：
- ndev（输入）：表示通信域中**设备（device）** 的数量，是一个整数，输入给通信库用于后续管理。
- devices（输入）：是一个**设备列表**，其中的每个元素是一个**逻辑设备ID**。通信库会根据这个顺序来创建通信域。特别要注意，devices数组中的ID值必须互不重复，不能包含重复的设备编号。
- comms（输出）：是**生成的通信域句柄数组**，每个句柄表示一个device对应的通信域实例。其数组大小应为 ndev × sizeof(HcclComm)。


#### 多机多卡 （多进程+多线程）

##### 有集群信息（ranktable）

如果手上有集群描述信息(ranktable file，下文简称ranktable)，则<font color='red'><b>每张卡都要启动一个单独的进程基于ranktablei进行通信域的创建</b></font>。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428171633.png)
**函数原型**
```cpp
HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)
```

**参数说明**

| **参数名**     | **输入/输出** | **描述**                                       |
| ----------- | --------- | -------------------------------------------- |
| clusterInfo | 输入        | rank table的文件路径（含文件名），作为字符串最大长度为4096字节，含结束符。 |
| rank        | 输入        | 本rank的rank id。                               |
| comm        | 输出        | 将初始化后的通信域以指针的信息回传给调用者。                       |

##### 无集群信息

多机集合通信场景，若无完整的集群信息配置ranktable文件，HCCL提供了基于root节点广播的方式创建通信域，详细流程如下： 
1. 配置环境变量`HCCL_IF_IP`，指定root通信网卡的IP地址。指定的网卡需为Host网卡，仅支持配置一个IP地址，要求是IPv4或IPv6格式，配置示例如下：`export HCCL_IF_IP=10.10.10.1
2. 在root节点调用`HcclGetRootInfo`接口，生成root节点rank标识信息rootInfo”，包括device ip、device id等信息。
3. 将root节点的rank信息广播至集群中的所有rank。
4. 在所有节点调用`HcclCommInitRootInfo`或者`HcclCommInitRootInfoConfig`接口，基于接收到的 “rootlnfo”，以及本rank的rank ids等信息，进行HCCL初始化。


需要注意：每个卡使用一个单独的进程进行如下操作。 使用HcclCommInitRootlnfo接口创建通信域的简单代码示例片段如下：

```cpp
#include <hccl/hccl.h>
#include <mpi.h>
using namespace std;

// 在root节点获取其rank信息
HcclRootInfo rootInfo;
int32_t rootRank = 0;
if (devId == rootRank) {
    HcclGetRootInfo(&rootInfo);
}

// 将root_info利用MPI通信机制广播到通信域内的其他rank
MPI_Bcast(&rootInfo, HCCL_ROOT_INFO_BYTES, MPI_CHAR, rootRank, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);

// 定义通信域句柄
HcclComm hcclComm;

// 初始化HCCL通信域
HcclCommInitRootInfo(devCount, &rootInfo, devId, &hcclComm);

/* 集合通信操作 */

// 销毁HCCL通信域
HcclCommDestroy(hcclComm);
```

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250428172446.png)


### 6.3-点对点通信Demo

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/devguide/hccl/hcclug/hcclug_000011.html

### 6.4-集合通信Demo


https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/devguide/hccl/hcclug/hcclug_000010.html

















