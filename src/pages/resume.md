---
title: Resume for CX
hide_table_of_contents: true
---

<center> 
<img src="/img/buaa.png"/>
</center>

---
<h1>
<span>丛兴👨‍💻</span>  &emsp; &emsp;<span>|🏠山东济宁|✨中共党员|📕研究生|❤️汉族|</span>
</h1>
---

### 🧑‍🎓教育经历
<strong><font face = "Microsoft YaHei" size = "4" color = "#183884" >东北大学 · 本科</font> </strong> &emsp;&emsp;&emsp;&emsp;&emsp;&thinsp;
<strong><font face = "Microsoft YaHei" size = "4" color = "#183884" >北京航空航天大学 · 硕士研究生</font> </strong> &emsp;&emsp;&emsp;&emsp;&emsp;&thinsp; <strong><font face = "Microsoft YaHei" size = "4" color = "#183884" >北京航空航天大学 · 博士研究生</font> </strong> &emsp;&emsp;&emsp;&emsp;&emsp;&thinsp;<font  size = "3" color = "#183884" ><strong>专业：</strong>计算机科学与技术</font>

**邮箱**：congxingcs@163.com  &emsp; &emsp;&thinsp; **电话**：17562036369&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**状态**：在读博士、党支部书记、TACO(CCF-A)审稿人

---

### 📄 发表论文

- **CB-SpMV:A Data Aggregating and Balance Algorithm for Cache-Friendly Block-Based SpMV on GPUs (第一作者, ICS’25, CCF-B)** [**[Paper]**](/docs/my_papers/ICS25_CB_SpMV/CB_SpMV_Paper) [**[Slide]**](/docs/my_papers/ICS25_CB_SpMV/CB_SpMV_Slide)

- **CB-Sparse:A Cache-Friendly Data Aggregating Algorithm for Block-Based Sparse Matrix Multiplication on GPUs (第一作者, 已投稿, TACO, CCF-A)**


---
### 💼实习经历

<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>华为技术有限公司📱</span><span>AI研发实习生</span><span>2025.04-至今</span>
</h4>

- **项目介绍**：负责NPU-CANN架构中的通信优化与建模工作
  - 针对CANN框架中HCCL通信算子，深入分析其执行过程，识别通信中的barrier点以及冗余代码逻辑，优化算子实现以减少通信阻塞，提升整体执行性能；
  - 对NPU的计算模块、内存管理单元以及通信路径进行微架构级别的仿真建模，构建NPU端Cache子系统模型，进一步用于性能预测和优化验证；


<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>抖音视界有限公司🎵</span><span>AML研发实习生</span><span>2024.06-2024.09</span>
</h4>

- **项目介绍**：利用Kokkos对lammps中的GPU代码进行优化和重构
  - 成功将AML模型中的预测计算逻辑重构并迁移至lammps主计算流程，消除预测逻辑与MD模拟之间的冗余通信，显著降低预测开销；
  - 借助NVIDIA Nsight分析工具，深入分析集成后系统的计算和数据传输瓶颈，提出面向GPU共享资源调度的具体优化策略，辅助Kokkos进行异构计算层级调度调整；


<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>百度在线网络技术(北京)有限公司🐻</span><span>云原生研发实习生</span><span>2024.03-2024.06</span>
</h4>


- **项目介绍**：负责封装集群调度器操作的插件以及相关云原生应用环境的部署和测试
  - 本项目架构分为三层，分别涉及的语言为：Java、Go和Python，我在工作中，需要打通三层代码逻辑，将Slurm等调度器的相关操作封装成Python插件。效果为，用户在前端进行操作，操作逻辑通过三层结构，最终应用到集群上；
  - 在云原生异构计算环境中部署和测试相关应用环境，比如机器人模拟领域的MuJoCo环境、汽车模拟领域的starccm+、abaqus以及optistruct等; 

---

### 🚀项目经历

<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>****千万核级并行算法以及深度优化技术</span><span>国家项目</span><span>2023.10-至今</span>
</h4>

- **项目介绍**：负责设计异构计算Kernel以及矩阵存储结构
  - 设计适合异构计算的数据结构，实现高效的矩阵存储结构，提高节点内计算效率;
  - 实现任务的自适应分配以及节点间的负载均衡和计算加速;


<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>基于机器学习的SpMV操作任务粒度自动选择模型</span><span>机器学习+研究方向实践</span><span>2023.11 – 2024.01</span>
</h4>

- **项目介绍**：
  - 利用MPI编程模型，获得每个矩阵执行SpMV最佳的任务粒度; 
  - 结合机器学习中常见的算法，利用从稀疏矩阵中提取出的多个属性，对多种模型进行训练，获得最佳模型;


<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>基于Vue的数据结构与算法可视化平台</span><span>本科毕业设计</span><span>2022.10 – 2023.04</span>
</h4>

- **项目介绍**：
  - 该平台由我个人独立进行开发。平台基于目前最先进的Vue3前端框架并结合相关组件库进行开发，具有快速、高效、易用和可维护性的优点。
  - 平台实现常见数据结构：线性表、树、图、队列、栈等的可视化。平台也实现了常见算法：排序、查找、遍历等算法的可视化。同时引入基于ChatGPT的人工智能助手，更快的帮助使用该平台的学生解决学习过程中的疑惑。


<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>MTCNN + Mobile Net 的口罩识别项目工程训练😷</span><span>二次开发负责人</span><span>2022.09-2022.10</span>
</h4>

- **项目介绍**：
  随着国内外疫情形势的不断发展，直到今天，疫情防控已经成为了国民日常生活中的一部分。在这种常态下，基于各种各样的需求，产生了相对应的产品，人脸识别技术也在此背景下得到了相对应的发展。所以，本次工程实训选择复现MTCNN+Mobile Net的口罩识别项目，并且实现了动态进行多张人脸检测的结果实时进行输出的效果。

<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>对于十四五规划下的农村新能源结构的调查与前景展望🛖</span><span>主要成员</span><span>2021.12-2022.03</span>
</h4>

- **项目介绍**：
  本课题为参加全国大学生节能减排科技大赛所提交的作品议题，本项目由我提出想法，并组织团队进行调查，组织队员从网上收集资料并做数据处理，最终完成作品书。本作品获得校级一等奖和省级二等奖的成绩，并且入选国家级赛道。


<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>基于OpenCV实现人体姿势识别实现人机交互地铁跑酷游戏</span><span>二次开发</span><span>2022.01-2022.02</span>
</h4>

- **项目介绍**：
  为了实现计算机图形学的相关学习，通过网上自学Opencv相关知识，通过python编程，利用开源库，借助别人训练好的模型，对人体的头部、肘部、腿部，脚部进行实时动态识别，实现实时实现识别人体动作，从而反映到地铁跑酷的游戏当中，实现人机交互游戏，但实现起来，延迟较为严重，可玩性不佳。

<h4 style={{ display: 'flex', justifyContent: 'space-between' }}>
<span>课程设计 —— 📝大学课程设计汇总</span><span>后端负责人</span><span>2019.09 - 至今</span>
</h4>

- 利用 DELPHI 实现**TFTP 协议**（课程设计优秀）
- 基于**JSP**的NEUQ宿舍管理系统设计与实现（课程设计优秀）
- 基于 8086 的计时抢答器设计（结课设计96分）
- 基于 **Docker 技术的** LAMP框架搭建网站实现与分析（结课论文97分）	
- 基于**QT多线程**机制实现模拟操作系统的基本功能  （课程设计优秀）   
- 基于 Verilog 实现五级流水CPU设计 （课程设计优秀）	
- 电子线路综合实现加法器 （课程设计优秀）	
- 基于C++实现公司员工考勤系统 （课程设计优秀）	

------

### 😊个人能力

- **编程语言**：主要使用：C++，CUDA，了解并基本会使用Python、Triton、C#、HTML、CSS、JavaScript等语言来完成相应的任务要求
- **开发工具**：了解QT、IDEA、Verilog、Pycharm、Delphi 7等集成开发环境的基本使用方法来进行课程设计或者项目的开发，知道Navicat、Powerdesign等数据库管理工具的基本使用方法，了解Cisco Packet Tracer和wireshark等网络模拟和监听软件的基本使用，知道Enterprise Architect、Visio、starUML等软件建模工具的基本使用方法，会使用Proteus 8来进行电子信息电路的设计以及硬件接口电路的设计。
- **管理工具**：会基本的Git语法来进行版本的控制
- **办公软件**：熟练的使用Word来进行报告的攥写、使用PowerPoint来进行PPT汇报展示，基本了解Excel的处理方法
- **编辑软件**：会熟练的使用Markdown语法进行文档的攥写，同时基本会使用Latex语言编写文档
- **语言能力**：英语四级（433）、英语六级（485）、普通话（二级乙等）
- **职业证书**：高中数学教师资格证书

----

### 🏆荣誉奖项

- **全国大学生节能减排设计大赛**校级一等奖、省级二等奖
- 第13届**蓝桥杯(软件类)** 省级三等奖	
- 图灵杯编程能力大赛校级二等奖两次	
- **美国大学生数学建模大赛**校级二等奖、国家级三等奖 	
- 第11届全国大学生电子商务“创新、创意及创业”挑战赛校级三等奖	
- ACM-HCCPC**河北省大学生程序设计大赛**省级二等奖	
- 第二届全国大学生算法设计与编程挑战赛（冬季赛网络赛）银奖
- 获得校级奖学金3次（一次二等奖学金，两次三等奖学金）、**励志奖学金2次**
- 被评为**校级优秀学生干部标兵**1次、校级优秀学生干部1次 
- 被评为优秀团员2次，**优秀团员标兵**1次 
- 被评为暑期**社会实践校级先进个人**

----

### 👨‍💼实践经历

- 北航计算机学院BY应用一 **党支部书记**

- 北航计算机学院SY23063班 **团支书** 

- 东大计算机与通信工程学院 **社会实践中心主任**

- **班级组织委员**

- **ACM俱乐部成员**  	


<script type="text/javascript" src="https://rf.revolvermaps.com/0/0/8.js?i=5ct701dzzey&amp;m=0&amp;c=ff0000&amp;cr1=ffffff&amp;f=arial&amp;l=33" async="async"></script>