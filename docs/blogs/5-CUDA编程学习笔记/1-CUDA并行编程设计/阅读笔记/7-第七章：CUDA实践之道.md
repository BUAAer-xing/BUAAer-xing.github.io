## 7.1-简介

在进行CUDA编程时，应该注意的一些方面。

## 7.2-串行编码和并行编码

### 7.2.1-CPU与GPU的设计目标

CPU与GPU的主要区别是缓存等级和数量的不同。过去，指令周期很长，但随着芯片密度的不断增长，指令周期缩短了。现在，内存访问成了现代处理器设计的主要瓶颈，目前一般通过**多级缓存**应对解决。然而，GPU选择了另一种设计方式，以费米架构的GPU为例，他们在每个SM上 都设计了一个共享内存。共享内存类似于传统处理器上的一级缓存，是一小块低延时、高带宽的内存。

费米架构的GPU每次内存访问会获取128字节的数据，并将这128字节的数据放入到一个缓存行中（Cache Line），随后相邻线程的访问将会命中缓存，即在缓存中找到需要的数据，避免了再次访问全局内存。

GPU设计与CPU设计的一个显著不同就是**SIMT执行模型**。在MMD模型中，每个线程都有独立的硬件来操作整个独立的指令流。如果线程执行相同的指令流但不同的数据，那么这种方法非常浪费硬件资源。而GPU直接提供了<font color='red'><b>一组硬件</b></font>来运行这N个线程，N正好为当前线程束的大小32（也就是说，GPU上在同一个warp中的线程并不是独立的，而是需要<font color='red'><b>成组</b></font>进行执行的）。SIMT模型解决了一个关键的问题，就是<font color='red'><b>程序员不必再为每个执行相同路径的线程写代码</b></font>。线程可以产生分支然后在之后的某一点汇集。但由于只有一组硬件设备执行不同的程序路径，因此程序的灵活性有所下降，**不同程序路径在控制流汇集之前顺序或轮流执行**。设计内核函数时，需要考虑到这一点。

GPU与CPU还有一个区别是：GPU采用懒惰计算模型进行计算任务的计算，而CPU采用热情计算模型进行计算任务的计算，这两种模型的主要区别在于任务调度和执行的时机上。
1. **GPU的懒惰计算模型**：
   懒惰计算模型指的是**在计算过程中，系统并不会立即执行计算任务，而是先将操作记录下来，直到遇到真正需要结果的时刻（如输出、同步等）才执行计算**。GPU由于其庞大的并行计算能力，经常使用这种懒惰计算策略来优化性能。这样可以避免一些不必要的计算，并允许系统在执行计算前对任务进行进一步的优化或合并。
   - 优势：减少不必要的中间计算开销，提升整体计算效率；有利于调度多个任务，充分利用硬件资源。
   - 典型应用：TensorFlow、PyTorch等深度学习框架在使用GPU进行计算时，往往采用懒惰计算模型，以最大化计算图优化的潜力。
2. **CPU的热情计算模型**：
   热情计算模型指的是**当遇到计算任务时，系统立即执行计算操作并生成结果**。这种方式的优点是简单直接，适合需要快速响应或无需复杂优化的计算任务。CPU因为其较强的单线程性能和顺序执行的优势，往往采用这种策略。它更适合处理复杂的控制逻辑和依赖关系明确的任务。
   - 优势：响应快，适合低延迟任务；逻辑上更容易理解和调试。
   - 典型应用：大多数CPU执行的常规程序，包括大部分操作系统中的任务调度和应用程序的执行。
这两种模型适用于不同的计算场景。GPU由于具备大规模并行计算能力和需要大量任务调度优化，因此偏向于懒惰计算，而CPU则由于单线程性能优势和复杂指令集的特性，更适合热情计算。

### 7.2.2 CPU与GPU上的最佳算法对比

在使用GPU解决一个问题时，需要考虑使用的线程数量是有限制的（换句话说，就是**一个线程块中启用的线程数量是有限制的**，比如在开普勒架构上，一个线程块中最多开启1024个线程。）。这个限制是由于受到寄存器使用的限制，一般合理且复杂的内核函数会将线程数限制为256或512。线程间通信的问题是分解问题的关键。线程间的通信只能通过重新调度单独的内核和使用全局内存来实现（注：线程束内的线程，可以通过硬件语言来进行通信）。但使用全局内存速度会慢一个数量级。


不要认为所有的计算都必须在GPU上执行。CPU也可以承担一些计算任务。比如先在CPU上进行一些初始化计算，当需要大规模并行时再将数据复制到GPU进行计算。选择一个适合硬件的算法并以正确的布局方式获取数据时提高GPU运行性能的关键。

## 7.3 数据集的处理

处理大规模数据，比较好的方法是，将其分成若干个小部分，然后再完成相应的操作之后再将它们合并。事实上，大多数并行处理算法都以不同的方式采用这个方法来避免更新公共数据结构带来的序列化瓶颈。


## 7.4 性能分析

Parallel Nsight 软件以及 软件。

## 7.5 总结

在开始书写代码之初，应该全盘考虑和理解寄存器、共享内存、缓存的用法以及全局内存的访问模式等设计的主要方面。











