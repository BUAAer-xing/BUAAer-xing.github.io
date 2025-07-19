## 0-概述

DeepEP: an efficient expert-parallel communication library **专家并行通信库**

DeepEP 是 DeepSeek 团队为**混合专家模型**（MoE, Mixture of Experts）和**专家并行**（EP, Expert Parallelism）场景设计的高性能通信库，<font color='red'><b>旨在解决大规模分布式训练和推理中的通信瓶颈问题</b></font>。其优化措施主要体现在以下几个方面：
- 首先，DeepEP 提供了高效的 **All-to-All 通信内核**（又可以进一步拆分为<font color='red'><b>dispatch内核</b></font>与<font color='red'><b>combine内核</b></font>），支持节点内的 NVLink 和节点间的 RDMA 通信，显著提升了数据传输效率。在实际测试中，单节点 NVLink 带宽利用率超过 95%，跨节点 RDMA 延迟仅为 163 微秒，极大地减少了通信延迟 。
- 其次，DeepEP 支持低精度数据类型，如 FP8 和 BF16，降低了通信数据量，进一步提升了通信效率。此外，库中引入了<font color='red'><b>基于 Hook 的通信与计算重叠机制</b></font>，允许在不占用 GPU 流式多处理器（SM）资源的情况下，实现通信和计算的并行执行，提高了整体计算资源的利用率 。
- 在内核优化方面，DeepEP 针对非对称带宽转发场景（如 NVLink 到 RDMA）进行了深度优化，确保在不同通信路径下都能达到高性能。同时，库中还使用了未公开的 PTX 指令 `ld.global.nc.L1::no_allocate.L2::256B`，**通过绕过 L1 缓存并直接访问 L2 缓存，以 256 字节的事务大小加载数据，进一步提升了内存访问效率** 。
- 在实际应用中，DeepEP 显著提升了 MoE 模型的训练和推理效率。例如，在 H800 GPU 上，使用 DeepEP 的常规内核进行训练时，吞吐量达到了 153 GB/s，接近 NVLink 的理论带宽上限；而在推理阶段，使用纯 RDMA 的低延迟内核，端到端延迟降至微秒级，带宽达到 46 GB/s，接近 RDMA 的理论极限 。

DeepEP 通过高效的通信内核、低精度支持、通信与计算重叠机制以及深度的内核优化，极大地提升了 MoE 模型在大规模分布式环境下的训练和推理效率，为实现高性能的专家并行计算提供了有力支持。

## 1-ALL-to-ALL通信算子

![[2-MoE中的All-to-All算子]]


## 2-DeepEP























