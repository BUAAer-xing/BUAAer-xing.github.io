---
sidebar_position: 0
title: 概述
id: my_papers_intro
slug: /my_papers_intro
---

| 标题 | 摘要 |
|-|-|
|【ICS‘25】CB-SpMV:A Data Aggregating and Balance Algorithm for Cache-Friendly Block-Based SpMV on GPUs|稀疏矩阵-向量乘法（SpMV）在计算科学、工程和机器学习中至关重要。尽管通过各种技术在GPU上提高SpMV性能的努力已经相当可观，但与数据局部性、硬件利用和负载均衡相关的问题仍然存在，为进一步优化留出了空间。本文提出了一种基于缓存友好的SpMV优化算法CB-SpMV，采用了一种新颖的数据收敛和适应性的二维块结构。CB-SpMV中的矩阵被划分为独立的子块，虚拟指针聚合不同类型的块内数据，以提高缓存级的数据局部性。为了增强硬件利用率，提出了一种块意识的列聚合策略和子块格式选择，以加速计算并适应不同的稀疏矩阵。最后，设计了一种块间负载均衡算法，以确保线程块之间的高效工作负载分配。在2,843个来自SuiteSparse集合的矩阵上的实验评估表明，CB-SpMV显著提高了缓存命中率，并在NVIDIA A100和RTX 4090 GPU上实现了与最先进的方法如cuSPARSE-BSR、TileSpMV和DASP相比，平均加速比高达3.95倍。|
|||