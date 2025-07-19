---
sidebar_position: 0
---

# 说明

## 论文概况

|标题|主要内容|
| - | - |
|A Communication-Avoiding 3D LU Factorization Algorithm for Sparse Matrices|借鉴 2.5D dense LU 算法的思想，将二维进程 grid 映射到三维 block 上面，本方法在进行映射时，借助 eliminate-tree(e-tree)的结构进行映射，将各个子树上所需要的父节点数据进行拷贝到每一个层上面，并行计算每个层的 schur-complement，最后将计算好的所有子树进行通信，合并到根节点上，也就是说，利用空间换时间的方式，实现通信量的减少，因此每层子树所需要的数据都已经拷贝一份了。|
|A communication-avoiding 3D sparse triangular solver|团队实现该求解器的思想和实现三维 LU 分解的思想基本一致，都是利用etree 中所涉及的依赖关系，然后对每个式子中可以独立进行计算的部分进行细分，经过拷贝共同需要的数据后，实现各个部分的并行计算，从而减少通信和计算时间。即，也是通过空间换取时间的措施，来实现直接求解器的加速操作。|
|Making Sparse Gaussian Elimination Scalable by Static Pivoting|通过各种技术来保持数值稳定性：预先将大元素转移到对角线上、迭代细化以及允许在最后进行迭代的修改等。文章还实现了基于 static pivoting 的 SuperLU_dist算法以及三角求解器算法。|
|Pangu LU A Scalable Regular Two-Dimensional Block-Cyclic Sparse Direct Solver on Distributed Heterogeneous Systems|提出一种新的直接求解器用于分布式异构系统的 $Ax = b$ 的求解——PanguLU，它使用规则的二维区块进行布局，并以块内稀疏子矩阵为基本单元，构建了一种新的分布式稀疏 LU 分解算法。由于存储的矩阵块是稀疏的，因此利用稀疏 BLAS 进行计算，进而避免不必要的填充，并优化稀疏属性，使得计算更加高效。|