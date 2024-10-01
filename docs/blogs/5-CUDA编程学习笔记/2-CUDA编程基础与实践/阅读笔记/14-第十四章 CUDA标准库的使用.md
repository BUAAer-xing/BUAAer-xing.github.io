## 14.1 CUDA标准库简介

<table border="1" cellpadding="10" width="100%">
  <tr>
    <th>库名</th>
    <th>简介</th>
  </tr>
  <tr>
    <td>Thrust</td>
    <td>类似于 C++ 的标准模板库 (standard template library)</td>
  </tr>
  <tr>
    <td>cuBLAS</td>
    <td>基本线性代数子程序 (basic linear algebra subroutines)</td>
  </tr>
  <tr>
    <td>cuFFT</td>
    <td>快速傅里叶变换 (fast Fourier transforms)</td>
  </tr>
  <tr>
    <td>cuSPARSE</td>
    <td>稀疏 (sparse) 矩阵库</td>
  </tr>
  <tr>
    <td>cuRAND</td>
    <td>随机数生成器 (random number generator)</td>
  </tr>
  <tr>
    <td>cuSolver</td>
    <td>稠密 (dense) 矩阵和稀疏矩阵计算库</td>
  </tr>
  <tr>
    <td>cuDNN</td>
    <td>深度神经网络 (deep neural networks)</td>
  </tr>
</table>

学习和使用库的优点：
- 可以节约程序开发时间。
- 可以获得更加值得信赖的程序。
- 可以简化代码。
- 可以加速程序。对于常见的计算来说，库函数能够获得的性能往往是比较高的。但是，**对于某些特定的问题，使用库函数得到的性能不一定能胜过自己的实现**。
	- 例如，Thrust和cuBLAS库中的很多功能是很容易实现的。有时，一个计算任务通过编写一个核函数就能完成，<font color='red'><b>用这些库却可能要调用几个函数，从而增加全局内存的访问量</b></font>。此时，用这些库就有可能得到比较差的性能。

## 14.2 Thrust 库

略

## 14.3 cuBLAS库

cuBLAS是BLAS在CUDA运行时的实现。BLAS的全称是basic linear algebra subroutines(或者basic linear algebra subprograms)，即基本线性代数子程序。

<font color='red'><b>cuBLAS API</b></font>，该API实现了BLAS的3个层级的函数，一共几十个：
- (1)第一层级的（十几个）函数处理矢量之间的运算，如矢量之间的内积。 
- (2)第二层级的（二十几个）函数处理矩阵和矢量之间的运算，如矩阵与矢量相乘。 
- (3)第三层级的（十几个）函数处理矩阵之间的运算，如矩阵与矩阵相乘。


## 14.4 cuSolver库

cuSolver专注于一些比较高级的线性代数方面的计算，如矩阵求逆和矩阵对角化。正因为比较高级，所以cuSolver库的实现是基于cuBLAS和cuSPARSE 两个更基础的库的。cuSolver的功能类似于Fortran中的LAPACK库。LAPACK 是Linear Algebra PACKage的简称，即一个线性代数库。

cuSolver库由以下3个相互独立的子库组成：
- (1)cuSolverDN(DN是DeNse的意思)：一个处理稠密矩阵线性代数计算的库。
- (2)cuSolverSP(SP是SParse的意思)：一个处理稀疏矩阵的线性代数计算的库。
- (3)cuSolverRF(RF是ReFactorization的意思)：一个特殊的处理稀疏矩阵分解的库。

## 14.5 cuRAND 库

这是一个与随机数生成有关的库。该库产生的随机数有伪随机数 (pseudorandom numbers)和准随机数(quasirandom numbers)之分，但只需关注伪随机数。伪随机数序列是用某种确定性算法生成的，且满足大部分真正的随机数序列所满足的统计性质。它在很多计算机模拟，特别是蒙特卡罗模拟中应用广泛。

