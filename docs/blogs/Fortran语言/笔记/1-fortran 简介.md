# 01-Fortran简介

## 功能和用途

Fortran（Formula Translation）是一种编程语言，最初于1957年开发。也可以称其为数学公式翻译器，即：把数学公式翻译成计算机机器语言，快捷地进行数值计算和科学数据的处理。

它是一种**面向科学和工程计算**的高级编程语言，用于数值计算和科学计算。

Fortran具有强大的数学和科学计算功能，并且在**高性能计算领域**被广泛使用。

## fortran 语言的发展

- 57年，fortran开始使用
- 66年，fortran的语言标准得到了统一 （Fortran 66）
- 77年，fortran引入结构化设计 （Fortran 77）
- 92年，fortran 加入面向对象、指针改良语法编写格式 （**Fortran 90**）
- 97年，fortran 支持平行运算，（Fortran 95）

## fortran 的后缀格式

Fortran语言的源代码文件通常使用以下后缀格式：

1. .f/.F/.for：这是Fortran 77的常见后缀格式，表示源代码文件是Fortran语言的程序。

2. **.f90**：这是Fortran 90及其后续版本的后缀格式，表示源代码文件是Fortran 90或更高版本的程序。

3. .f95：这是Fortran 95的后缀格式，表示源代码文件是Fortran 95的程序。

4. .f03：这是Fortran 2003的后缀格式，表示源代码文件是Fortran 2003的程序。

5. .f08：这是Fortran 2008的后缀格式，表示源代码文件是Fortran 2008的程序。

这些后缀格式有助于标识Fortran语言的源代码文件类型，并提供了一致的命名约定。

## fortran是编译语言

fortran语言的运行与C和C++类似，都需要经过编译链接形成最后的可执行文件才能进行运行。

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230723202249.png)

由于fortran语言针对的是高性能计算，需要大量的数据输入和输出，因此，输入和输出通常以文件的形式存在。

而且，fortran语法可以与其他语言进行联合使用，比如，先使用fortran语言进行数据的处理，并对结果进行输出，而后，通过python对数据结果进行可视化显示。


## 编译器

和C++一样（[[2-C++的编译器|C++常见的编译器]]有：GCC、Clang、MVC++等），作为编译语言，Fortran语言也有许多编译器用来编译fortran语言的源代码。

比如：

1. GNU Fortran (**gfortran**)：这是[[3-GNU项目|GNU]]编译器套件（GCC）中的Fortran编译器。它是一个免费的开源编译器，广泛用于Linux和其他操作系统。

2. Intel Fortran Compiler：这是英特尔公司提供的Fortran编译器，专注于优化和性能。它支持多平台，并提供了一些特定于英特尔处理器的优化功能。

3. IBM XL Fortran：这是IBM公司提供的Fortran编译器，适用于IBM Power和IBM Z系列的平台。它具有高度的优化能力和对并行计算的支持。

4. NAG Fortran Compiler：这是由数值算法集团（NAG）提供的商业Fortran编译器。它专注于数值计算和科学计算，并提供了广泛的数值算法库。

5. PGI Fortran Compiler：这是由NVIDIA公司提供的Fortran编译器，主要用于GPU加速计算。它支持Fortran和CUDA混合编程，可以实现高性能的并行计算。

这些编译器在不同的平台和应用场景中具有各自的特点和优势。选择合适的Fortran编译器取决于你的需求、平台支持和个人偏好。
