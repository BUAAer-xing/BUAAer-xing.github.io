---
authors: [BUAAer-xing]
---


# 在mac上安装和使用Lapack和Lapacke

## Lapack和Lapacke简要介绍

**LAPACK**（Linear Algebra PACKage）是一种用于解决线性代数问题的软件库。它提供了许多常用的线性代数运算和求解方法，如矩阵分解、特征值计算、线性方程组求解等。LAPACK 是 Fortran 编写的，因此它的接口对于 Fortran 用户来说是自然的，但对于其他编程语言的用户来说可能需要使用相应的接口进行调用。

**LAPACKE** 是 LAPACK 的 C 语言封装，它提供了一组 C 语言调用 LAPACK 的接口，使 LAPACK 可以更容易地与 C 语言集成。LAPACKE 提供了一种在 C 语言中调用 LAPACK 子例程的方式，这对于那些使用 C 语言的应用程序和库来说是非常方便的。

## 安装Lapacke

### （1）使用brew进行安装lapack包

这里在mac上推荐大家使用包管理工具--brew进行安装，可以节省很多的时间。

安装命令如下：

`brew install lapack` 

这里为何仅仅安装lapack呢？是因为**lapacke所包含的代码已经集成到lapack中了**，如果想要使用lapacke就必须带着lapack一起进行安装。[Lapack官网](https://www.netlib.org/lapack/)的解释如下：**LAPAСK C INTERFACE is now included in the LAPACK package (in the lapacke directory)**

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240131102328.png)

同时，我们也可以直接在官网上下载下来源码进行查看，也可以看到lapack的文件结构。

![image.png|center|300](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240131102725.png)

### （2）查看lapack包的头文件路径和依赖路径

使用以下命令进行查看：

`brew list lapack`

如果出现下面的类似内容，就说明lapack已经成功安装了。

```
(base) buaaer@AirM1 ~ % brew list lapack
/opt/homebrew/Cellar/lapack/3.12.0/include/ (5 files) ------> 重要：头文件的路径
/opt/homebrew/Cellar/lapack/3.12.0/lib/libblas.3.12.0.dylib
/opt/homebrew/Cellar/lapack/3.12.0/lib/liblapack.3.12.0.dylib
/opt/homebrew/Cellar/lapack/3.12.0/lib/liblapacke.3.12.0.dylib
/opt/homebrew/Cellar/lapack/3.12.0/lib/cmake/ (8 files)
/opt/homebrew/Cellar/lapack/3.12.0/lib/pkgconfig/ (3 files)
/opt/homebrew/Cellar/lapack/3.12.0/lib/ (6 other files) -------->重要：头文件中函数的实现文件路径
```

## 使用Lapacke前的准备工作

### （1）为什么要进行准备工作？

这里，我使用的是vscode进行的编程，在引入相关头文件时，我发现vscode并无法直接检索到头文件所在的位置，因此我们需要在编程环境中进行手动配置vscode检索的头文件路径。

```c
#include<stdio.h>
#include<lapacke.h> // lapacke.h 头文件位置需要在安装后，在vscode编程环境中进行手动配置
int main (int argc, const char * argv[])
{
   double a[5][3] = {1,1,1,2,3,4,3,5,2,4,2,5,5,4,3};
   double b[5][2] = {-10,-3,12,14,14,12,16,16,18,16};
   lapack_int info,m,n,lda,ldb,nrhs;
   int i,j;

   m = 5;
   n = 3;
   nrhs = 2;
   lda = 3;
   ldb = 2;

   info = LAPACKE_dgels(LAPACK_ROW_MAJOR,'N',m,n,nrhs,*a,lda,*b,ldb);

   for(i=0;i<n;i++)
   {
      for(j=0;j<nrhs;j++)
      {
         printf("%lf ",b[i][j]);
      }
      printf("\n");
   }
   return (info);
}
```

### （2）生成/打开vscode当前工作区的配置文件

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240131104249.png)

首先，检查自己的项目文件夹下，是否有该.vscode文件，如果有就可以直接看下一步操作，如果没有，可以按照下面的步骤自动生成该配置文件。

前提条件：已经安装了C/C++的vscode插件。（估计要用到这个库的人，这些东西已经搞好了！）

自动生成当前工作区配置文件的步骤：
1. `command + shift + P` 打开vscode命令行。
2. 输入：C++，可以看到配置环境的选项。
	- ![600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240131104828.png)
3. 点击该项，即可生成如下的文件。
	- ![image.png|left|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240131105034.png)
	- 标亮的那个地方就是我们需要添加的头文件路径，其他的都是由插件自动生成的，无需我们管理。

### （3）将lapack的头文件路径添加到当前工作区的配置文件中

将上面lapack包的头文件添加到头文件路径下即可。

如何得到该头文件路径，请看：**安装Lapacke->(2)查看lapack包的头文件路径和依赖路径**

![image.png|center|800](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240131105722.png)

这样，vscode就可以检测到相关的头文件，并且可以给出相应的函数提示了。

## 使用CMake工具进行编译代码

为了简化编译的命令，这里直接使用CMake进行自动编译。

相关教程，可以看我之前写的博客：[简单CMake入门](https://blog.csdn.net/qq_45575167/article/details/134255316)

建立CMakeLists.txt文件，文件中的内容如下：

```
cmake_minimum_required(VERSION 3.10)
project(learn_lapack) # 项目名称

include_directories(/opt/homebrew/Cellar/lapack/3.12.0/include/) # 头文件路径
link_directories(/opt/homebrew/Cellar/lapack/3.12.0/lib/) # lib库路径，实现头文件中的函数

find_package(LAPACK REQUIRED) # 查找LAPACK库

add_executable(hello main.c) # 添加可执行文件

target_link_libraries(hello PRIVATE -llapacke -llapack -lblas -lm) # 链接LAPACK库
```

在这里简单解释一下，为什么在vscode当前的工作区配置中已经配置了头文件路径，这里为什么还要在CMake中配置一遍？

原因是，在vscode环境中配置的头文件路径仅仅是为了让vscode可以找到，而我们知道vscode本质上只是一个文本编辑器，并没有编译的功能，因此，即使vscode可以找到，也仅仅是vscode可以根据该路径下的头文件中的函数名给**出智能提醒**而已，并无法在编译运行时传递到CMake中，因此，在CMake中，我们仍然需要对头文件路径以及依赖包路径进行配置，才可以正确执行程序。

## 测试代码和执行结果

```c
#include<stdio.h>
#include<lapacke.h>

int main (int argc, const char * argv[])
{
   double a[5][3] = {1,1,1,2,3,4,3,5,2,4,2,5,5,4,3};
   double b[5][2] = {-10,-3,12,14,14,12,16,16,18,16};
   lapack_int info,m,n,lda,ldb,nrhs;
   int i,j;

   m = 5;
   n = 3;
   nrhs = 2;
   lda = 3;
   ldb = 2;

   info = LAPACKE_dgels(LAPACK_ROW_MAJOR,'N',m,n,nrhs,*a,lda,*b,ldb);

   for(i=0;i<n;i++)
   {
      for(j=0;j<nrhs;j++)
      {
         printf("%lf ",b[i][j]);
      }
      printf("\n");
   }
   return (info);
}

```

这段代码的作用是使用 LAPACK 中的 `dgels` 函数来解决最小二乘问题，即找到一个最接近矩阵方程 $Ax = B$ 的解 $x$，其中 $A$ 是系数矩阵，$B$ 是右侧矩阵。

![image.png|center|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240131111305.png)

结果如上所示，正确执行出相关结果，表示正确。
