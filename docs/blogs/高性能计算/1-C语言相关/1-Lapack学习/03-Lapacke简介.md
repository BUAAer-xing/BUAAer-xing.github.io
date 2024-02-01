大部分源自对文档的翻译，添加一些自己不懂的解释。
## 概述

LAPACKE（Linear Algebra PACKage for C Environment）是一个用于C语言的线性代数库，它构建在LAPACK（Linear Algebra PACKage）之上。LAPACK是一组高性能、可移植的线性代数例程，主要用于解决线性代数问题，如线性方程组的求解、特征值计算、奇异值分解等。

LAPACKE旨在为C语言用户提供方便的接口，使其能够轻松地使用LAPACK中的例程。它提供了一组C语言函数，这些函数调用LAPACK例程，并提供了更符合C语言编程风格的接口。LAPACKE的设计旨在简化线性代数计算的使用，并与C语言的语法和习惯相一致。

LAPACKE的功能涵盖了广泛的线性代数运算，包括解线性方程组、特征值计算、奇异值分解等。通过LAPACKE，开发者可以在C语言中利用LAPACK的高性能线性代数例程，无需直接调用Fortran接口。

与CBLAS类似，LAPACKE的详细信息通常包含在特定的头文件中，以便在C程序中使用。在使用LAPACKE时，开发者可以通过简单的C语言调用来利用LAPACK中的功能，从而进行高效的线性代数计算。

文档描述了一个对LAPACK的两层C语言接口，包括**高级接口**和**中级接口**。

1. **高级接口（High-Level Interface）：**
    - **内部处理内存分配：** 高级接口会在其内部处理所有工作空间（workspace）的内存分配，无需用户手动管理。这意味着在使用高级接口时，开发者不必担心手动分配和释放内存，因为所有这些操作都由接口自动处理。
    - **便捷性：** 高级接口旨在提供更为便捷的使用体验，让用户能够更轻松地调用LAPACK功能而不用过多关注内部细节。
2. **中级接口（Middle-Level Interface）：**
    - **要求用户提供工作空间数组：** 与高级接口不同，中级接口要求用户像原始的Fortran接口一样，手动提供工作空间数组。这表示用户需要了解并预先分配适当大小的内存来存储中间计算结果和临时数据。
    - **类似Fortran接口：** 中级接口的设计更接近原始的Fortran接口，这对于已经熟悉Fortran风格的用户可能更为熟悉。用户需要对工作空间的管理有一定的了解。

**这两个接口都支持列主序和行主序矩阵**。这两个接口的原型、相关宏和类型定义都包含在头文件`lapacke.h`中。

### 函数的命名规则

- 高级接口的命名方案是：<font color='red'><b>采用Fortran LAPACK例程的名称，将其转换为小写，并添加前缀</b></font>。
	- 例如，LAPACK子例程变成。例如，LAPACK子例程变成`LAPACKE_dgetrf`。
- 中级接口的命名方案是：<font color='red'><b>采用Fortran LAPACK例程的名称，将其转换为小写，然后添加前缀和后缀</b></font>。
	- 例如，LAPACK子例程变成`LAPACKE_dgetrf_work`。

#### Lapack的命名规则

所有函数都是以 <font color='red'><b>XYYZZZ</b></font> 的形式命名，对于某些函数，没有第六个字符，只是 XYYZZ 的形式。

- **X - 第一个字符代表的是数据类型：**
	- S    ------>      REAL，单精度实数
	- D    ------>      DOUBLE PRECISION，双精度实数
	- C    ------>      COMPLEX，单精度复数
	- Z    ------>      COMPLEX\*16 或 DOUBLE COMPLEX
	- 注意：在新版 LAPACK 中含有使用重复迭代法的函数 DSGESV 和 ZCDESV。
		- 头 2 个字母表示使用的精度：DS，输入数据是 double 双精度，算法使用单精度。ZC，输入数据是 complex\*16，算法使用 complex 单精度复数
- **YY - 第二三个字符代表的是数组的类型：**
	- GE    ------>      general 一般情形（即非对称，在有些情形下为矩形）
	- DI     ------>      diagonal，对角矩阵
	- 还有很多。
- **ZZZ - 最后三个字母 ZZZ 代表计算方法：**
	- 比如，S   GE   BRD 是一个单精度函数，用于把一个实数一般阵压缩为双对角阵（a bidiagonal reduction，即 BRD）。

### 复杂数据类型

复杂数据类型由宏`lapack_complex_float`和`lapack_complex_double`定义，分别表示单精度和双精度复杂数据类型。

假定**在整个过程中，实部和虚部在内存中是连续存储的，实部在前。**

`lapack_complex_float`和`lapack_complex_double`这两个宏可以是C99的`_Complex`类型、C语言中定义的结构体类型、C++ STL中的复数类型，或者自定义的复数类型。

有关详细信息，请参阅`lapacke.h`。

---
以下是我的猜测：

在C语言中，结构体（Struct）用于存储多个不同数据类型的元素。如果要表示复数类型的结构体，可以使用以下示例：

```c
// 定义复数结构体
typedef struct {
    float real;  // 实部
    float imag;  // 虚部
} lapack_complex_float;

typedef struct {
    double real;  // 实部
    double imag;  // 虚部
} lapack_complex_double;
```

上述代码定义了两个结构体类型，`lapack_complex_float`和`lapack_complex_double`，分别表示单精度和双精度复数。每个结构体包含两个成员，分别用于存储实部（`real`）和虚部（`imag`）。

使用这些结构体类型时，可以通过创建相应的结构体变量，并为其成员赋值来表示复数。例如：

```c
// 创建并初始化复数变量
lapack_complex_float complexFloatNumber = {1.0f, 2.0f};  // 1.0 + 2.0i
lapack_complex_double complexDoubleNumber = {3.0, 4.0};   // 3.0 + 4.0i
```

这样，你就可以使用这些结构体变量来表示复数，并将其传递给需要复数作为参数的函数或操作。

---

### 数组参数

**数组以指针形式传递**，而不是指向指针的形式。

所有接受一个或多个二维数组指针的 LAPACKE 程序都会接收一个额外的 int 类型参数。该参数必须等于 **LAPACK_ROW_MAJOR** 或 **LAPACK_COL_MAJOR**，这两者在 lapacke.h 中定义，指定数组是按行主序还是列主序存储的。如果一个程序有多个数组输入，它们必须都使用相同的顺序。

<font color='skyblue' face='宋体-简'>请注意，使用行主序可能会比使用列主序需要更多的内存和时间，因为该程序必须将行主序转置为底层 LAPACK 程序所需的列主序。</font>

在FORTRAN LAPACK程序中，每个二维数组参数都有一个额外的参数，指定其主导维度。
- 对于行主序的二维数组，假定行内的元素是连续的，而从一行到下一行的元素被假定为主导维度分开。
- 对于列主序的二维数组，假定列内的元素是连续的，而从一列到下一列的元素被假定为主导维度分开。

### 参数的别名问题

除非另有说明，否则在调用与 LAPACK 的 C 接口相关的函数时，只有输入参数（即按值传递的标量和带有 const 限定符的数组）才能在法律上进行别名。

### INFO 参数

LAPACKE接口函数将它们的 lapack_int 返回值设置为 INFO 参数的值，该参数包含错误和退出条件等信息。

如果执行结果是正确的，则返回的参数是0，如果在执行过程中，遇到错误，则返回的参数则是发生错误的代码。

### NaN检查

高级接口在调用任何LAPACK例程之前，包括了一个可选的NaN检查，它默认启用。

此选项影响所有例程。**如果输入包含任何NaN值，则对应的矩阵输入参数将用INFO参数错误标记。例如，如果发现第五个参数包含NaN，函数将返回值-5。**

NaN检查以及其他参数可以通过在lapacke.h中定义LAPACK_DISABLE_NAN_CHECK宏来禁用。中级接口不包含NaN检查。

### 整数

在LAPACKE中，具有FORTRAN整数类型的变量被转换为lapack_int。这符合可修改整数类型大小的要求，尤其是在使用ILP64编程模型时：将lapack_int重新定义为long int（8字节）就足以支持该模型，因为默认情况下lapack_int被定义为int（4字节），从而支持LP64编程模型。

### 逻辑值

FORTRAN中的逻辑值被转换为lapack_logical，其被定义为lapack_int。

### 内存管理

所有的内存管理都由`LAPACKE_malloc`和`LAPACKE_free`函数处理。这允许用户通过修改lapacke.h中的定义，轻松使用自己的内存管理器而不是默认的内存管理器。

在这个接口中，应该在这些内存管理例程和底层LAPACK例程是线程安全的范围内是线程安全的。

### 新的错误代码

由于高级接口不使用工作数组，因此在用户耗尽内存时需要错误通知。

- 如果无法分配工作数组，函数将返回LAPACK_WORK_MEMORY_ERROR；
- 如果内存不足以完成转置，将返回LAPACK_TRANSPOSE_MEMORY_ERROR。

## 函数列表

该部分列出了目前在LAPACKE C接口中可用的LAPACK子例程。

下面给出了LAPACK的基本名称；

相应的LAPACKE函数名称为LAPACKE_xbase或LAPACKE_xbase_work，其中x为类型：s或d表示单精度实数，c或z表示双精度复数，base表示基本名称。

函数原型在文件lapacke.h中给出。有关例程及其参数的详细信息，请参阅LAPACK文档。

此处省略。

### 实函数

以下是在高级接口和中级接口中支持单精度（s）和双精度（d+）的LAPACK子例程基本名称：

```
bdsdc bdsqr disna gbbrd gbcon gbequ gbequb gbrfs gbrfsx gbsv gbsvx gbsvxx gbtrf
gbtrs gebak gebal gebrd gecon geequ geequb gees geesx geev geevx gehrd gejsv
gelqf gels gelsd gelss gelsy geqlf geqp3 geqpf geqrf geqrfp gerfs gerfsx gerqf
gesdd gesv gesvd gesvj gesvx gesvxx getrf getri getrs ggbak ggbal gges ggesx
ggev ggevx ggglm gghrd gglse ggqrf ggrqf ggsvd ggsvp gtcon gtrfs gtsv gtsvx
gttrf gttrs hgeqz hsein hseqr opgtr opmtr orgbr orghr orglq orgql orgqr orgrq
orgtr ormbr ormhr ormlq ormql ormqr ormrq ormrz ormtr pbcon pbequ pbrfs pbstf
pbsv pbsvx pbtrf pbtrs pftrf pftri pftrs pocon poequ poequb porfs porfsx posv
posvx posvxx potrf potri potrs ppcon ppequ pprfs ppsv ppsvx pptrf pptri pptrs
pstrf ptcon pteqr ptrfs ptsv ptsvx pttrf pttrs sbev sbevd sbevx sbgst sbgv
sbgvd sbgvx sbtrd sfrk spcon spev spevd spevx spgst spgv spgvd spgvx sprfs spsv
spsvx sptrd sptrf sptri sptrs stebz stedc stegr stein stemr steqr sterf stev
stevd stevr stevx sycon syequb syev syevd syevr syevx sygst sygv sygvd sygvx
syrfs syrfsx sysv sysvx sysvxx sytrd sytrf sytri sytrs tbcon tbrfs tbtrs tfsm
tftri tfttp tfttr tgevc tgexc tgsen tgsja tgsna tgsyl tpcon tprfs tptri tptrs
tpttf tpttr trcon trevc trexc trrfs trsen trsna trsyl trtri trtrs trttf trttp
tzrzf
```

### 复杂函数

以下是在高级接口和中级接口中支持复杂单精度（c）和复杂双精度（z）的LAPACK子例程基本名称：

```
bdsqr gbbrd gbcon gbequ gbequb gbrfs gbrfsx gbsv gbsvx gbsvxx gbtrf gbtrs gebak
gebal gebrd gecon geequ geequb gees geesx geev geevx gehrd gelqf gels gelsd
gelss gelsy geqlf geqp3 geqpf geqrf geqrfp gerfs gerfsx gerqf gesdd gesv gesvd
gesvx gesvxx getrf getri getrs ggbak ggbal gges ggesx ggev ggevx ggglm gghrd
gglse ggqrf ggrqf ggsvd ggsvp gtcon gtrfs gtsv gtsvx gttrf gttrs hbev hbevd
hbevx hbgst hbgv hbgvd hbgvx hbtrd hecon heequb heev heevd heevr heevx hegst
hegv hegvd hegvx herfs herfsx hesv hesvx hesvxx hetrd hetrf hetri hetrs hfrk
hgeqz hpcon hpev hpevd hpevx hpgst hpgv hpgvd hpgvx hprfs hpsv hpsvx hptrd
hptrf hptri hptrs hsein hseqr pbcon pbequ pbrfs pbstf pbsv pbsvx pbtrf pbtrs
pftrf pftri pftrs pocon poequ poequb porfs porfsx posv posvx posvxx potrf potri
potrs ppcon ppequ pprfs ppsv ppsvx pptrf pptri pptrs pstrf ptcon pteqr ptrfs
ptsv ptsvx pttrf pttrs spcon sprfs spsv spsvx sptrf sptri sptrs stedc stegr
stein stemr steqr sycon syequb syrfs syrfsx sysv sysvx sysvxx sytrf sytri sytrs
tbcon tbrfs tbtrs tfsm tftri tfttp tfttr tgevc tgexc tgsen tgsja tgsna tgsyl
tpcon tprfs tptri tptrs tpttf tpttr trcon trevc trexc trrfs trsen trsna trsyl
trtri trtrs trttf trttp tzrzf ungbr unghr unglq ungql ungqr ungrq ungtr unmbr
unmhr unmlq unmql unmqr unmrq unmrz unmtr upgtr upmtr
```

### 混合精度函数

以下LAPACK子例程基本名称仅支持双精度（d）和复杂双精度（z）：

```
sgesv sposv
```

## 例子

```cpp
/* Calling DGELS using row-major order */

#include <stdio.h>
#include <lapacke.h>

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
   return(info);
}
```














