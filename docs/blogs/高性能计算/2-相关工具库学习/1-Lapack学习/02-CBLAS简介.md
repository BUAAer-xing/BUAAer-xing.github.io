## 1. 介绍

CBLAS（C Basic Linear Algebra Subprograms）是对基本线性代数子程序（BLAS）的C语言接口。BLAS是一组高度优化的线性代数例程，用于执行矩阵乘法、矩阵-向量乘法以及其他与线性代数相关的基本运算。这些例程通常由Fortran语言编写，但CBLAS提供了对它们的C语言接口，使得在C程序中能够方便地使用这些高性能的线性代数例程。

CBLAS接口支持多种矩阵排序方式，包括行主序和列主序，通过使用`layout`参数进行控制。

<font color='red'><b>它还提供了一种命名方案，将Fortran BLAS例程的名称转换为小写，并添加前缀`cblas_`。</b></font>

CBLAS接口的详细信息，包括函数原型、宏和类型定义，通常包含在头文件cblas.h中。通过使用CBLAS，开发者可以在C程序中高效地执行各种线性代数操作，而无需直接使用底层的Fortran实现。

### 1.1 命名方案

CBLAS接口的命名方案是将Fortran BLAS例程名转换为小写，并添加前缀`cblas_`。例如，BLAS例程`DGEMM`变为`cblas_dgemm`。

CBLAS例程还支持带有`_64`后缀，以在LP64接口库（默认构建配置）中启用对大数据数组的支持。该后缀允许在一个应用程序中混合使用LP64和ILP64编程模型。例如，`cblas_dgemm`可以与支持64位整数类型的`cblas_dgemm_64`混合使用。

### 1.2 整数

Fortran类型整数的变量在CBLAS中转换为`CBLAS_INT`。默认情况下，CBLAS接口使用32位整数类型构建，但可以重新定义为64位整数类型。

## 2. 函数列表

本节包含当前可用的CBLAS接口列表。

### 2.1 BLAS 第一级

* 单精度实数:
  ```
  SROTG SROTMG SROT SROTM SSWAP SSCAL
  SCOPY SAXPY SDOT  SDSDOT SNRM2 SASUM 
  ISAMAX
  ```
* 双精度实数:
  ```
  DROTG DROTMG DROT DROTM DSWAP DSCAL
  DCOPY DAXPY DDOT  DSDOT DNRM2 DASUM 
  IDAMAX
  ```
* 单精度复数:
  ```
  CROTG CSROT CSWAP CSCAL CSSCAL CCOPY 
  CAXPY CDOTU_SUB CDOTC_SUB ICAMAX SCABS1
  ```
* 双精度复数:
  ```
  ZROTG ZDROT ZSWAP ZSCAL ZDSCAL ZCOPY 
  ZAXPY ZDOTU_SUB ZDOTC_SUB IZAMAX DCABS1
  DZNRM2 DZASUM
  ```
  
### 2.2 BLAS 第二级

* 单精度实数:
  ```
  SGEMV SGBMV SGER  SSBMV SSPMV SSPR
  SSPR2 SSYMV SSYR  SSYR2 STBMV STBSV
  STPMV STPSV STRMV STRSV
  ```
* 双精度实数:
  ```
  DGEMV DGBMV DGER  DSBMV DSPMV DSPR
  DSPR2 DSYMV DSYR  DSYR2 DTBMV DTBSV
  DTPMV DTPSV DTRMV DTRSV
  ```
* 单精度复数:
  ```
  CGEMV CGBMV CHEMV CHBMV CHPMV CTRMV
  CTBMV CTPMV CTRSV CTBSV CTPSV CGERU
  CGERC CHER  CHER2 CHPR  CHPR2
  ```
* 双精度复数:
  ```
  ZGEMV ZGBMV ZHEMV ZHBMV ZHPMV ZTRMV
  ZTBMV ZTPMV ZTRSV ZTBSV ZTPSV ZGERU
  ZGERC ZHER  ZHER2 ZHPR  ZHPR2
  ```
  
### 2.3 BLAS 第三级

* 单精度实数:
  ```
  SGEMM SSYMM SSYRK SSERK2K STRMM STRSM
  ```
* 双精度实数:
  ```
  DGEMM DSYMM DSYRK DSERK2K DTRMM DTRSM
  ```
* 单精度复数:
  ```
  CGEMM CSYMM CHEMM CHERK CHER2K CTRMM
  CTRSM CSYRK CSYR2K
  ```
* 双精度复数:
  ```
  ZGEMM ZSYMM ZHEMM ZHERK ZHER2K ZTRMM 
  ZTRSM ZSYRK ZSYR2K
  ```

## 3. 例子

本节包含从C程序调用CBLAS函数的示例。

### 3.1 调用 DGEMV

变量声明应如下所示：
```
   double *a, *x, *y;
   double alpha, beta;
   CBLAS_INT m, n, lda, incx, incy;
```
然后，CBLAS函数调用为：
```
cblas_dgemv( CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta,
                y, incy );
```

### 3.2 调用 DGEMV_64

变量声明应如下所示：
```
   double *a, *x, *y;
   double alpha, beta;
   int64_t m, n, lda, incx, incy;
```
然后，CBLAS函数调用为：
```
cblas_dgemv_64( CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta,
                y, incy );
```