# 02-Fortran语言程序设计基础

在fortran语言中，不区分英文字母的大小写

## fortran程序的主要结构

fortran程序通常以`program + 程序名` 进行开头，以 `end` 进行结尾

```fortran
program main

  ! This is a comment line; it is ignored by the compiler
  print *, 'Hello, World!'

end program main 
! end program
! end
```

## fortran的两种编写格式

Fortran语言有两种主要的编写格式，分别是**固定格式**（Fixed Form）和**自由格式**（Free Form）。

固定格式（Fixed Form）和自由格式（Free Form）的对比可以通过以下表格来展示：

|     特点     |     固定格式     |     自由格式     |
|:-:|-----------------|-----------------|
| 列限制       | 有，前6列用于行号和标签字段 | 无列限制，代码可以从任意列开始 |
| 缩进         | 有，以6个空格为一个缩进级别 | 无缩进要求，根据个人偏好进行 |
| 行长度       | 有限制，不超过72列，超出部分需要使用连字符 | 无行长度限制，可以编写较长的代码 |
| 注释         | 以"!"开头，从第7列开始 | 可以从任意列开始，使用"!" |
| 标识符       | 有限制，通常在第7列到第72列之间 | 无限制，可以出现在任意列 |
|扩展名|往往以.for或.f为扩展名|往往以.f90为扩展名|

通过以上对比，可以看出自由格式相对于固定格式更加灵活和现代化。

自由格式在大多数现代Fortran编译器中得到广泛支持，并**成为Fortran 90及其后续版本中的推荐编写格式**。

固定格式仍然保留是为了向后兼容和支持旧有的代码库。

## fortran的数据类型

Fortran中基本的数据类型及其描述如下：

| 数据类型    | 描述                                                         |
|:-------------:|--------------------------------------------------------------|
| Integer     | 整数类型，用于存储**整数**值。                                     |
| Real        | 浮点数类型，用于存储**实数**值。（小数和整数）                                   |
| Complex     | 复数类型，包含实部和虚部，用于存储复数值。                     |
| Character   | 字符类型，用于存储字符和**字符串值**。                             |
| Logical     | 逻辑类型，用于存储**逻辑值**（True或False）。                       |

这些数据类型在Fortran中用于声明变量，以便存储不同类型的数据。

根据需要，可以选择适当的数据类型来满足特定的计算和存储需求。

## fortran的数学符号

Fortran中常用的数学运算符号如下：

| 运算符   | 描述                 |
|:----------:|:----------------------:|
| +        | 加法                 |
| -        | 减法                 |
| *        | 乘法                 |
| /        | 除法                 |
| **       | 幂运算               |
| mod()    | 取模运算             |
| sqrt()   | 开平方               |
| exp()    | 指数函数             |
| log()    | 自然对数             |
| abs()    | 绝对值               |
| max()    | 最大值               |
| min()    | 最小值               |
| real()   | 转换为实数类型       |
| int()    | 转换为整数类型       |
| cmplx()  | 转换为复数类型       |

这些运算符和函数可用于进行数学计算和操作，根据需要选择适当的运算符和函数来实现所需的数学运算。
