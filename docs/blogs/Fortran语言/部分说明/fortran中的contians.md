`contains`是Fortran中的一个关键字，用于标识程序的主体部分中包含子程序或函数定义的位置。

在Fortran程序中，`contains`关键字通常用于将子程序或函数的定义放置在主程序内部。

通过使用`contains`关键字，可以将相关的子程序或函数与主程序组织在一起，提高代码的可读性和可维护性。

`contains`关键字通常位于**主程序的末尾**，在<font color="#c00000">它之后可以定义一个或多个子程序或函数</font>，

这些子程序或函数可以在主程序中进行调用，以实现代码的模块化和重用。

以下是一个示例程序，展示了`contains`关键字的使用：

```fortran
program main_program
  implicit none
  
  ! 主程序变量声明
  
  ! 主程序执行语句
  
  contains
    ! 子程序或函数定义
  
end program main_program
```
