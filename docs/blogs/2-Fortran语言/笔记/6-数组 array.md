# 06-数组array

## 基本使用

### 一维数组

数组可以一次声明处一长串相同数据类型的变量

声明数组的语法（以非自定义类型integer为例）
- `integer a(10)`
- `integer,dimension(10) :: a` : 先声明a是整型，再声明a是大小为10的数组

自定义类型的数组声明
- `type(person)::a(10)` : 用person新类型来声明数组

自定义类型数组的访问
- `a(2)%name`：访问数组的第2个自定义元素的name属性


### 二维数组

声明二维数组（以非自定义类型integer为例）
- `integer a(3,3)`
- `integer,dimension(3,3) :: a`

访问：`a(1,2)`

### 多维数组

fortran 最多可以声明高达7维的数组

每一个维度都有对应的大小

```fortran
integer a(D1,D2,…,Dn) ! n维数组
a(I1,I2,…,In) ! 使用n维数组时，要给n个坐标值
```

### 特殊的数组声明

在没有特别赋值的情况下，数组的索引值都是从1开始的（即数组的下界默认为1）

integer a(5) ===> a(1)、a(2)、a(3)、a(4)、a(5)

可以特别赋值数组的坐标值的使用范围：（）指的是**闭区间**

- `integer a(0:5)`  ===> a(0)、a(1)、a(2)、a(3)、a(4)、a(5)
- `integer a(-3:3)` ===> a(-3)、a(-2)、a(-1)、a(0)、a(1)、a(2)、a(3)
- `integer a(5, 0:5)`===> a(1~5, 0~5)
- `integer b(2:3, -1:3)` ===> b(2~3, -1~3)

## 数组内容的设置

### 赋初值

#### data 语句

在Fortran中，`DATA`语句用于在程序中为变量赋初始值。`DATA`语句的语法如下：

```fortran
DATA variable1 / value1 /, variable2 / value2 /, ...
```

其中，`variable1`, `variable2`, ... 是要赋值的变量，`value1`, `value2`, ... 是对应的初始值。

```fortran
program dataexample
  integer :: x, y, z
  real :: a, b, c
  
  data x / 10 /, y / 20 /, z / 30 /
  data a / 1.5 /, b / 2.5 /, c / 3.5 /
  
  write(*, *) "x =", x
  write(*, *) "y =", y
  write(*, *) "z =", z
  
  write(*, *) "a =", a
  write(*, *) "b =", b
  write(*, *) "c =", c
end program dataexample
```

#### 数组内容通过data来初始化

例如
```fortran
program main
    implicit none
    integer a(5)
    integer b(5)
    integer c(5)
    data a /1,2,3,4,5/
    ! 不做任何赋初值操作
    data c/5*0/
    print *,a
    print *,b
    print *,c
end
```

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725163647.png)

#### 通过隐含式循环来赋值

例如：

```fortran
integer a(5)
integer i
data (a(i),i=2,4) /2,3,4/
```

初值的设置结果为：2，3，4 位置上的数组值分别为 2，3，4

```fortran
program main
	integer a(8)
	integer i
	data (a(i),i=2,4) /2,3,4/
	print *,a
	print *,(a(i),i=2,4)
end

```

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725164604.png)

注意📢：
- 隐含式`(a(i),i=min,max,stride)`
- 也可以进行多层嵌套
	- `((a(i,j),j=1,2),i=1,2)`

####  在Fortran90中赋初始值

可以省略 data

`integer a(5) = (/ 1,2,3,4,5 /)` ! 括号和除号之间不能有空格

直接把初值写在声明后面时，每个元素都要给定初始值

而且，直接赋值的`(/  ...  /) `的 `...` 中也支持**隐含式循环**

``` fortran
program main
    integer :: i
    integer :: a(5) = (/ 1,2,3,4,5 /)
    integer :: b(5) = (/ 1,(0,i=2,4),5 /)
    integer :: c(5) = 100 ! 一次把整个数组内容设置为同一个数值
    print *,a
    print *,b
    print *,c
end
```

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725165439.png)

### 对整个数组的操作

fortran 90 可以通过简单的命令来操作数组

- `a=5`
	- a是一个任意维数及大小的数组。
	- 这个命令是把数组a的每个元素的值都设置为5。
	- 以一维的情況来说，即a(i) = 5
- ` a=(/   1,2,3   /)`
	- a(1)=1, a(2)=2，a(3)=3。
	- 等号右边所提供的数字个数，必须跟数组a的大小一样
- `a=b`
	- a和b是同样维数及大小的数组。
	- 这个命令会把数组a同样位置元素的内容设置成和数组b一样。
	- 以一维的情况来说，即$a(i) = b(i)$
- `a=b+c`
	- a,b，c是三个同样维数及大小的数组
	- 这个命令会把数组b及c中同样位置的数值相加，得到的数值再放回数组a同样的位置中。
	- 以二维的情况来说，即$a(i,j)=b(i,j)+c(i,j)$
- `a=b-c`
	- a，b，c是三个同样维数及大小的数组
	- 这个命令会把数组b及c中同样位置的数值相减，得到的数值再放回数组a
	- 同样的位置中。以二维的情況来说，即$a(i,j)=b(i,j)-c(i,j)$
- `a=b*c`
	- a，b，c是三个同样维数及大小的数组
	- 执行后数组a的每一个元素值为相同位置的数组b元素乘以数组c元素。
	- 以二维的情况来说，即$a(i,j)=b(i, j) \times c(i, j)$
- `a=b/c`
	- a，b，c是三个同样维数及大小的数组
	- 执行后数组a的每一个元素值为相同位 置的数组b元素除以数组C元素。
	- 以二维的情况来说，即$a(i,j) = b(i,j) \div c(i,j)$
- `a=sin(b)`
	- 矩阵a的每一个元素为矩阵b元素的sin值
	- 数组b必须是实型数组，才能使用sin函数。
	- 以一维的情况来说，即$a(i)=sin(b(i))$
- `a=b>c`
	- a,b,c是三个同样维数及大小的数组
	- 其中数组a是逻辑型数组，数组b、c则为元素间可以进行比较的变量类型。
	- 以一维情況来说，即为
```fortran
if (b(i)>c(i)) then 
	a(i)=.true.
else
	a(i)=.false.
endif
```

### 对部分数组的操作

和python中的类似，fortran中使用的前闭后闭区间

- `a(3:3)`
	- 与a(3）相同
- `a(3:5)=5`
	- 把a(3)~a(5)的内容设置成5，其他值不变
- `a(3:)=5`
	- 把a(3)之后元素的内容设置成5，其他值不变
- `a(3:5)=(/3,4,5/)`
	- 设置a(3)=3， a(4)=4, a(5)=5，其他值不变。
	- 等号左边所赋值的数组元素数目必须跟等号右边提供的数字个数相同
- `a(1:5:2)=3`
	- 设置a(1)=3, a(3)=3， a(5)=3。类似隐含式循环
- `a(1:10)=a(10:1:-1)`
	- 使用类似隐含式循环的方法把a(1~10)的内容给翻转
- `a(:)=b(:,2)`
	- 假设a声明为a(5)，b声明为b(5,5)。
	- 等号右边b(：,2)是取出b(1~5,2)这5个元素。
	- 而a是一维数组，所以a(:)和直接使用a是一样的，都是指a(1~5)这5个元素。
	- 只要等号两边的元素数日一样多就成立。
	- 执行结果为a(i)=b(i,2）
- `a(:,:)=b(:,:,1)`
	- 假设a声明为a(5,5），b声明为b(5,5,5)。
	- 等号右边b(;,:1）是取出b（1~5，1~5，1)这25个元素。
	- 而a是二维数组，所以a(:,:)和直按使用a是一样的，都是指a(1~5,1~5)这25个元素。
	- 执行结果为a(i,j)=b(i,j,1)

### where 语句

fortran 95新添加的内容

对比mysql的语句进行理解

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725175225.png)

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725175249.png)

### forall 语句

fortran 95 使用隐含式循环来使用数组的算法

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725175421.png)

## 数组的保存规则

### 一维数组

连续存放

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725175840.png)


### 二维数组

按照列的大小，从小往大开始排序

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725175858.png)

所以，以后循环的内层循环应该从i开始，然后外层循环是j，这样可以提高效率

```fortran
do i=1,4
	do j=1,4
		....
		a(j,i)
	end do
end do
```

### 多维数组

先放入较低维的元素，再放入较高维的元素

最先变化的是低维的元素，最高维的元素（下标）最后变化。


![image.png|center|300](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725180236.png)

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725180458.png)

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725180839.png)

### 数据的存储方式

根据内存的排列顺序来设置数值

```fortran
integer :: a(2,2) = (/ 1,2,3,4 /)
!a(1,1)=1,a(2,1)=2,a(1,2)=3,a(2,2)=4
```

![image.png|center|300](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230725193338.png)

## 动态数组

在Fortran中，可以使用`allocate`关键字来动态分配数组的内存空间。

示例：

```fortran
program dynamic_array_example
  integer, allocatable :: dynamic_array(:)
  integer :: size, i
  
  ! 获取数组大小
  write(*, *) "Enter the size of the array:"
  read(*, *) size
  
  ! 动态分配数组内存空间
  allocate(dynamic_array(size))
  
  ! 初始化数组
  do i = 1, size
    dynamic_array(i) = i
  end do
  
  ! 输出数组元素
  write(*, *) "Array elements:"
  do i = 1, size
    write(*, *) dynamic_array(i)
  end do
  
  ! 释放数组内存空间
  deallocate(dynamic_array)
  
end program dynamic_array_example
```

在上述示例中，声明了一个可分配的整数数组`dynamic_array`，并没有为其分配内存空间。

然后，通过使用`allocate`关键字动态地为数组分配内存空间，根据用户输入的大小来确定数组的大小。

接下来，使用循环结构初始化数组的每个元素，将其设置为数组索引值。

然后，我们使用循环结构输出数组的每个元素。

最后，我们使用`deallocate`关键字释放数组的内存空间，以便在程序执行完毕后释放内存。

通过动态数组，可以在运行时根据需要动态地分配和管理数组的内存空间，使程序更加灵活和可扩展。

