---
title: Shell编程
---
# Shell 编程

## 1-学习shell的意义

1. 在进行服务器集群管理时，需要编写shell程序来进行服务器管理
2. 编写一些shell脚本进行运行程序或者是服务器的维护
3. 编写shell程序来管理集群


## 2-shell 的概述

Shell是一种命令行解释器，它是操作系统和用户之间进行交互的界面。在计算机领域中，Shell是一种用于执行命令、管理文件和程序的软件。 ^74c152

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709171520.png)

Shell提供了一个命令行界面，**用户可以通过输入命令来与操作系统进行交互**。用户可以使用Shell来执行各种操作，如运行程序、管理文件和目录、设置环境变量、执行脚本等。

Shell还提供了一些特殊的功能，如命令历史记录、命令补全、管道操作等，以方便用户进行操作和管理。

在Unix和类Unix系统中，常见的Shell包括Bourne Shell (sh)、Bash (Bourne Again Shell)、C Shell (csh)、Korn Shell (ksh)等。每种Shell都有自己的特点和语法，但它们都提供了类似的基本功能。

Shell是一种命令行解释器，它提供了用户与操作系统交互的界面，使用户能够执行命令、管理文件和程序，并进行各种操作和管理。

## 3-Shell 脚本的执行方式

### 脚本格式要求

1. 脚本要以`#!/bin/bash`开头
	- 当在一个脚本文件的开头添加`#!/bin/bash`时，它告诉操作系统在执行该脚本时使用**Bash来解释和执行脚本中的命令**。
	- 这样就可以直接运行脚本文件，而不需要在命令行中显式地指定解释器。
2. 脚本需要有**可执行权限**!!!

### 编写第一个Shell脚本

创建一个shell脚本，用来输出hello world！

```bash
#!/bin/bash
echo "hello world!"
```

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720180017.png)

### 脚本的常用执行方式

- 方式一：**输入脚本的绝对路径或相对路径**
	- ❗️:首先要赋予新建脚本执行的权限（+x），然后再去执行脚本
	- 比如：
		- 相对路径：`./hello.sh`
		- 绝对路径：`/home/shell-test/hello.sh`
- 方式二：**sh + 脚本** 进行运行
	- 说明：不用赋予脚本执行权限，直接执行即可
	- 比如：sh hello.sh 也可以使用绝对路径

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720181217.png)

# 4-shell 的变量

### 变量介绍

Linux shell 中的变量分为 ： **系统变量**和**用户自定义变量**

#### 系统变量
系统变量的例子：\$HOME、\$PWD、\$SHELL 、\$USER 等等

显示所有的系统变量：`set`

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720193912.png)

#### 自定义变量

基本语法
- 定义变量： `变量名 = 值`
- 撤销变量： `unset 变量`
- 声明静态变量： `readonly 变量 `
	- 注意： 静态变量不能进行 unset

定义变量的规则：
- 和其它语言一样
- **等号两侧不能有空格**
- 变量名称一般习惯为大写

变量的基本操作

```bash
#!/bin/bash
#定义变量
A=100
#输出变量需要加上$
echo A=$A
echo "A=$A"
#撤销变量A
unset A
echo $A
#声明静态的变量B，不能unset
readonly B=2
echo "B=$B"
#将指令返回的结果赋值给变量
DATE=`date`
CAL=$(cal)
echo "$DATE and $CAL"

:<<! 
多行注释
!

```

📢：在echo输出的两种方式中，采用字符串输出可以直接输出空格，而如果不加双引号进行输出，则源文件中的空格将会被忽略

**将Linux命令的返回值赋值给变量**
(1):通过反引号运行里面的命令，并把结果返回给变量A
```bash
A=`date`
```
(2):通过$()运算，运行里面的命令，并把结果返回给变量A
```bash
A=$(date) #这种写法更好，即使管道操作符，也能正确执行
```

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720195436.png)

# 5-环境变量（全局变量）

### 概述

这里的环境变量可以近似的看做为全局变量

![[全局变量示例.excalidraw|center|600]]

### 基本语法

- `export 变量名=变量值`
	- 将shell变量输出为环境变量（全局变量）
- `source 配置文件` 
	- 让修改后的信息立即生效
- `echo $变量名` 查询环境变量的值

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720204450.png)

# 6-位置参数变量

### 概要

位置参数变量：**让脚本可以通过命令行获取到各个参数的信息**

在Shell脚本中，位置参数变量是一组特殊的变量，用于**存储通过命令行传递给脚本的参数值**。

### 基本语法

位置参数变量以数字作为前缀，从1开始递增，表示参数的位置顺序。

常用的位置参数变量：

1. `$0`：表示**脚本本身的名称**。

2. `$1`、`$2`、`$3`，以此类推：表示命令行中传递给脚本的位置参数的值。例如，`$1`表示第一个参数，`$2`表示第二个参数，以此类推。十以上的参数，需要用到大括号进行包含。比如 `${10}`

3. `$*`：表示所有位置参数的值，作为一个单独的**字符串**。参数之间以空格分隔。
 ^9f4879
4. `$@`：表示所有位置参数的值，作为一个**数组**。每个参数都可以单独访问。

5. `$#`：表示传递给脚本的**位置参数的个数**。

通过使用这些位置参数变量，可以在Shell脚本中获取和处理命令行传递的参数值。

例如，`$1`可以用于获取第一个参数的值，`$#`可以用于获取参数的个数。

### 位置参数示例

以下是一个示例脚本，演示如何使用位置参数变量：

```bash
#!/bin/bash

echo "脚本名称：$0"
echo "第一个参数：$1"
echo "第二个参数：$2"
echo "所有参数：$*"
echo "参数个数：$#"
```

当执行这个脚本并传递参数时，脚本将打印出相应的位置参数值和参数个数。

![image.png|center|600|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720205539.png)

# 7-预定义变量

### 概述

预定义变量：就是shell设计者事先已经定义好的变量，可以直接在shell脚本中进行使用

### 基本用法

1. `$$`：表示当前脚本的进程ID。
2. `$!`：表示最后一个在后台运行的进程的进程ID。
3. `$?`：表示上一个命令的退出状态码。如果命令执行成功，该值为0；如果命令执行失败，该值为非零。

# 8-运算符

### 概要

在shell中如何进行各种运算操作

### 基本语法

计算表达式，在shell中表达式的计算，需要通过特殊的格式进行

三种方式
- `$((运算式))`
- `$[运算式]` ⭐️ 推荐用法
- `expr m + n`
	- expr 运算符之间要有空格，如果希望将expr的结果赋值给某个变量，使用\`\`
	- ![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720221811.png)
	- 运算符号
		- \\* 乘号前面需要有个转移符号
		- + - / %

### 示例

求出命令行两个参数的和

```bash
#!/bin/bash

echo "first:  $1"
echo "second: $2"
echo "$1+$2=$[$1+$2]"

```

![image.png|center|700](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720223001.png)

# 9-条件判断

### 基本语法

`[ condition ]` (注意：condition前后要有空格)

如果非空，返回true。 可使用`$?`进行验证（0为true，>1为false）

### 判断语句

#### 字符串比较

- `=`： 字符串比较，两个字符串是否相等

#### 两个整数进行比较

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720225015.png)

#### 按照文件权限进行判断

- `-r` ：有读的权限
- `-w` ：有写的权限
- `-x` ： 有执行的权限

#### 按照文件类型进行判断

- `-f`  ：文件存在，并且是一个常规文件
- `-e`  ：文件存在
- `-d`  ：文件存在，并且是一个目录

```bash
#!/bin/bash

# 判断字符串是否相等
if [ "ok" = "ok" ]
then
	echo "ok"
fi

# 判断两个数字的大小
if [ 1 -gt 2 ]
then
	echo "1 > 2"
elif [ 1 -lt 2 ]
then
	echo "1 < 2"
fi

# 判断文件是否存在
if [ -f "test.sh" ]
then
	echo "test.sh is exist"
fi

# 判断权限
if [ -r "test.sh" ]
then
	echo "test.sh is readable"
fi

if [ $0 = "test.sh" ]
then
	echo "test.sh"
fi
```

注意📢：如果要将if和then写在同一行，需要加上`:`

比如：`if [ "ok" = "ok" ]; then`

### 流程控制

#### 第一种：基础语法

```bash
if [ 条件 ]
then 
代码
fi
```

#### 第二种 ：多分支

```bash
if [ 条件 ]
then 
代码
elif [ 条件 ] 
then
代码
fi
```

# 10-case语句

### 基本语法

```bash
case $变量名 in
"值1")
程序1
;;
"值2")
程序2
;;
*)
程序3
;;
esac
```

### 示例

![image.png|center|300](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720234717.png)

# 11-for循环和while循环

### for 基本语法

#### 方式一

```bash
for 变量 in 值1 值2 值3 ......
do
	程序
done
```

#### 方式二

```bash
for((初始值;循环控制条件;变量变换))
do
	程序
done
```

### for循环示例

#### 方式一

区分` $@ `与` $* `的区别         ( [[shell编程#^9f4879|$*]] 和 [[shell编程#^9f4879|$@]]）

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230721000147.png)

#### 方式二

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230721002419.png)

### while循环基本语法

```bash
while [ 条件判断式 ]
do
	程序代码
done
```

# 12-read 读取控制台输入

### 基本语法

- `read [选项] [参数]`
- 选项
	- -p ：指定读取值时的提示符
	- -t  ：指定读取值时等待的时间（s），如果没有在指定的时间内输入，就不再等待了
- 参数
	- 变量：指定读取值的变量名

### 示例

```bash
#！/bin/bash
read -p "请输入NUM1：" NUM1
echo "你输入的NUM1的值为：$NUM1"
```

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230721011009.png)

# 12-函数介绍

### 概述

和其它语言一样，shell编程语言也有函数，其中分为系统函数和自定义函数。

### 系统函数 （语言自带函数）

#### basename 

- 基本语法：`basename [pathname] [suffix]`  或 `basename [string] [suffix]`
- 功能：返回完整路径最后的/部分，**通常用于获取文件名**
- 选项
	- suffix为后缀，如果suffix被制定了，basename会将pathname或string中的suffix去掉
- 应用示例
	- ![image.png|left|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230721014934.png)

#### dirname 

- 基本语法：dirname 文件绝对路径
- 功能：返回完整路径最后的前面部分，**通常用于返回路径部分**
- 应用示例
	- ![image.png|left|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230721015159.png)

![[dirname_and_basename.excalidraw|center|600]]


### 自定义函数

#### 基本语法

```bash
# 函数定义
function  函数名()
{
	代码
	[return int;]
}

# 函数调用

函数名 [值]
```

#### 应用示例

编写一个名为getsum的函数，用于计算两个参数的和

```bash
#!/bin/bash
# 定义函数
function getsum()
{
	SUM=$[$n1+$n2]
	echo $SUM # echo 后面的是函数的返回值
}

read -p "first:   " n1
read -p "second:  " n2

ANS=$(getsum $n1 $n2)

echo "SUM=$ANS"

```

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230721022151.png)


