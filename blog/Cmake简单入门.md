---
authors: [BUAAer-xing]
---

# Cmake简单入门

<center><font color="#c00000" size='6'>CMake可以生成不同平台下的Makefile，有了CMake不用再写复杂的Makefile</font></center>

视频教程：[CMake 6分钟入门，不用再写复杂的Makefile](https://www.bilibili.com/video/BV1bg411p7oS)

<iframe src="https://player.bilibili.com/player.html?bvid=BV1bg411p7oS&autoplay=0" scrolling="no" frameborder="no" framespacing="0" allowfullscreen="true" autoplay="no" width='100%' height='500px'> </iframe>


## 先前知识

[[Makefile简单入门]]

## Cmake特性

CMake是一个用于管理C/C++项目的**跨平台构建工具**。

1. 跨平台：CMake是跨平台的，**可以在多种操作系统上使用**，包括Windows、Linux、macOS等。**它可以生成适用于各种编译器和构建系统的配置文件。**

2. 自动生成构建系统：CMake的一个主要特点是**它可以自动生成适用于不同构建工具的配置文件**，如Makefiles、Visual Studio项目文件、Xcode项目文件等。这使得开发人员能够轻松地在不同平台上构建和编译项目。

## Cmake简单示例

```cmake
cmake_minimum_required(VERSION 3.10)

project(hello)

add_executable(hello main.cpp factorial.cpp printhello.cpp)
```

## 将过程文件放在当前文件夹下

在当前文件夹learn_makefile执行以下命令：

```shell
cmake .
make
./hello
```

![image.png|left|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20231106204637.png)

## 将过程文件单独放置在一个文件夹下

新建build文件夹

进入build文件夹，在build文件夹下，执行以下命令：

```shell
cmake .. # 去上一层文件中去找CMakeLists.txt文件来进行构建

make # 此时所有的一切都是在build的目录下生成的

./hello
```

![image.png|left|1000](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20231106205730.png)

![image.png|left|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20231106205814.png)




