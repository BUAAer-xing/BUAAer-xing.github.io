课程来源：[于仕琪教授：Makefile 20分钟入门，简简单单，展示如何使用Makefile管理和编译C++代码](https://www.bilibili.com/video/BV188411L7d2/)

##   操作环境

Macos+Vscode

## 前提准备

### 新建文件夹 

```shell
mkdir learn_makefile
```
### 新建三个cpp文件和一个头文件

```cpp
// mian.cpp
#include <iostream>
#include "functions.h"
using namespace std;
int main()
{
    printhello();
    cout << "This is main:" << endl;
    cout << "The factorial of 5 is: " << factorial(5) << endl;
    return 0;
}
```

```cpp
// factorial.cpp
#include "functions.h"
int factorial(int n)
{
    if(n == 1)
        return 1;
    else
        return n * factorial(n-1);
}
```

```cpp
// printhello.cpp
#include <iostream>
#include "functions.h"
using namespace std;
void printhello()
{
    cout << "Hello world" << endl;
}
```

```cpp
// function.h
#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_
void printhello();
int factorial(int n);
#endif
```

## 不使用makefile进行编译链接操作

进入learn_makefile文件夹下进行操作

```shell
g++ main.cpp factorial.cpp printhello.cpp -o main
./main
```

![image.png|left|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20231106195030.png)

## 创建Makefile文件 （版本一）

```cmake
# Makefile文件

# VERSION 1
# hello为生成的可执行文件，依赖于后面的三个.cpp文件
# g++前面加一个TAB的空格
hello: main.cpp printhello.cpp factorial.cpp
	g++ -o hello main.cpp printhello.cpp factorial.cpp
```

在learn_makefile文件夹下，执行以下命令：<font color="#c00000">makefile文件的使用方法</font>

```shell
make
./hello
```

![image.png|left|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20231106200629.png)

## 创建Makefile文件（版本二）

```cmake
# VERSION 2
CXX = g++
TARGET = hello
OBJ = main.o printhello.o factorial.o
# make时执行g++ 先找TARGET，TARGET不存在找OBJ，OBJ不存在，编译三个.cpp文件生成.o文件
# 然后再编译OBJ文件，生成可执行文件hello
$(TARGET): $(OBJ)
	$(CXX) -o $(TARGET) $(OBJ)
# main.o这样来的，编译main.cpp生成
main.o: main.cpp
	$(CXX) -c main.cpp
printhello.o: printhello.cpp
	$(CXX) -c printhello.cpp
factorial.o: factorial.cpp
	$(CXX) -c factorial.cpp
```

## 创建Makefile文件（版本三）

```cmake
# VERSION 3
CXX = g++
TARGET = hello
OBJ = main.o printhello.o factorial.o
 
# 编译选项，显示所有的warning
CXXLAGS = -c -Wall
 
# $@表示的就是冒号前面的TARGET $^表示的是冒号后OBJ的全部.o依赖文件
$(TARGET): $(OBJ)
	$(CXX) -o $@ $^
 
# $<表示指向%.cpp依赖的第一个，但是这里依赖只有一个
# $@表示指向%.o
%.o: %.cpp
	$(CXX) $(CXXLAGS) $< -o $@
 
# 为了防止文件夹中存在一个文件叫clean
.PHONY: clean
 
# -f表示强制删除，此处表示删除所有的.o文件和TARGET文件
clean:
	rm -f *.o $(TARGET)
```

## 创建Makefile文件（版本四）

```cmake
# VERSION 4
CXX = g++
TARGET = hello
# 所有当前目录的.cpp文件都放在SRC里面
SRC = $(wildcard *.cpp)
# 把SRC里面的.cpp文件替换为.o文件
OBJ = $(patsubst %.cpp, %.o,$(SRC))
 
CXXLAGS = -c -Wall
 
$(TARGET): $(OBJ)
	$(CXX) -o $@ $^
 
%.o: %.cpp
	$(CXX) $(CXXLAGS) $< -o $@
 
.PHONY: clean
clean:
	rm -f *.o $(TARGET)
```

