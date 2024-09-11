## 10.1-简介

为了利用GPU加速任务，直接使用CUDA编写程序并不是唯一可选的过程。有三种常见的开发CUDA应用程序的方式：
- 使用函数库
- 基于指令的编程
- 直接编写CUDA内核

## 10.2-函数库

如果是开发中，应该尽可能的使用函数库，它们基本是由特定领域的专家开发的，它们是可靠而高效的。

一些常见的、免费的函数库如下：
- `Thrust`：一个C++ STL 实现的函数库
- `CuBLAS`：BLAS（基本线性代数）函数库的GPU版本
- `cuFFT`：GPU加速的快速傅立叶变换函数库
- `cuSparse`：稀疏矩阵数据的线性代数库

### 10.2.1-函数库通用规范

作为一般原则，NVIDIA提供的函数库，不对调用者进行内存管理，而是<font color='red'><b>希望调用者提供指向设备中被分配的内存区域的指针</b></font>（注意：是设备中的指针，说明已经是分配并进行数据传输完成后的指针）。 这就可以是的很多设备上的函数可以一个接一个地执行而无须在两个函数调用间使用不必要的设备/主机进行传输操作。

由于函数库不执行内存操作，因此，分配和释放内存变成了调用者的职责。这甚至扩大到为函数库用到的暂存空间或缓冲区提供内存。

虽然对程序员来说这可能是一笔开销，但<font color='red'><b>实际上这是一种很好的设计原则并且在设计函数库时也应该遵循</b></font>。内存分配是代价很高的操作。资源是很有限的。**让函数库在后台持续不断地分配、释放内存，不如在启动时执行一次分配操作，然后在程序退出时再执行一次释放操作**。

### 10.2.2-NPP函数库

略

### 10.2.3-Thrust函数库

#### C++模版系统

一个相同功能的函数体，需要不同的变量传入，C中无法使用相同的函数名来表示，因此，在C++中试图使用<font color='red'><b>函数重载</b></font>的方式来解决这个问题。这就使得多个函数能够使用相同的名字，**根据传递参数的类型，系统会调用合适的函数**。然而，即使现在可以使用相同的名字调用各个函数，但函数库的提供者仍然需要编写不同变量类型对应的具体函数体来解决int8、int16等不同参数类型。<font color='red'><b>C++模版系统</b></font>解决了这个问题，通过模版系统，可以得到一个<font color='red'><b>通用的函数</b></font>，并且只有参数的类型发生改变。

C++中的模板机制是一种用于实现泛型编程的工具。它允许你编写通用的代码，从而可以在不指定具体数据类型的情况下进行函数或类的定义。模板主要有两种形式：**函数模板**和**类模板**。

##### 函数模板
函数模板使得函数可以适用于不同类型的参数。可以通过模板定义一个函数，随后这个函数可以被用于不同的数据类型，而不需要为每种数据类型单独定义函数。
```cpp
#include <iostream>
using namespace std;

template <typename T>
T add(T a, T b) {
    return a + b;
}

int main() {
    cout << add(3, 4) << endl;          // 使用int类型
    cout << add(2.5, 3.7) << endl;      // 使用double类型
    return 0;
}
```
上面的 `add` 函数使用了模板参数 `T`，它可以是任意数据类型。当调用该函数时，编译器会根据传递的参数类型生成相应的具体版本。

##### 类模板
类模板允许创建通用的类定义，使得类可以处理不同的数据类型。与函数模板类似，类模板也使用模板参数来适应不同的类型。
```cpp
#include <iostream>
using namespace std;
template <typename T>
class Box {
   private:
    T value;
   public:
    Box(T v) : value(v) {}
    T getValue() {
        return value;
    }
};
int main() {
    Box<int> intBox(100);          // 使用int类型
    Box<double> doubleBox(99.99);  // 使用double类型
    cout << intBox.getValue() << endl;
    cout << doubleBox.getValue() << endl;
    return 0;
}
```
上面的 `Box` 类是一个类模板，它可以存储任意类型的 `value`。在创建对象时，你可以通过提供具体的数据类型（如 `int` 或 `double`）来实例化模板类。

##### 模板的其他特性
- **模板特化（Template Specialization）**：<font color='red'><b>为特定的类型定义模板的特化版本，提供不同的实现</b></font>。
- **非类型模板参数**：模板参数不仅可以是类型，还可以是常量，如整型常量或指针。
```cpp
#include <iostream>
using namespace std;
template <typename T>
void print(T value) {
    cout << "Generic Template: " << value << endl;
}
// 模板特化
template <>
void print<int>(int value) {
    cout << "Specialized Template for int: " << value << endl;
}
int main() {
    print(3.14);     // 调用泛型模板
    print(100);      // 调用特化的int模板
    return 0;
}
```
**NVCC编译器实际上是C++的前端而不是C的前端**。**Thrust函数库支持很多STL容器**，因此这就使其能支持大规模的并行处理器。数组（STL所说的向量）在Thrust中被很好地支持。然而，并不是所有的容器都适用。

#### namespace 命名空间

命名空间有点类似于为函数指定一个函数库前缀，是的编译器可以辨别应该在哪个函数库中寻找函数。<font color='red'><b>C++命名空间实际上是一个类选择器</b></font>。假如，使用了不同的命名空间，那么同一个函数可以有不同的调用，比如：
```cpp
ClassNameA::my_func();
ClassNameB::my_func();
```
它的功能主要是：<font color='red'><b>防止同名的变量、函数或类之间产生冲突</b></font>。通过命名空间，可以将代码组织在特定的作用域内，从而提高代码的可维护性和可读性。

#### 仿函数

仿函数（functor），也称为函数对象（function object）。仿函数指的是一个重载了函数调用运算符 `operator()` 的类或结构体实例。这使得该类的对象可以像函数一样被调用，从而具备函数的行为。仿函数常用于需要函数回调、算法参数化等场景，特别是在标准模板库（STL）中，它被广泛应用于标准算法中。

##### 仿函数的定义

为了创建仿函数，需要定义一个<font color='red'><b>类或结构体并重载 `operator()` 运算符</b></font>。这样，该类的实例就可以像普通函数一样调用。
```cpp
#include <iostream>
using namespace std;
// 定义一个仿函数类
class MyFunctor {
public:
    // 重载函数调用运算符
    void operator()(int x) {
        cout << "仿函数被调用，参数为: " << x << endl;
    }
};
int main() {
    MyFunctor f;  // 创建仿函数对象
    f(10);        // 像调用函数一样调用仿函数
    return 0;
}
```
在上面的例子中，`MyFunctor` 类重载了 `operator()`，因此 `f(10)` 会调用该运算符，使对象 `f` 像函数一样工作，输出相应的内容。

##### 带状态的仿函数

与普通函数不同的是，仿函数可以保存状态。通过在类中定义成员变量，仿函数可以在调用时携带和操作这些状态。 
```cpp
#include <iostream>
using namespace std;
// 带状态的仿函数
class Adder {
private:
    int value;
public:
    Adder(int init) : value(init) {}
    // 重载函数调用运算符
    int operator()(int x) {
        return value + x;
    }
};
int main() {
    Adder add5(5);  // 创建一个初始值为5的仿函数对象
    cout << add5(10) << endl;  // 输出15
    Adder add10(10);  // 创建一个初始值为10的仿函数对象
    cout << add10(10) << endl;  // 输出20
    return 0;
}
```
在这个例子中，`Adder` 类存储了一个 `value` 值，当仿函数对象被调用时，它将该 `value` 与传入的参数相加。这展示了仿函数保存状态的能力，允许你在函数调用时使用额外的信息。

#### 使用Thrust函数

Thrust函数库分别为主机和设备提供了向量类型并且分别驻留在主机和设备的<font color='red'><b>全局内存</b></font>中。注意：如果向量在设备上，那么对于每个这样的访问，Thrust通过PCI-E总线在后台执行单独的传输。

thrust:: 这部分指定了类命名空间。

C++中的对象也有一个析构函数，当对象超出域的范围时，该函数会调用。该函数负责回收所有的析构函数分配的或对象在运行时使用的资源。因此，对于Thrust向量，没有必要调用free或cudafree。


```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>

using namespace std;

int main() {
    // 创建一个主机向量
    thrust::host_vector<int> h_vec(5);
    
    // 初始化主机向量中的数据
    h_vec[0] = 4;
    h_vec[1] = 2;
    h_vec[2] = 3;
    h_vec[3] = 1;
    h_vec[4] = 5;
    
    // 将数据从主机向量复制到设备向量
    thrust::device_vector<int> d_vec = h_vec;
    
    // 对设备向量中的数据进行排序
    thrust::sort(d_vec.begin(), d_vec.end());
    
    // 将结果复制回主机向量
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    
    // 输出排序后的结果
    cout << "Sorted vector: ";
    for(int i = 0; i < h_vec.size(); i++) {
        cout << h_vec[i] << " ";
    }
    cout << endl;
    
    return 0;
}
```

#### 关于Thrust的问题

Thrust总是使用默认的流，并且无法像NPP函数库那样改变它。因为没有流参数可以传递，也没有函数可以设置当前选择的流。因此，任何从Thrust调用的函数会返回一个值，任何复制回主机的操作都会引发一个隐式的同步。

#### 多CPU/GPU的思考

像常规的多GPU编程一样，Thrust支持单线程/多GPU和多线程/多GPU模型。**它并不会隐式地利用多GPU，而是让程序员选择创建多线程或是在合适的地方使用 <font color='red'><b>cudaSetDevice</b></font>调用以选择可以正确工作的设备**。

### 10.2.4 CuRAND 函数库

**cuRAND** 是 NVIDIA CUDA 提供的一个用于 GPU 上生成伪随机数的库。它提供了**高效的随机数生成方法**，可以大规模并行化，因此非常适合在 GPU 上进行蒙特卡洛仿真、随机采样等应用。cuRAND 可以生成多种类型的随机数，如均匀分布、高斯分布等。

## 10.3 CUDA运行SDK

**CUDA SDK** 是 NVIDIA 提供的开发工具包，专门用于在 GPU 上进行并行计算开发。SDK 是 **Software Development Kit** 的缩写，即软件开发工具包，CUDA SDK 包含了编写、调试、优化和运行 CUDA 应用程序所需的各种工具、库和示例代码。

CUDA SDK 的主要内容包括：
1. **CUDA 编译器 (nvcc)**：用于将 CUDA C/C++ 代码编译成可在 NVIDIA GPU 上运行的二进制文件。CUDA 程序通常是 C/C++ 程序中嵌入的内核代码，这些内核代码由 nvcc 编译器进行特殊处理。
2. **CUDA 库**：CUDA SDK 提供了一系列库，这些库可以显著简化和加速常见的并行计算任务。例如：
   - **cuBLAS**：用于高性能的线性代数运算。
   - **cuFFT**：用于快速傅里叶变换。
   - **cuRAND**：用于随机数生成。
   - **Thrust**：用于并行的模板库，类似于 C++ 的 STL。
   - **cuDNN**：用于深度学习的优化库。
3. **CUDA 工具**：
   - **Nsight 系列工具**：用于 CUDA 程序的调试和性能分析，帮助开发者发现性能瓶颈和优化代码。
   - **CUDA Profiler**：用于分析 CUDA 程序的性能，提供 GPU 内存使用、内核执行时间等详细数据。
4. **示例代码**：CUDA SDK 包含了多个示例程序，演示如何使用 CUDA 开发各种应用。这些示例涵盖了矩阵乘法、向量加法、图像处理、蒙特卡洛方法等常见并行计算任务。
5. **头文件和库文件**：用于开发与 CUDA API 交互的应用。CUDA SDK 包含了开发 GPU 计算应用所需的所有头文件和库，例如 `cuda.h` 和 `cudart.lib`。

### CUDA SDK 的用途：
- **GPU 加速计算**：使用 CUDA SDK，可以让开发者通过简单的编程接口，在 GPU 上执行计算密集型任务，以获得比传统 CPU 快得多的性能提升。
- **并行编程**：CUDA SDK 提供了一种编写并行代码的标准方法，使得程序可以在多个 GPU 核心上并行执行，从而加速大量计算任务。
- **科学计算和机器学习**：CUDA SDK 已成为许多科学计算、机器学习（尤其是深度学习）框架的基础，例如 TensorFlow、PyTorch 都依赖 CUDA 进行 GPU 加速。

Tips：如果使用了结构体，则需要考虑它的**合并影响**并且（至少）使用lign指令。一个更好的解决方案是<font color='red'><b>创建数组结构体而不是结构体数组</b></font>。例如，使用分开的红、绿和蓝(RGB)的色彩平面而不是交错的RGB值。

## 10.4 基于指令的编程

**OpenACC** 是一种用于并行编程的编译器指令（或称编译器注释）规范，旨在简化在异构计算环境（如 CPU 和 GPU）中编写并行代码。它通过高层次的编译器指令，允许程序员快速将已有的串行代码加速，尤其是对现有的科学和工程计算代码进行并行化，而无需深入到底层的 CUDA 或 OpenCL 等编程模型。

一些关键的指令：
- `#pragma acc parallel`：告诉编译器该代码块可以并行执行。
- `#pragma acc kernels`：标记代码中的一组内核，这些内核应该被并行化并可能在 GPU 上运行。
- `#pragma acc loop`：用于并行化循环（常见于数值计算中的密集循环）。
- `#pragma acc data`：管理数据在主机（CPU）和设备（GPU）之间的传输。

例子：
```cpp
#include <stdio.h>
#define N 1000
int main() {
    int a[N], b[N], c[N];
    // 初始化数组 a 和 b
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    // 使用 OpenACC 并行化数组加法
    #pragma acc parallel loop
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    // 输出结果
    for (int i = 0; i < 10; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");
    return 0;
}
```

编译方法：`nvc -acc -o example_openacc example_openacc.c`

## 10.5 编写自己的内核

无论是自己编写内核还是抽象为别人的问题，这些基本准则（合并内存访问、充分利用硬件、避免资源竞争、了解硬件局限、数据局部性原理）都是十分重要的！！！






