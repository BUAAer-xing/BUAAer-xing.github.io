错误有两种分类：一类是编译错误，一类是运行时错误。
- 编译错误：在编译过程中就可以被编译器捕捉，称为编译错误。
- 运行时错误：有些错误在编译期间没有被发现，但在运行的时候出现，被称为运行时刻的错误。

这章主要介绍的是如何检测运行时刻的错误，包括使用一个检查CUDA运行时API函数返回值的宏函数以及使用CUDA_MEMCHECK工具。

## 4.1-一个检测CUDA运行时错误的宏函数

```cpp
#define CHECK_CUDA(call)                                                    \
do                                                                          \
{                                                                           \
    const cudaError_t error_code = call;                                    \
    if (error_code != cudaSuccess)                                          \
    {                                                                       \
        printf("CUDA Error:\n");                                            \
        printf("     - File: %s\n", __FILE__);                              \
        printf("     - Line: %d\n", __LINE__);                              \
        printf("     - Error code: %d\n", error_code);                      \
        printf("     - Error text: %s\n", cudaGetErrorString(error_code));  \
        cudaDeviceReset();                                                  \
        exit(1);                                                            \
    }                                                                       \
} while (0)
```

这里需要解释一下，为什么要使用do...while(0)来对检测块进行包裹：

这个 `do { ... } while (0)` 结构在宏定义中使用是一个常见的技巧，目的是为了确保宏在被调用时的行为与普通的代码块一致，并避免潜在的语法问题。具体原因如下：
1. **保持语法一致性**: 如果不使用 `do...while(0)`，而只是用花括号 `{ ... }` 包裹宏内容，在调用宏时如果宏调用后面加了分号（如 `CHECK(call);`），会引发语法错误或意外行为。通过使用 `do { ... } while (0)`，宏在调用时就像普通的函数调用一样，需要一个分号结束，语法上更加统一。
2. **避免宏展开导致的语法错误**: 假设不使用 `do...while(0)`，直接写成：
   ```cpp
   #define CHECK(call) { /* error checking code */ }
   ```
   那么如果在 `if` 或者 `else` 中使用这个宏，比如：
   ```cpp
   if (condition)
       CHECK(call);
   else
       other_statement;
   ```
   这段代码会产生语法错误，因为在 `CHECK(call)` 宏展开后，`if` 语句只会作用在第一行，而不会包括整个宏的代码块，这样 `else` 就会找不到与之对应的 `if`。使用 `do { ... } while(0)` 可以避免这种问题，使得整个宏在逻辑上被视为一个完整的语句块。
3. **零次循环的安全性**: `do...while(0)` 保证了宏中的代码只执行一次。因为 `while(0)` 条件恒为 `false`，循环体只会执行一次，达到与普通代码块相同的效果。


### 4.1.1-检查运行时的API函数

直接将宏定义套用在cuda的调用函数外层即可。
```cpp
CHECK(cudaFree(0));
```


### 4.1.2-检查核函数

用上述方法不能捕捉调用核函数的相关错误，因为核函数不返回任何值（核函数必须使用void进行修饰）。有一个方法可以捕捉调用核函数可能发生的错误：

在调用核函数之后加上如下两条语句：
```cpp
CHECK(cudaGetLastError());
CHECK(cudaDeviceSynchronize());
```

其中，第一条语句的作用是捕捉第二个语句之前的最后一个错误，第二条语句的作用是同步主机与设备。之所以要同步主机与设备，是因为<font color='red'><b>核函数的调用是异步</b></font>的， 即主机发出调用核函数的命令后会立即执行后面的语句，不会等待核函数执行完毕。

需要注意的是，上述同步函数是比较耗时的，如果在程序的较内层循环调用的话，很可能会严重降低程序的性能。所以，**一般不在程序的较内层循环调用上述同步函数**。**只要在核函数的调用之后还有对其他任何能返回错误值的API 函数进行同步调用，都能够触发主机与设备的同步并捕捉到核函数调用中可能发生的错误**。

❗️❗️：数据传输函数也会起到了一种隐式的同步主机和设备的作用。在一般情况下，如果想要获得精确的出错位置，还是需要显式的同步。这里的显式同步有两种方式：
- 调用`cudaDeviceSynchronize()`函数。
- 临时将环境变量`CUDA_LAUNCH_BLOCKING`的值设置为1。
	- 这样设置之后，所有核函数的调用都不将再是异步的，而是同步的。也就是说，主机调用一个核函数之后，必须等待核函数执行完毕，才能继续向下执行。

## 4.2 用CUDA-MEMCHECK检查内存错误

CUDA 提供了名为 CUDA-MEMCHECK 的工具集，具体包括 memcheck、racecheck、initcheck、synccheck 共 4 个工具。它们可由可执行文件 cuda-memcheck 调用：
```
$ cuda-memcheck --tool memcheck [options] app_name [options]  
$ cuda-memcheck --tool racecheck [options] app_name [options]  
$ cuda-memcheck --tool initcheck [options] app_name [options]  
$ cuda-memcheck --tool synccheck [options] app_name [options]  
```
对于 memcheck 工具, 可以简化为：
```
$ cuda-memcheck [options] app_name [options]  
```

