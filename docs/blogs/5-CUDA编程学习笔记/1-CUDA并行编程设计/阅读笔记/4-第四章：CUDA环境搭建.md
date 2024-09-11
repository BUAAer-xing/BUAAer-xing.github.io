## 4.1-安装显卡驱动和CUDA

略

## 4.2-编译模型

在cuda中，文件扩展名决定编译时使用nvcc还是主机编译器。这一点和hipcc是不同的。

同时，这里需要注意的是，CUDA的编译也支持代码托管，通过定义环境变量`CUDA_DEVCODE_CACHE`指向一个目录，该目录将指示运行时把编译的二进制文件保存以备后用。通常情况下，每次编译需要把PTX代码根据未知GPU的类型进行转换。采用代码托管，这种启动延时就可以避免了。

## 4.3-错误处理

几乎所有的CUDA函数调用，都会返回类型为`cudaError_t`的整数值。cudaSuccess以外的任何值都将代表致命错误。

检查CUDA函数是否正确执行的宏定义可以为：
```cpp
#define CUDA_CALL(x) {                                           \
    const cudaError_t a = (x);                                   \
    if(a != cudaSuccess) {                                       \
        printf("\nCUDA Error: %s (err_num=%d)\n",                \
               cudaGetErrorString(a), a);                        \
        cudaDeviceReset();                                       \
        assert(0);                                               \
    }                                                            \
}
```


检查CUDA内核函数是否正确执行：

```cpp
void wait_exit() {
    printf("Press Enter to exit...\n");
    getchar();  // 等待用户按下 Enter 键
}
__host__ void cuda_error_check(const char * prefix, const char * postfix)
{
    // 检查最后一个CUDA错误，如果不是成功状态，则进行错误处理
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        // 打印错误信息，包含前缀、CUDA错误信息和后缀
        printf("\n%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
        // 重置CUDA设备
        cudaDeviceReset();
        // 等待用户按键（假设 wait_exit() 是等待用户操作的函数）
        wait_exit();
        // 退出程序，返回非0表示异常退出
        exit(1);
    }
}
```

注意，**内核函数调用与CPU代码是异步执行**的，所以上述方法并非万无一失。异步执行表示当调用cudaPeekAtLastError时，GPU代码正在后台运行。如果此时没有检测到错误，则不会输出错误，函数继续执行下面的代码行。通常情况下，下一条代码行是 从GPU内存将数据复制到CPU内存的操作。内核函数的错误可能会导致随后的应用程序接口调用失败，一般应用程序接口的调用是紧跟内核函数调用的。**针对全部的应用程序接口调 用，均使用CUDA CALL包裹**，这种错误就会被标记出来。 

也可以**强制在内核函数完成后再进行错误检查**，只需在`cudaPeekAtLastError`调用之前加入`cudaDeviceSynchronize`调用即可。然而，这一强制行为只能在调试版程序或者想让CPU在GPU占满时处于闲置状态时使用。<font color='red'><b>这种同步操作适合进行调试，但会影响性能</b></font>。所以，如果这些调用只是为了调试，请务必在产品代码中删去。

