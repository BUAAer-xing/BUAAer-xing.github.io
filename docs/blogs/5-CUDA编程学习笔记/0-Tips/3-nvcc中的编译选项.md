`nvcc` 是 NVIDIA CUDA 编译器，它支持许多编译选项用于控制编译过程。以下是一些常用的 `nvcc` 编译选项：
### 1. **编译输出控制**
- `-o <filename>`：指定输出文件的名称。例如 `-o my_program`。
- `-c`：只编译，不链接。生成 `.o` 文件（目标文件），常用于分阶段编译。
- `-keep`：保留编译过程中产生的中间文件（如 `.ptx`, `.cubin` 等），便于调试和分析。
### 2. **编译模式**
- `-g`：生成调试信息，用于调试程序。此选项允许使用 CUDA 调试工具（如 `cuda-gdb`）来调试代码。
- `-G`：生成调试级别的代码，包括设备代码的调试信息，通常用于调试设备上的 CUDA 内核。
- ⭐️`-O<n>`：优化级别，`-O0` 表示关闭优化，`-O3` 表示最高优化级别。默认为 `-O3`。
### 3. **架构选项**
- ⭐️`-arch=<compute capability>`：指定目标 GPU 的计算能力（Compute Capability）。例如：
  - `-arch=sm_35`：为 Compute Capability 3.5 生成代码（例如用于 Kepler 架构的 GPU）。
  - `-arch=sm_60`：为 Compute Capability 6.0 生成代码（例如用于 Pascal 架构的 GPU）。
  - `-arch` <font color='red'><b>是最常用的选项之一，用于确保生成的代码能在目标 GPU 上运行</b></font>。
- `-code=<compute capability>`：生成特定目标架构的代码。与 `-arch` 一起使用时，可以确保生成不同架构的兼容代码。例如：
  ```bash
  nvcc -arch=compute_60 -code=sm_60 my_program.cu
  ```
### 4. **PTX 和 SASS 生成**
- `-ptx`：生成 PTX 中间代码，而不是直接生成二进制文件。PTX 是 CUDA 的中间表示形式，便于跨架构执行。
- `-cubin`：生成 `.cubin` 文件，这是一种包含设备代码的二进制文件格式。
### 5. **调试和性能分析**
- `-lineinfo`：在设备代码中包含源代码的行号信息。这对性能分析工具（如 `nvprof` 或 `Nsight Systems`）非常有用。
- `-Xptxas="-v"`：编译内核时输出寄存器使用等信息，便于优化内核性能。例如，告诉你每个内核使用的寄存器数。
### 6. **警告和错误处理**
- `-Werror`：将所有警告视为错误，阻止生成目标文件。用于强制代码质量。
- `-w`：禁止警告输出。
### 7. **设备内存模型**
- `-Xcompiler`：传递编译器选项给宿主编译器（如 GCC 或 MSVC）。例如 `-Xcompiler -fPIC` 用于生成位置无关代码。
- `-Xlinker`：将选项传递给链接器。例如 `-Xlinker -rpath=<path>`。
### 8. **静态和动态库**
- `--shared`：生成共享库（`.so` 文件）。例如，编写 CUDA 动态库时使用。
- `-L<path>`：指定库路径，例如 `-L/usr/local/cuda/lib64`。
- `-l<library>`：链接库，例如 `-lcudart` 链接 CUDA 运行时库。
### 9. **主机代码编译**
- `--compiler-bindir <path>`：指定主机代码编译器的路径，常用于在特定环境中使用不同的宿主编译器。
- `--std=<version>`：指定 C++ 标准版本，例如 `--std=c++11` 或 `--std=c++14`。
### 10. **其他有用选项**
- `--dryrun`：显示编译过程中将要执行的命令，但不实际执行它们。这对调试编译选项很有用。
- `--device-debug`：生成设备代码的调试信息。
- `--verbose`：显示详细的编译信息，便于调试编译过程。