# 06-Linux磁盘分区和挂载

## Linux分区

linux采用了一种叫‘载入’的处理方法，他的整个文件系统中包含了一整套的文件和目录，且**将一个分区和一个目录联系起来**。

这时，要**载入的一个分区将使它的存储空间在一个目录下进行获得**。

### 查看所有设备的挂载情况

`lsblk` 或者 `lsblk -f`

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230719233123.png)

## 将磁盘进行挂载的案例

### 增加一块磁盘的总体步骤

以虚拟机为例

1.  为虚拟机添加硬盘
2. 分区
3. 格式化
4. 挂载
5. 设置可以进行自动挂载

### 1-在虚拟机中增加磁盘

添加完硬盘后，需要重启系统，服务器才能识别新的硬盘。

### 2-分区

查看挂载的新的硬盘名称，为**sdb**，由于[[1-Linux目录结构##^93516e|/dev]] 文件目录是进行硬件管理的文件目录，所以，如果要对新的硬盘进行操作，就要对/dev/sdb 文件进行操作。
![[1-Linux目录结构##^93516e]]

分区命令：`fdisk /dev/sdb`  (`fdisk /dev/新加入的硬盘名称`)

开始对/sdb进行分区

- m 显示命令列表
- p 显示磁盘分区 等同于 fdisk -l
- n  新增分区
- d  删除分区
- w 写入并退出

### 3-格式化分区

命令：`mkfs -t ext4 /dev/sdb1` 

将新分好的区进行格式化操作

其中，ext4 是文件类型

### 4-挂载分区

分区必须挂载上才能进行使用

挂载：将一个分区与一个目录联系起来

`mount 设备名称 挂载目录`

`unmount 设备名称或挂载目录`

❗️用命令行进行挂载的分区，服务器重启后，会失效！！！

### 5-进行永久挂载

永久挂载：通过修改`/etc/fstab`实现永久挂载

添加完成后，执行mount -a 即刻生效

![image.png](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720090811.png)

## 磁盘情况查询

### 查询系统整体磁盘使用情况

基本语法：`df -h`

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720091041.png)

### 查询指定目录的磁盘占用情况

- 基本语法：`du [选项] /目录`
	- 查询指定目录的磁盘占用情况，默认为当前目录
	- 选项
		- -s 指定目录占用大小汇总
		- -h 带计量单位，方便human进行阅读
		- -a 含文件
		- --max-depth=1 子目录深度
		- -c 列出明细的同时，增加汇总值


## 磁盘情况-工作实用指令

前提指令：wc

wc  -  print  newline, word, and byte counts for each file

`wc -l` 输出文件的行数

### 统计文件夹下文件的个数

`ls -l | grep "^-" | wc -l`

### 统计文件夹下目录的个数

`ls -l | grep "^d" | wc -l`

### 统计文件夹下文件的个数包括子文件夹里的

`ls -lR | grep "^-" | wc -l`

### 统计文件夹下目录的个数包括子文件夹里的

`ls -lR | grep "^d" | wc -l`

### 以树状显示目录结构

📢：如果没有tree，可以通过 yum install tree 进行安装

