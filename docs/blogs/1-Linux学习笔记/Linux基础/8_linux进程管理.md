# 08-linux进程管理

## 基本介绍

在linux中，每个执行的程序都称为一个进程，每个进程都分配一个ID号（PID）

每个进程都可能以两种方式存在，**前台与后台**

一般系统的服务都是以后台进程的方式存在的，而且都会常驻在系统中，直到关机才结束。

## 显示系统执行的进程

### ps 指令基本介绍

ps（process show） 是用来显示目前系统中，有哪些正在执行，以及它们的执行情况。

`ps [选项]`

可选选项

1. ps -a 显示当前终端的所有进程信息
2. ps -u 以用户的格式显示进程信息
3. ps -x 显示后台进程运行的参数

**一般来说，进行结合使用**，即为：`ps -aux`

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720111233.png)


4. ps -e: 显示当前所有进程
5. ps -f: 全格式

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720112230.png)

## 终止进程kill和killall

### 基本语法

- `kill [选项] 进程号` 
	- 通过进程号来杀死/终结进程
	- 选项
		-  `-9` 表示强迫进程立即停止
- `killall 进程名称`
	- 通过进程名称来杀死进程

## 查看进程树 pstree

❗️:如果显示没有此命令，在cenos中可以通过`yum install psmisc`来进行安装此命令。

基本语法： `pstree [选项] ` 

可以更加直观的来看进程信息

常用选项
-  -p 显示进程的pid
-  -u 显示进程所属的用户

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720114754.png)


## 服务 service 管理

### service介绍

服务（service）的本质就是进程，但是该进程是运行在后台的。通常会监听某个端口，等待其它程序的请求。

因此，后台程序又被称为守护程序。

### service管理指令

- `service 服务名 [start|stop|restart|status]`
- 在cenos7以后，很多服务不再使用service去管理，而是使用systemctl
- ⭐️service 指令管理的服务在` /etc/init.d` 中进行查看（[[1-Linux目录结构##^f9a9d2|linux中的/etc目录]]）

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720120406.png)

### chkconfig 指令

通过chkconfig命令，可以给服务的各个运行级别设置自启动/关闭

也就是说，服务的自启动和关闭是针对不同的机器运行级别的

显示chkconfig指令支持的服务：`chkconfig --list`

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720121210.png)

使用chkconfig指令

格式：`chkconfig --level 级别 服务名 on/off`

![[3-Linux 实操##指定运行级别]]

❗️：当使用chkconfig重新设置服务在不同级别下的自启动和关闭时，需要重启机器才会生效。

## systemctl 管理指令

### 基本指令

`systemctl [start|stop|restart|status] 服务名`

⭐️：**systemctl** 指令管理的服务在`/usr/lib/systemd/system`中查看

![[1-Linux目录结构##^050dfd]]

### systemctl 设置服务的自启动状态

- 查看服务开机启动状态
	- `systemctl list-unit-files`
- 设置服务开机启动
	- `systemctl enable 服务名`
- 关闭服务开机启动
	- `systemctl disable 服务名`
- 查询某个服务是否是自启动的
	- `systemctl is-enabled 服务名`

### 示例

查看systemctl 支持的firewalld服务

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720164120.png)

发现存在firewalld.service

则可以通过命令进行控制并查看

- `systemctl status firewalld`
- `systemctl stop firewalld`
- `systemctl start firewalld`

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720164421.png)

### 打开或者关闭指定端口

- 打开端口
	- `firewall-cmd --permanent --add-port=端口号/协议`
- 关闭端口
	- `firewall-cmd --permanent --add-port=端口号/协议`
- 重新载入，才能生效
	- `firewall-cmd --reload`
- 查询端口是否开放
	- `firewall-cmd --query-port=端口/协议`

在win中，可以通过`telnet IP地址 端口`来测试某个主机的端口是否打开

## 动态监控进程

### 基本用法

`top [选项]`

"top"和"ps"都是在Unix和类Unix系统中使用的命令，用于查看系统中运行的进程信息。它们之间的区别：
1. 功能
	- top是一个**动态**的进程监视工具，它**实时显示系统中运行的进程信息**，包括CPU使用率、内存使用情况等。
	- ps是一个**静态**的进程查看工具，它**一次性**显示当前系统中的进程信息，不会实时更新。
2. 显示方式
	- top以交互式的方式显示进程信息，可以动态排序和过滤进程，还可以实时更新显示。
	- ps以命令行的方式显示进程信息，需要使用不同的选项来指定要显示的信息。

### 选项说明

- `-d 秒数` ：指定top命令每间隔几秒更新，默认为3s
- `-i` : 使用top不显示任何闲置或者僵死进程
- `-p` : 通过指定监控进程ID来仅仅监控某个进程的状态

### 在top中交互操作

|操作|功能|
|:-:|:-:|
|P|以CPU使用率来进行排序，默认是此选项|
|M|以内存的使用率排序|
|N|以PID来进行排序|
|u|输入用户名，监视特定用户|
|k|输入要结束的进程ID号，终结指定的进程|
|q|退出


## 查看网络状态

### 查看系统网络状态 netstat

- 基本语法 `netstat [选项]`
	- 查看系统网络状态
	- 选项说明
		- `-an` 按照一定顺序排列输出
		- `-p` 显示哪个进程在调用

![image.png|center|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720173231.png)

### 检测主机连接命令 ping

检测连接远程主机是否正常

