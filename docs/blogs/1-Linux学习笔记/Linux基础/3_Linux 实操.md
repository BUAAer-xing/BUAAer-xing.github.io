# 03-Linux实操



## 开关机、重启、用户登陆注销

### 关机&重启
- 关机&重启之前的操作
	- `sync` 把内存的数据同步到磁盘上
- 关机指令
	- `shutdown -h now`   立刻关机
	- `shutdown -h 1 `       1分钟后关机
	- `halt `关机
- 重启指令
	- `shutdown -r now`    立刻重新启动计算机
	- `reboot` 重启

### 用户登陆和注销

注销只能在shell环境下进行使用

- 登陆
	- 登陆普通用户 `su - 用户名`
	- 登陆root用户 `sudo su` 或者 `su -root`
- 注销
	- 退出当前用户 `logout` 

## 用户管理

Linux系统是一个多用户多任务的操作系统，任何一个要使用系统资源的用户，都必须首先想系统管理员申请一个账号，然后通过这个账号再进入系统。

### 添加用户

```bash
useradd 用户名
```

添加一个系统操作用户，当用户创建成功后，会自动在home目录下创建和用户同名的目录

```shell
useradd -d 指定目录 新的用户名
```

给新创建的用户指定存储路径，而不是存储在`/home`目录下(-d d就是directory 目录的意思)

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709121001.png)

### 修改用户密码

```shell
passwd 用户名
## 如果不加用户名，则默认是修改当前用户的密码
```

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709154930.png)

### 删除用户

现在的系统用户列表如下所示：
![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709155258.png)
- 删除用户但要保留用户文件

```
userdel 用户名 
```

- 删除用户同时删除用户文件

```
userdel -r 用户名
```

###  查询用户信息

```
id 用户名
```

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709160157.png)

### 切换用户

如果当前用户的权限不够，可以通过 su - 用户名，切换到高权限用户，比如root用户

```
su - 切换用户名
```

❗️❗️注意： 
- 从权限高的用户切换到权限低的用户，不需要输入密码，如果从权限低的切换到权限高的用户，则需要输入切换的用户密码
- 当需要返回到原来的用户时，可以使用`exit/logout`指令，进行退出

### 查看当前用户

```
whoami 或者 who am I
```

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709161559.png)

- `who am i` 这个指令，显示的是第一次登陆系统的用户，如果通过`su`指令进行了用户的切换，则仍然会是第一次登陆系统的用户。

- `whoami` 这个指令则会显示目前正在进行操作的用户

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709162237.png)

### 用户组的添加和删除

^ec4c90

 用户组的作用在于：系统可以对有共性【权限】的多用户进行统一的管理

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709163453.png)

- 新增组
	- `groupadd 组名`
- 删除组
	- `groupdel 组名`
- 增加用户时直接加上组
	- `useradd -g 用户组 用户名`

❗️❗️注意： 如果在增加用户时，没有指定组，则会在创建用户时，同时创建一个名为用户名的组。

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709165110.png)

- 修改用户的组
	- `usermod -g 用户组 用户名`

### 用户和组相关文件

![[1-Linux目录结构##^4af414]]

- **/etc/passwd 文件**
	- 用户user的配置文件，记录用户的各种信息
	- 每行信息的含义
	- 用户名:口令:用户标识号:组标识号:注释性描述:主目录:登陆shell【[[shell编程|shell介绍]]】
	- ![image.png](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709170317.png)
- **/etc/shadow文件**
	- 口令的配置文件
	- 每行的含义
	- 登录名:加密口令:最后一次修改时间:最小时间间隔:最大时间间隔:警告时间:不活动时间:失效时间:标志

- **/etc/group文件**
	- 组的配置文件，记录Linux包含组的信息
	- 每行的含义
	- 组名:口令:组标识号:组内用户列表

❗️❗️注意：口令一般是不可见的，表示形式为x或为空。

## 实用指令

###  指定运行级别

|运行级别|级别含义|
|:-:|:-:|
|0|关机|
|1|单用户状态（找回丢失的密码）|
|2|多用户状态没有网络服务|
|3|多用户状态有网络服务|
|4|系统未使用保留给用户|
|5|图形界面|
|6|系统重启|

### init 命令

通过init命令来切换不同的运行级别

比如：init [0123456] 然后关机，再启动即可进行切换

比如 init 0 表示关机； init 6 表示系统重启

### 帮助指令

- **man 获得帮助信息**
	-  `man [命令或配置文件]`
	- 比如：查看 ls 命令的帮助信息 ：` man ls `
	- 选项可以进行组合使用，比如组合 ls -a 和 ls -l 为： ls -al 或 ls -la 都可以
	- ![image.png|left|700](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230713162605.png)
- **help 指令**
	- 语法： `help 命令`
	- 获得shell内置命令的帮助信息
	- 查看cd命令的帮助信息
		- ![image.png|left|800](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230713162946.png)
### 文件目录类

- **pwd 指令**
	-  显示当前工作目录的绝对路径
	- ![image.png](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230713163127.png)
- **ls指令**
	- `ls [选项] [目录或者文件]`
	- 常用选项
		- -a：显示当前目录所有的文件和目录，包括**隐藏的**
		- -l ：以列表的方式进行显示信息
- **cd 指令**
	- 切换到指定目录
	- `cd [参数]`
		- cd ～ ：回到自己的家目录中
		- cd ..   ： 回到上一级目录
- **mkdir 指令**
	- 创建目录
	- `mkdir [选项] 要创建的目录`
	- 常用选项 
		- -p ： 创建多级目录
	-   ![image.png](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230713164518.png)
- **rmdir 指令**
	- 删除**空目录**
	- `rmdir [选项] 要删除的空目录`
	- ❗️删除的是空目录，如果目录下有内容时，则无法进行删除
- 删除非空目录
	- `rm -rf 要删除的目录`
		- -r ( -R, --recursive)：递归地删除目录及其内容
		- -f ( --force )： 强制删除，忽略是否有文件和相关参数
- **touch 指令**
	- 创建空文件
	- `touch 文件名`
- **cp 指令**
	- cp指令拷贝文件到指定目录
	- `cp [选项] source dest`
		- source 拷贝的源文件名
		- dest 拷贝到的目的位置
	- 常用选项
		- -r ： 递归复制整个文件夹
	- 注意❗️：
		- 强制覆盖不提示的方法，在cp前面加上`\`
		- 例如：`\cp 源文件 目的位置`
- **rm 指令**
	- 移除文件或目录
	- `rm [选项] 要删除的文件或目录`
	- 常用选项
		- -r ( -R, --recursive)：递归地删除目录及其内容
		- -f ( --force )： 强制删除，忽略是否有文件和相关参数
	- 举例：删除非空目录
		- `rm -rf 要删除的目录`
		- <font color="##c00000">强制删除不提示的方法</font>：**带上** `-f` **参数即可**
- **mv 指令**
	- 移动文件或目录 或 **重命名**
	- 重命名： `mv oldname newname` (在同一个目录下，才可以进行重命名)
	- 移动文件： `mv 移动文件 目标目录` (不在同一个目录下，进行文件的移动操作)
- **cat 指令**
	- 查看文件的内容
	- `cat [选项] 要查看的文件`
	- 常用选项
		- -n ： 显示行号
	- ❗️注意：
		- 为了浏览方便，一般会带上 <font color="##c00000" size='5' >管道命令</font> `| 其他指令 `
		-  例如： `cat -n /etc/profile | [其他命令]`
- **more 指令**
	-  基于VI编辑器的文本过滤器，可以以全屏幕的方式，按页显示文本文件的内容。
	- `more 要查看的文件`
	- 使用more以后，可以使用的交互指令：
		- ![image.png|left|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230715163810.png)
- **less 指令**
	- 分屏查看文件内容，less指令在显示文件内容时，并不是将整个文件加载之后才显示，而是根据显示需要，加载的内容，<font color="##c00000">对于显示大型文件具有较高的效率</font>！
	- `less 需要显示的文件`
	- 使用less以后，可以使用的交互指令：
		- ![image.png|left|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230715164756.png)
		- 当输入`/要查找的字符串`时，查找到以后，输入n，可以继续向下查找匹配的字符串，输入N，可以继续向上查找匹配的字符串
		- 当输入`?要查找的字符串`时，查找到以后，输入n，可以继续向上查找匹配的字符串，输入N，可以继续向下查找匹配的字符串

- **echo 指令**
	- 输入内容到控制台
	- `echo [选项] [输出内容]`

- **head 指令**
	- 用于显示文件的开头部分内容，默认情况下head指令**显示文件的前10行**
	- 基本语法
		- `head 文件名`
		- `head -n 5 文件名` 查看文件头5行内容

- **tail 指令**
	- 用于输出文件尾部内容，默认情况下，tail指令显示文件的前10行内容
	- 基本语法
		- `tail 文件` 查看尾部后10行的内容
		- `tail -n 5 文件` 查看尾部后5行的内容
		- `tail -f 文件` 实时追踪该文件的更新

- **> 指令**
	- 输出重定向指令
	- 基础语法
		- `ls -l > 文件` 将列表的内容写入文件
		- `cat file1 > file2` 将文件1的内容覆盖到文件2中

- **>> 指令**
	- 追加指令
	- 基础语法
		- `ls -al >> 文件` 将列表内容追加到文件的末尾
		- `echo 内容 >> 文件` 在文件尾部追加内容

- **ln 指令**
	- 软链接，也称为符号链接，类似于windows中的快捷方式。
	- `ln -s [原文件或目录] [软链接名]`
		- 给一个原文件创建一个软链接
	- ![image.png|left|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230715202453.png)
- **history 指令**
	- 查看已经执行过的历史命令，也可以执行历史命令
	- `history` 显示所有的历史命令
	- `history 10` 显示最近的十条指令
	- `!5` <font color="##c00000">执行历史编号为5的指令</font>

### 时间日期类

写shell脚本输出日志时可能会用到

- **data指令**
	- 显示当前日期📅
	- 基本语法
		- `date` 显示当前日期
		- `date "+%Y"` 显示当前年份
		- `date "+%m"` 显示当前月份
		- `date "+%d"` 显示当前的天
		- `date "+%Y-%m-%d %H:%M:%S"` 具体到今天的每一分，每一秒。
	- 可选选项（设置日期）
		- `data -s 字符串时间`
- **cal 指令**
	- 查看日历📅

### 搜索查找类🔍

- **find 指令**
	- find指令将从指定目录向下递归地遍历各个子目录，将满足条件的文件或目录显示在终端
	- 基本语法：`find [搜索范围] [选项]`
	- 选项说明
		- `-name 文件名` 按照指定的文件名查找文件
		- `-user 用户名` 查找属于指定用户名的所有文件
		- `-size 文件大小` 按照指定的文件大小查找文件
			- 注意
			- + 大于 - 小于 = 等于
			- 单位有： k M G
	- 比如：查找/home目录下的hello.txt文件
		- `find /home -name hello.txt`
- **locate 指令**
	- lacate指令可以快速定位文件路径，locate指令利用事先建立的系统中所有文件名称以及路径的locate数据库实现快速定位到给定的文件。locate指令无需便利整个文件系统，查询速度较快。
	- 注意：为了保证查询结果的准确度，管理员必须定期更新locate时刻。
	- 基本语法：`locate 文件名称`
	- 💡：由于locate指令基于数据库进行查询，所以<font color="##c00000">第一次运行前，必须使用updatedb指令创建locate数据库</font>
- **which指令**
	- 可以查看某个指令在哪个目录下
	- 例如： `which ls`
- **find 🆚 locate**
	- find是在硬盘上查找
	- locate是在数据库中查找
- **grep指令和管道符号 |**
	- grep 过滤查找
	- 管道符号：用于将前一个命令的处理结果输出给后面的命令进行处理。
	- 基本语法：`grep [选项] 查找内容 源文件`
	- 常用选项
		-  -n ： 显示匹配行以及行号
		-  -i ： 忽略字母大小写
	- 📋比如：在hello.txt 中，查找yes 所在行，并显示行号
		- 1⃣️ `cat hello.txt | grep "yes"`
		- 2⃣️ `grep -n "yes" hello.txt`
	- ![image.png|left|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230717221104.png)

### 压缩和解压类

- **gzip 和 gunzip**
	- gzip： 压缩文件 ，gunzip ：解压文件
	- 仅仅是对**文件**进行操作，不包含目录
	- 基本语法：
		- `gzip 文件` ： 压缩文件，<font color="##c00000">只能将文件压缩为*.gz文件</font>
		- `gunzip 文件.gz` ： 解压.gz文件
- **zip 和 unzip**
	- 解压和压缩文件
	- 基本语法
		- `zip [选项] xxx.zip 要压缩的内容` **压缩文件和目录**的命令
		- `unzip [选项] xxx.zip` 解压缩文件
	- 常用选项
		- zip ： -r （recursive）： 递归压缩（==**压缩目录**==）
		- unzip： -d 目录 ： 指定解压后文件的存放目录

- **tar 指令 ⭐️⭐️**
	- tar是打包指令，最后打包后的文件是.tar.gz的文件
	- 基本语法
		- `tar [选项] xxx.tar.gz 打包的内容`
	- 选项说明
		- -c 产生.tar 的打包文件(--creat)
		- -v 显示详细信息
		- -f 指定压缩后的文件名
		- -z 打包的时候同时压缩
			- ![image.png](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230718092827.png)
		- -x 解包.tar 文件(--extract)
	- 压缩文件:`-z(gzip)c(产生打包文件)v(显示详细信息)f(指定压缩后的文件夹)`
	- 解压文件:`-z(gunzip)x(解包.tar文件)v(显示详细信息)f(文件夹)`
	- 指定压缩目录和解压目录 `-C`
		- ![image.png](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230718094023.png)
	- 