# 02-vi和vim的使用

## vi和vim的区别

- vi 是linux系统中内置的文本编辑器
- vim具有程序编辑能力

## vi和vim常用的三种模式

- **正常模式**
	- 使用vim打开一个文件，就默认进入正常模式
	- 可以使用方向键【上下左右】来移动光标
	- 可以使用【删除字符/删除整行】来处理文件内容
	- 也可以使用【复制/粘贴】快捷键
- **插入模式**
	- 一般来说，在正常模式下，按下`i`或者`I`，也就是insert的首字母，即可进入插入模式。
	- 插入模式=编辑模式，可以进行输入操作
- **命令行模式**
	- 进入命令行模式步骤：首先按下`esc建`，再输入`：`即可进入。
	- 可以提供相关指令，完成读取、存盘、替换、离开vim编辑器、显示行号等操作
	- 输入含义
		- :wq 保存并退出
		- :q 退出
		- :q! 强制退出，不保存

![image.png|center|300](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230709105256.png)

