---
authors: [BUAAer-xing]
---

# Ubantu软件管理命令

## 软件包格式

deb 软件包命令遵行如下约定：soft_ver-rev_arch.deb 

- soft 软件包名称
- ver 软件版本号
- revUbuntu 修订版本号
- arch 目标架构名称

## 使用 dpkg 命令来管理 deb 软件包

```shell
dpkg -i | --install xxx.deb 安装 deb 软件包
dpkg -r | --remove xxx.deb 删除软件包
dpkg -r -P | --purge xxx.deb 连同配置文件一起删除
dpkg -I | -info xxx.deb 查看软件包信息
dpkg -L xxx.deb 查看包内文件
dpkg -l 查看系统中已安装软件包信息
dpkg-reconfigure xxx 重新配置软件包
```

## APT

如果一个软件依赖关系过于复杂，使用 dpkg来安装它，并不是一个明智的选择，这个时候需要用到 APT 软件包管理系统。

APT 可以自动的检查依赖关系，通过预设的方式来获得相关软件包，并自动事实上，在多数情况下，推荐使用 APT 软件包管理系统。

 APT 系统需要一个软件信息数据库和至少一个存放着大量 deb 包的软件仓库，我们称之为 源 。 源可以是网络服务器，安装 CD 或者本地软件仓库。

需要修改 /etc/apt/sources.list 文件，使 APT 系统能够连接到 源。

APT 系统主要包括 apt-get 和 apt-cache 等命令。

通常是复合命令，包含若干个子命令。

```shell
apt-get install xxx 安装 xxx
-d 仅下载
-f 强制安装
apt-get remove xxx 卸载 xxx
apt-get update 更新软件信息数据库
apt-get upgrade 进行系统升级
apt-cache search 搜索软件包
```

说明：建议经常使用`sudo apt-get update` 命令来更新您的软件信息数据库

