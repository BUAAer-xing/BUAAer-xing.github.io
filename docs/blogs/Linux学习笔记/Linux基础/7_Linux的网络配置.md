# 07-Linux的网络配置

## 设置主机名

- 查看主机名：`hostname`
- 修改主机名：`vim /etc/hostname` ，重启后生效
![[1-Linux目录结构#^f9a9d2]]

## 设置hosts映射

在win中：

C:\\Windows\\System32\\drivers\\etc\\hosts

在linux中：

修改/etc/hosts文件，进行指定

# 主机名解析分析过程

![image.png|center|600](https://jsd.cdn.zzko.cn/gh/NEUQer-xing/Markdown_images@master/images-2/20230720100409.png)


## hosts

"hosts" 是一个计算机中的文件，用于映射主机名和IP地址。

它通常用于在本地计算机上设置自定义的主机名解析规则。通过编辑hosts文件，你可以将特定的主机名映射到特定的IP地址，以便在浏览器或其他网络应用程序中访问这些主机名时，可以直接使用指定的IP地址进行连接。这对于测试网站、阻止恶意网站或在本地环境中进行开发和调试非常有用。

## DNS

DNS代表域名系统（Domain Name System）。

它是互联网中的一种网络协议，用于将域名（如example.com）转换为IP地址（如192.0.2.1），以便计算机能够识别和访问特定的网络资源。

DNS起到了类似电话簿的作用，帮助用户在互联网上定位和访问网站、电子邮件服务器和其他网络服务。


