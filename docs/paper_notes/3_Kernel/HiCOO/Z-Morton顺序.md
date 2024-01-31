## 莫顿码

### 概述

莫顿码是**将多维数据转化为一维数据的编码**。

莫顿编码定义了**一条 Z 形的空间填充曲线**，因此莫顿编码通常也称Z阶曲线(Z-order curve)。 在 N 维空间中对于彼此接近的坐标具有彼此接近的莫顿码, 可以应用于为一个整数对产生一个唯一索引。例如，对于坐标系中的坐标点使用莫顿编码生成的莫顿码，可以唯一索引对应的点。这些索引为“Z”形排序 。如下图以Z形(左上->右上->左下->右下)分别代表1\*1、2\*2、4\*4、8\*8 平方单位：
![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240110181235.png)

### 编码规则

十进制编码规则：首先，行列号转为二进制（从第0行0列开始）；然后行列号交叉排列（yxyx…）；最后将二进制结果转为十进制。Morton编码是按左上，右上，左下，右下的顺序从0开始对每个格网进行自然编码的。如下图（二维空间）：展示了8\*8的图像每个像素的空间编码，从000000到111111，用一维二进制数，编码了x,y值在0-7的位置坐标。图中蓝色数字代表x轴，红色数字代表y轴，网格中的二进制数由x和y的二进制数交叉构成。

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240110181417.png)
### 更加高纬度的空间

![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20240110181459.png)
### 意义

Z-Morton曲线的命名来自于Morton码，也被称为**Z-order码**。

Morton码是一种将多维空间中的坐标映射到一维空间的方法，它与Z-Morton曲线有关联。在Morton码中，<font color='red'><b>相邻的多维坐标被编码为相邻的一维索引</b></font>，这种编码方式使得<font color='red'><b>在多维空间中的数据能够以线性方式进行存储和访问</b></font>。

Z-Morton曲线和Morton码的特点使得它们在空间索引数据结构（比如四叉树、八叉树等）、并行计算、以及一些计算几何算法中有着广泛的应用，因为**它们能够有效地处理多维数据，并提供高效的访问方式**。