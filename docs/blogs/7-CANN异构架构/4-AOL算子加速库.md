
## 4.1-接口简介

为加速模型算力释放，CANN（Compute Architecture for Neural Networks）提供了算子加速库（Ascend Operator Library，简称AOL）。

该库提供了一系列丰富的深度优化、硬件亲和的高性能算子，如Neural Network、Digital Vision Pre-Processing算子等，为神经网络在昇腾硬件上加速计算奠定了基础。为方便开发者调用算子，提供了**单算子API执行方式**调用算子（基于C语言的API，无需提供IR（Intermediate Representation）定义），以便开发者快速且高效使能模型创新与应用，API的调用流程如图所示。

![image.png|center|400](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427145908.png)

算子规格清单如下：
![image.png|center|600](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/20250427150022.png)

## 4.2-AOL加速库

更加详细的内容：[https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/aolapi/operatorlist_00001.html](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/aolapi/operatorlist_00001.html)



















