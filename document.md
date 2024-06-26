## 背景

运动恢复结构，SFM，Structure From Motion，旨在从多视角的图像数据恢复出三维场景的结构信息，
是三维重建的一个重要环节，
SFM系统的输入是图像，输出是稀疏点云
（为什么稀疏？并不对图像中的所有点进行重建，而是仅仅对关键点进行重建）

我们学习过很多种类的运动恢复结构，题目中限定相机内参数已知，所以我们这里进行欧式结构恢复

## 两张图像——最简单的情形

如何恢复结构？
已知（Poses，K，和对应点）：三角化
线性解法，非线性解法
要求相机位姿必须已知！

不同于双目立体视觉，在SFM中相机位姿是未知的，也是需要计算的一部分
如何？

还是两幅图像的情形，但相机位姿未知
已知若干对应点
则可以估计基础矩阵！（归一化八点法）

已知相机内参数K，则可以计算本质矩阵E。

相机相对位姿可以通过分解本质矩阵得到！
设第一个相机参考系为世界系，可以得到第二个相机的位姿为R，T。
则又回到了已知R，T和对应点的情形，可以应用三角化！

## 更多图像！增量式扩充三维点云

问题的关键：获取相机位姿！（这也被成为相机注册register）

如何获取其他的相机位姿？PnP，N点透视解算！
已知若干个三维点和二维点的对应（大于6对），则可以解算相机位姿R，T

在初始的重建之后，我们已经获取了部分三维点，如果新的相机恰好拍摄到了这些点，则可以获取这个相机位姿（相机注册）

已经注册的相机之间现在之前没有被重建过的匹配点现在可以重建了！

## 深入其中的细节

### 1. SIFT特征提取

### 2. BF特征匹配

两两之间匹配（局部匹配），构造共视图
匹配要经过Ratio Test过滤
只有两两之间匹配的特征点超过一定阈值（80对）才可以保留这条边

### 3. 构建Track

遍历所有的边，构建Track，三维点留空

### 4. 初始重建

对每个边：
	进行鲁棒基础矩阵F估计，再次筛选对应点
	尝试进行初始重建（本质矩阵分解获得位姿，三角化重建）
	计算光束夹角中位数
只选取光束夹角中位数在3到60度之间的（且越低越好）作为初始重建结果
初始相机注册
标记这条边已经使用过
维护已经重建的三维点的对应Track

## 5. 补充三维点

选取最好的一条边，用来三维重建
（选取可以有很多策略）
尝试通过PnP注册这条边的没有被注册的顶点
通过三角化，重建这条边上没有被重建过的对应点
标记这条边为已使用过
维护这条边上的重建好的Track

重复这个流程，直到所有的边都已经被标记过为止。