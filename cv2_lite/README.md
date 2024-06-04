# cv2-lite

一个使用numpy和scipy实现的opencv部分功能，作为原理展示，包括：

- cv2.findFundamentalMat
  给定若干个2D点对估计基础矩阵，提供RANSAC鲁棒估计
- cv2.solvePnP
  给定相机内参数和若干个2D-3D点对，估计摄像机位姿，提供线性解法和非线性解法
- cv2.solveP3P
  给定相机内参数和3对2D-3D点对，使用纯几何方法获取4种可能的相机位姿
- cv2.recoverPose
  分解本质矩阵获取相机位姿，用于双目相机标定
- cv2.triangulatePoints
  给定相机内参数，相机位姿以及2D点匹配，用于重建匹配点三维坐标

补充：
- 这里是一个简单的[非线性优化器](./least_squares.py)的例子，用来替代scipy.least_square，展示优化器的工作原理
  
