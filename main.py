"""
Incremental MultiView Structure From Motion
         ---- A simple Practice in Python

Requirements:
opencv-python
numpy
matplotlib
joblib
tqdm
networkx

@author: Hanmo Zhang
@email: zhanghanmo@bupt.edu.cn
"""
from sfm import load_calibration_data, SFM
from sfm.visualize import visualize_edge, visualize_points3d, visualize_graph


if __name__ == '__main__':
    # 将提取到的特征点以cv2.KeyPoint列表的形式存放在图的结点上，将过滤后的匹配cv2.DMatch列表，还有计算得到的本质矩阵E，基础矩阵F，存放在图的边上。
    K = load_calibration_data('./ImageDataset_SceauxCastle/images/K.txt')
    sfm = SFM('./ImageDataset_SceauxCastle/images', K)
    X3d, colors = sfm.reconstruct(return_colors=True)
    visualize_points3d(X3d, colors)
