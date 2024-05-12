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
import numpy as np

from sfm import build_graph, load_calibration_data, initial_register, compute_tracks, apply_increment
from sfm.visualize import visualize_edge, visualize_points3d, visualize_graph


if __name__ == '__main__':
    # 将提取到的特征点以cv2.KeyPoint列表的形式存放在图的结点上，将过滤后的匹配cv2.DMatch列表，还有计算得到的本质矩阵E，基础矩阵F，存放在图的边上。
    K = load_calibration_data('./ImageDataset_SceauxCastle/images/K.txt')
    G = build_graph('./ImageDataset_SceauxCastle/images', K)

    G = compute_tracks(G)
    X3d, colors = initial_register(G, K)  # (N, 3)
    visualize_graph(G)
    # visualize_edge(G, u, v)
    # visualize_points3d(X3d)

    increment_mask = []
    while True:
        ret, X3d, increment_mask, colors = apply_increment(G, K, X3d, increment_mask=increment_mask, colors=colors, min_ratio=0.05)
        if not ret:
            break

    visualize_points3d(X3d, color_indices=increment_mask, colors=np.array(colors, dtype=float))
