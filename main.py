"""
MultiView Structure From Motion
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

from sfm import build_graph, restore_graph, load_calibration_data, initial_register, compute_tracks, apply_increment
from sfm.visualize import visualize_edge, visualize_points3d, visualize_graph


if __name__ == '__main__':
    # 将提取到的特征点以cv2.KeyPoint列表的形式存放在图的结点上，将过滤后的匹配cv2.DMatch列表，还有计算得到的本质矩阵E，基础矩阵F，存放在图的边上。
    K = load_calibration_data('./ImageDataset_SceauxCastle/images/K.txt')
    G = build_graph('./ImageDataset_SceauxCastle/images', K)
    G = restore_graph(G)  # restore the cv2.KeyPoint and cv2.DMatch from pickleable dict

    # visualize_graph(G)
    G = compute_tracks(G)
    X3d = initial_register(G, K)  # (N, 3)
    # visualize_edge(G, u, v)
    # print(initial3d.shape)

    visualize_points3d(X3d)
    color_indices = []
    while True:
        X3d, color_indices = apply_increment(G, K, X3d, increment_mask=color_indices)
        visualize_points3d(X3d, color_indices=color_indices)
        # print(sum(1 for u, v in G.edges if G[u][v].get('dirty')))
        if all(G[u][v].get('dirty') for u, v in G.edges):
            break
