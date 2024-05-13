import cv2
import networkx as nx
import numpy as np

from .transforms import RT_from_H


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)


def display_depth(img_left, img_right, K, R1, R2, t1, t2):
    # Step 1: Calculate relative rotation and translation
    R_rel = np.dot(R2, R1.T)
    t_rel = t2 - np.dot(R_rel, t1)
    # Step 2: Stereo rectify
    R1_rect, R2_rect, P1_rect, P2_rect, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1=K, distCoeffs1=None,
        cameraMatrix2=K, distCoeffs2=None,
        imageSize=img_left.shape[:2][::-1],
        R=R_rel, T=t_rel
    )

    # Step 3: Compute the disparity map
    # Assuming you've already initialized stereo as StereoBM or StereoSGBM object
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=21)
    # 调整视差参数
    stereo.setMinDisparity(0)

    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    # 归一化视差图以便显示
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_normalized = cv2.resize(disparity_normalized, (640, 480))

    cv2.imshow('Disparity', disparity_normalized)
    cv2.waitKey(0)

    # Step 4: Convert disparity to depth
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth = points_3D[:, :, 2]

    # Step 5: Normalize and display the depth map
    # 调整深度图归一化策略
    valid_depth_mask = ((depth > 0) & (depth < 50000)).astype(np.uint8)  # 假设有效深度范围为0到5000
    depth_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U,
                                     mask=valid_depth_mask)


    # depth_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_normalized = cv2.resize(depth_normalized, (640, 480))  # Resize for display purposes
    cv2.imshow('Depth Map', depth_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 检查Q矩阵是否正确
    print(Q)


def display_edge_depth(G: nx.DiGraph, edge, K):
    u, v = edge
    n1, n2 = G.nodes[u], G.nodes[v]
    image_left = cv2.cvtColor(n1['image'], cv2.COLOR_BGR2GRAY)
    image_right = cv2.cvtColor(n2['image'], cv2.COLOR_BGR2GRAY)
    if 'H' in n1 and 'H' in n2:
        H1 = n1['H']
        H2 = n2['H']
        R1, T1 = RT_from_H(H1)
        R2, T2 = RT_from_H(H2)
        display_depth(image_left, image_right, K, R1, R2, T1, T2)

