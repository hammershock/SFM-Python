import cv2
import networkx as nx
import numpy as np

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)


def display_depth(img_left, img_right, K, R1, R2, t1, t2):
    # Step 1: Calculate relative rotation and translation
    R_rel = np.dot(R2, R1.T)
    t_rel = t2 - np.dot(R_rel, t1)

    # Step 2: Stereo rectify
    R1_rect, R2_rect, P1_rect, P2_rect, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1=K, distCoeffs1=None,
        cameraMatrix2=K, distCoeffs2=None,
        imageSize=img_left.shape[::-1],
        R=R_rel, T=t_rel
    )

    # Step 3: Compute the disparity map
    # Assuming you've already initialized stereo as StereoBM or StereoSGBM object
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img_left, img_right)

    # Step 4: Convert disparity to depth
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth = points_3D[:, :, 2]

    # Step 5: Normalize and display the depth map
    depth_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_normalized = cv2.resize(depth_normalized, (640, 480))  # Resize for display purposes
    cv2.imshow('Depth Map', depth_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_edge_depth(G: nx.DiGraph, edge, K):
    u, v = edge
    n1, n2 = G.nodes[u], G.nodes[v]
    image_left = n1['image']
    image_right = n2['image']
