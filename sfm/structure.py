import cv2
import numpy as np

from .transforms import H_from_RT, Homogeneous2Euler, Euler2Homogeneous


def skew(x):
    """ Create a skew symmetric matrix *A* from a 3d vector *x*.
        Property: np.cross(A, v) == np.dot(x, v)
    :param x: 3d vector
    :returns: 3 x 3 skew symmetric matrix from *x*
    """
    x = np.array(x).flatten()
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def reconstruct_one_point(pt1, pt2, m1, m2):
    """
        pt1 and m1 * X are parallel and cross product = 0
        pt1 x m1 * X  =  pt2 x m2 * X  =  0
    """
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return P / P[3]


def compute_P_from_essential(E):
    """ Compute the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. E = [t]R
    :returns: list of 4 possible camera matrices.
    """
    U, S, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices (Hartley p 258)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s


def triangulate_edge(G, K, edge, H1=None, H2=None):
    """
    输入: 两个相机的位姿
    输出: 三维世界坐标

    G: 共视图
    K: 3 * 3 相机内参
    edge: tuple[int, int] 三角化的边
    H1: 4 * 4 相机外参1
    H2: 4 * 4 相机外参2
    """
    u, v = edge
    edge_data = G[u][v]

    # Essential Matrix Decomposition.
    pts1 = np.float32([G.nodes[u]['kps'][m.queryIdx].pt for m in edge_data['matches']])  # p2d in matches of image u
    pts2 = np.float32([G.nodes[v]['kps'][m.trainIdx].pt for m in edge_data['matches']])  # p2d in matches of image v

    if H1 is None or H2 is None:
        # initialize camera pose with Essential Matrix Decomposition
        _, R, T, mask = cv2.recoverPose(edge_data['E'], pts1, pts2, K)
        mask = mask.astype(bool).flatten()
        H1 = np.eye(4)  # build the coord on the first Image
        H2 = H_from_RT(R, T)

    # projection matrices
    M1, M2 = K @ H1[:3], K @ H2[:3]

    # triangulate points
    X3d_H = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)  # (4, N)
    X3d_E = Homogeneous2Euler(X3d_H)  # (3, N)
    X3d_H = Euler2Homogeneous(X3d_E)  # (4, N)

    # Check if points are in front of both cameras
    # Transform points back to each camera coordinate system
    P1 = np.linalg.inv(H1) @ X3d_H
    P2 = np.linalg.inv(H2) @ X3d_H

    # Create masks where Z values are positive in both camera coordinate systems
    mask = (P1[2, :] > 0) & (P2[2, :] > 0)
    edge_data['mask_inliers'] = mask

    return M1, M2, H1[:3], H2[:3], X3d_E, mask