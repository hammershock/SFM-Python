import cv2
import numpy as np

from .transforms import H_from_RT, normalize_homogeneous


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


def generate_visibility_mask(H1, H2, X3d_H):
    # Generate Mask
    P1 = np.linalg.inv(H1) @ X3d_H
    P2 = np.linalg.inv(H2) @ X3d_H

    # Create masks where Z values are positive in both camera coordinate systems
    mask = (P1[2, :] > 0) & (P2[2, :] > 0)
    return mask


def triangulate_edge(pts1, pts2, K, H1, H2):
    """
    输入: 两个相机的位姿
    输出: 三维世界坐标

    G: 共视图
    K: 3 * 3 相机内参
    edge: tuple[int, int] 三角化的边
    H1: 4 * 4 相机外参1
    H2: 4 * 4 相机外参2
    """
    # projection matrices
    M1, M2 = K @ H1[:3], K @ H2[:3]

    # triangulate points
    X3d_H = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)  # (4, N)
    X3d_H = normalize_homogeneous(X3d_H)  # (4, N)
    X3d_E = X3d_H[:3, :]
    mask = generate_visibility_mask(H1, H2, X3d_H)
    return X3d_E, mask

