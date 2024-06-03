"""
opencv-style recoverPose
"""
from itertools import product

import numpy as np

from .triangulate_points import triangulate_points


def decompose_essential_matrix(E):
    """
    Decompose the essential matrix into possible rotations and translations.

    Parameters:
    E (ndarray): Essential matrix (3x3).

    Returns:
    list: Possible rotation matrices (list of 3x3 ndarrays).
    list: Possible translation vectors (list of 3x1 ndarrays).
    """
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    T = U[:, 2]

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return (R1, R2), (T, -T)


def reproj_err(pts1, pts2, X3d, K, R, T):
    """
    Compute the reprojection error.

    Parameters:
    pts1 (ndarray): Points from the first image (2xN).
    pts2 (ndarray): Points from the second image (2xN).
    X3d (ndarray): 3D points in non-homogeneous coordinates (3xN).
    K (ndarray): Camera intrinsic matrix (3x3).
    R (ndarray): Rotation matrix (3x3).
    T (ndarray): Translation vector (3x1).

    Returns:
    float: The total reprojection error.
    """
    # Project 3D points back to 2D
    P1_proj = K @ X3d
    P1_proj /= P1_proj[2, :]
    P2_proj = K @ (R @ X3d + T.reshape(-1, 1))
    P2_proj /= P2_proj[2, :]

    # Compute reprojection error
    err1 = np.linalg.norm(pts1.T - P1_proj[:2, :], axis=0)
    err2 = np.linalg.norm(pts2.T - P2_proj[:2, :], axis=0)

    total_error = np.sum(err1 + err2)
    return total_error


def recover_pose(E, pts1, pts2, K):
    """
    Recover the camera pose from the essential matrix.

    Parameters:
    E (ndarray): Essential matrix (3x3).
    pts1 (ndarray): Points from the first image (Nx2).
    pts2 (ndarray): Points from the second image (Nx2).
    K (ndarray): Camera intrinsic matrix (3x3).

    Returns:
    tuple: (R, T, mask)

    Example:
    >>> import cv2
    >>> E = np.array([[0.15686827, -1.34403392, 1.32388976],
    ...               [7.5760253, -0.49269394, 47.90053954],
    ...               [-2.0351812, -48.6806962, -0.71538189]])
    >>> pts1 = np.array([[289.48815918, 1171.46777344],
    ...                  [310.84591675, 1441.82336426],
    ...                  [320.15234375, 822.1864624],
    ...                  [321.52130127, 1386.18334961],
    ...                  [346.14727783, 1472.06518555]])
    >>> pts2 = np.array([[218.17744446, 1207.95532227],
    ...                  [244.09414673, 1497.68286133],
    ...                  [246.15814209, 835.26000977],
    ...                  [255.79579163, 1437.98071289],
    ...                  [283.95248413, 1529.9609375]])
    >>> K = np.array([[2.90588e+03, 0.00000e+00, 1.41600e+03],
    ...               [0.00000e+00, 2.90588e+03, 1.06400e+03],
    ...               [0.00000e+00, 0.00000e+00, 1.00000e+00]])

    >>> error, R, T, mask = recover_pose(E, pts1, pts2, K)
    >>> _, R_cv2, T_cv2, mask_cv2 = cv2.recoverPose(E, pts1, pts2, K)
    >>> np.allclose(R, R_cv2)
    True
    >>> np.allclose(T, T_cv2)
    True
    >>> np.array_equal(mask, mask_cv2)
    True
    """
    # Decompose essential matrix
    candidates = decompose_essential_matrix(E)

    def attempt_to_recover(R, T):
        # Projection matrices
        M1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        M2 = K @ np.hstack((R, T.reshape(-1, 1)))
        # Triangulate points
        X3d_H = triangulate_points(M1, M2, pts1.T, pts2.T)  # (4, N)
        # Convert to non-homogeneous coordinates
        X3d = X3d_H[:3, :] / X3d_H[3, :]
        # Check points are in front of both cameras
        P1 = X3d
        P2 = R @ X3d + T.reshape(-1, 1)
        mask = (P1[2, :] > 0) & (P2[2, :] > 0)
        num_positive_depth = np.sum(mask)
        error = reproj_err(pts1, pts2, X3d, K, R, T)
        return num_positive_depth, (R, T, mask, error)

    _, (R, T, mask, error) = max((attempt_to_recover(R, T) for R, T in product(*candidates)), key=lambda x: x[0])

    return error, R, T[:, np.newaxis], mask.astype(int)[:, np.newaxis] * 255


if __name__ == "__main__":
    E = np.array([[0.018, -0.503, 0.866], [0.358, -0.261, -0.544], [-0.913, 0.564, 0.329]])
    pts1 = np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]])
    pts2 = np.array([[0.55, 0.55], [0.65, 0.65], [0.75, 0.75]])
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    reprojection_error, R, T, mask = recover_pose(E, pts1, pts2, K)
    print("Reprojection Error:", reprojection_error)
    print("R:", R)
    print("T:", T)
    print("mask:", mask)
