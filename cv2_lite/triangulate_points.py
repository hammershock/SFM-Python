"""
opencv-style Triangulate Points
"""
import numpy as np
from scipy.optimize import least_squares


__all__ = ['skew', 'triangulate_points_linear', 'triangulate_points']


def skew(x):
    """
    Return the skew-symmetric matrix of a vector.

    Parameters:
    x (ndarray): A 3-element vector.

    Returns:
    ndarray: A 3x3 skew-symmetric matrix.
    """
    assert x.shape == (3, )
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def triangulate_points_linear(M1: np.ndarray, M2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Linear triangulation to get initial 3D points.

    Parameters:
    M1 (ndarray): Projection matrix of the first view.
    M2 (ndarray): Projection matrix of the second view.
    pts1 (ndarray): Points from the first image (2xN).
    pts2 (ndarray): Points from the second image (2xN).

    Returns:
    ndarray: Triangulated 3D points in homogeneous coordinates (4xN).
    """
    assert M1.shape == M2.shape == (3, 4), f"shape of M Matrix should be (3, 4)"
    assert pts1.shape[1] == pts2.shape[
        1], f"input 2D point should have the same size, but got {M1.shape} and {M2.shape}"
    assert pts1.shape[0] == pts2.shape[0] == 2, f"input points should be 2-dim but got {M1.shape[1]}-dim"
    num_points = pts1.shape[1]
    X3d_H = np.zeros((4, num_points))

    for i in range(num_points):
        x1, y1 = pts1[:, i]
        x2, y2 = pts2[:, i]

        skew1 = skew(np.array([x1, y1, 1]))
        skew2 = skew(np.array([x2, y2, 1]))

        A = np.vstack((skew1 @ M1, skew2 @ M2))
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X3d_H[:, i] = X / X[-1]  # normalize

    return X3d_H


def triangulate_points_linear2(M1, M2, pts1, pts2):
    """
    Triangulate points using nonlinear triangulation to refine 3D points.

    Parameters:
    M1 (ndarray): Projection matrix of the first view.
    M2 (ndarray): Projection matrix of the second view.
    pts1 (ndarray): Points from the first image (2xN).
    pts2 (ndarray): Points from the second image (2xN).

    Returns:
    ndarray: Refined triangulated 3D points in homogeneous coordinates (4xN).
    """
    num_points = pts1.shape[1]
    X = np.zeros((4, num_points))

    for i in range(num_points):
        A = np.array([
            pts1[0, i] * M1[2, :] - M1[0, :],
            pts1[1, i] * M1[2, :] - M1[1, :],
            pts2[0, i] * M2[2, :] - M2[0, :],
            pts2[1, i] * M2[2, :] - M2[1, :]
        ])
        _, _, Vt = np.linalg.svd(A)
        X[:, i] = Vt[-1]

    X /= X[3]
    return X


def __reproj_err(params, pts1, pts2, M1, M2):
    """
    target function of optimizer
    Compute the reprojection error.

    Parameters:
    params (ndarray): Flattened 3D points (3*N).
    pts1 (ndarray): Points from the first image (2xN).
    pts2 (ndarray): Points from the second image (2xN).
    M1 (ndarray): Projection matrix of the first view.
    M2 (ndarray): Projection matrix of the second view.

    Returns:
    ndarray: Reprojection error (2*N).
    """
    num_points = pts1.shape[1]
    X3d = np.reshape(params, (3, num_points))

    X3d_hom = np.vstack((X3d, np.ones(num_points)))

    pts1_proj = (M1 @ X3d_hom)[:2] / (M1 @ X3d_hom)[2]
    pts2_proj = (M2 @ X3d_hom)[:2] / (M2 @ X3d_hom)[2]

    error = np.hstack((pts1_proj - pts1, pts2_proj - pts2))
    error = np.square(error).flatten()
    return error


def triangulate_points(M1, M2, pts1, pts2):
    """
    Nonlinear triangulation to refine 3D points.

    Parameters:
    M1 (ndarray): Projection matrix of the first view.
    M2 (ndarray): Projection matrix of the second view.
    pts1 (ndarray): Points from the first image (2xN).
    pts2 (ndarray): Points from the second image (2xN).

    Returns:
    ndarray: Refined triangulated 3D points in homogeneous coordinates (4xN).

    example:
    >>> import cv2
    >>> M1 = np.array([
    ...    [2.26648588e+03, -1.20335940e+01, 2.30479750e+03, -8.10676693e+03],
    ...    [-5.27039708e+02, 2.80897851e+03, 1.18663534e+03, -8.67245706e+02],
    ...    [-3.31554792e-01, -8.05176652e-02, 9.39993790e-01, -2.70143322e-01]
    ... ])

    >>> M2 = np.array([
    ...    [2.57889420e+03, 2.21368015e+01, 1.94884818e+03, -5.51844918e+03],
    ...    [-2.65746745e+02, 2.88325548e+03, 1.09199408e+03, -4.01890574e+02],
    ...    [-1.92197430e-01, -2.02833813e-02, 9.81146642e-01, -7.57015427e-02]
    ... ])

    >>> pts1 = np.array([
    ...    [356.43856812, 1148.45336914],
    ...    [360.64306641, 1452.07055664],
    ...    [362.01452637, 1746.91137695],
    ...    [380.1892395, 1124.84753418],
    ... ])

    >>> pts2 = np.array([
    ...    [238.98179626, 1011.63299561],
    ...    [251.32275391, 1309.41662598],
    ...    [263.76693726, 1595.79040527],
    ...    [265.42037964, 989.33227539],
    ... ])

    >>> M1.shape
    (3, 4)
    >>> pts1.shape
    (4, 2)
    >>> X3d_H = triangulate_points(M1, M2, pts1.T, pts2.T)  # our X3d
    >>> X3d_H_cv2 = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)  # cv2 X3d result

    >>> X3d_custom = X3d_H / X3d_H[3]
    >>> X3d_cv2 = X3d_H_cv2 / X3d_H_cv2[3]

    >>> np.allclose(X3d_custom, X3d_cv2, 1e-3)
    True
    """
    X3d_H_initial = triangulate_points_linear(M1, M2, pts1, pts2)
    X3d_initial = X3d_H_initial[:3] / X3d_H_initial[3]

    res = least_squares(__reproj_err, X3d_initial.flatten(), args=(pts1, pts2, M1, M2))

    X3d_refined = np.reshape(res.x, (3, pts1.shape[1]))
    X3d_H_refined = np.vstack((X3d_refined, np.ones(pts1.shape[1])))

    return X3d_H_refined

