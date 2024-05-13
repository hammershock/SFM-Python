import cv2
import numpy as np


def H_from_RT(R, T):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = T.flatten()
    return H


def RT_from_H(H):
    R = H[:3, :3]
    T = H[:3, 3].flatten()
    return R, T


def H_from_rtvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = tvec.flatten()
    return H


def Homogeneous2Euler(points_H):
    assert points_H.shape[0] == 4 or points_H.shape[0] == 3  # (4, N) or (3, N)
    scale = points_H[-1]
    points_H /= scale
    return points_H[:3]


def Euler2Homogeneous(points_E):
    assert points_E.shape[0] == 3 or points_E.shape[0] == 2
    points_H = np.vstack((points_E, np.ones((1, points_E.shape[1]))))
    return points_H


def normalize_homogeneous(points_H):
    assert points_H.shape[0] == 4 or points_H.shape[0] == 3
    points_H /= points_H[-1]
    return points_H

