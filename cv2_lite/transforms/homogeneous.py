import numpy as np

from .Rodrigues import rvec_to_matrix


__all__ = ['euclidean_to_homogeneous', 'homogeneous_to_euclidean', 'H_from_RT', 'RT_from_H', 'H_from_rtvec']


def euclidean_to_homogeneous(points):
    return np.hstack((points, np.ones((points.shape[0], 1))))


def homogeneous_to_euclidean(points):
    return points[:, :-1] / points[:, -1][:, np.newaxis]


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
    R = rvec_to_matrix(rvec)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = tvec.flatten()
    return H
