import numpy as np


def rvec_to_matrix(rvec):
    rvec = rvec.flatten()
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        return np.eye(3)
    r = rvec / theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    I = np.eye(3)
    r_cross = np.array([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ])
    R = cos_theta * I + sin_theta * r_cross + (1 - cos_theta) * np.outer(r, r)
    return R


def matrix_to_rvec(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta < 1e-6:
        return np.zeros(3)
    r = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return r * theta

