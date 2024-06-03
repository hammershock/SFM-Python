"""
Experimental Implementation of the PNP RANSAC,
"""
import numpy as np

from .solve_pnp import _solve_pnp_linear, _solve_pnp_nonlinear, reproj_error
from .transforms import matrix_to_rvec, rvec_to_matrix, euclidean_to_homogeneous


def solve_pnp_ransac(point2ds, point3ds, K, num_iterations=1000, reprojection_threshold=300.0):
    lowest_err = float('inf')

    def calc_(pt3ds, pt2ds):
        nonlocal lowest_err
        R, tvec = _solve_pnp_linear(pt3ds, pt2ds, K)
        rvec = matrix_to_rvec(R)
        success, rvec, tvec = _solve_pnp_nonlinear(pt3ds, pt2ds, K, rvec, tvec)

        errors = reproj_error(pt3ds, pt2ds, K, rvec_to_matrix(rvec), tvec)
        errors = np.linalg.norm(errors, axis=1)
        lowest_err = min(np.sum(errors), lowest_err)
        inliers = np.where(errors < reprojection_threshold)[0]
        return errors, inliers, (rvec, tvec)

    def samples():
        for _ in range(num_iterations):
            if lowest_err < reprojection_threshold:
                return
            indices = np.random.choice(len(point3ds), 6, replace=False)
            subset_point3ds, subset_point2ds = point3ds[indices], point2ds[indices]
            yield subset_point3ds, subset_point2ds

    gen = (calc_(pt3ds, pt2ds) for pt3ds, pt2ds in samples())
    errs, best_inliers, (rvec, tvec) = min(gen, key=lambda x: x[0].sum())
    assert len(best_inliers) > 0, "Not enough inliers found!"
    return rvec, tvec


def _solve_pnp_linear2(point2ds, point3ds, K):
    """Another Implementation of the solve_pnp_linear"""
    N = point3ds.shape[0]
    X_4 = euclidean_to_homogeneous(point3ds)
    x_3 = euclidean_to_homogeneous(point2ds)
    K_inv = np.linalg.inv(K)
    x_n = K_inv.dot(x_3.T).T
    A = np.zeros((3 * N, 12))

    for i in range(N):
        X = X_4[i].reshape((1, 4))
        zeros = np.zeros((1, 4))

        u, v, _ = x_n[i]

        u_cross = np.array([[0, -1, v],
                            [1, 0, -u],
                            [-v, u, 0]])
        X_tilde = np.vstack((np.hstack((X, zeros, zeros)),
                             np.hstack((zeros, X, zeros)),
                             np.hstack((zeros, zeros, X))))
        a = u_cross.dot(X_tilde)
        A[3 * i:3 * i + 3] = a

    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape((3, 4))
    R = P[:, :3]
    U_r, _, V_rT = np.linalg.svd(R)
    R = U_r.dot(V_rT)
    C = P[:, 3]
    t = -np.linalg.inv(R).dot(C)

    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    return R, t
