import numpy as np
from scipy.optimize import least_squares

from cv2_lite.transforms import matrix_to_rvec, rvec_to_matrix
from cv2_lite.check_inputs import check_input_shapes


@check_input_shapes
def reproj_error(point3ds, point2ds, K, R, tvec):
    projected_points_2d = (K @ (R @ point3ds.T + tvec.reshape(3, 1))).T
    projected_points_2d /= projected_points_2d[:, 2].reshape(-1, 1)
    projected_points_2d = projected_points_2d[:, :2]
    error = (projected_points_2d - point2ds)
    return error


@check_input_shapes
def _solve_pnp_linear(point3ds, point2ds, K):
    num_points = point3ds.shape[0]
    point2ds_h = np.hstack((point2ds, np.ones((num_points, 1))))
    normalized_point2ds = np.linalg.inv(K) @ point2ds_h.T
    normalized_point2ds = normalized_point2ds[:2].T

    A = np.zeros((2 * num_points, 12))
    for i in range(num_points):
        X, Y, Z = point3ds[i]
        u, v = normalized_point2ds[i]
        A[2*i] = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1].reshape(3, 4)
    R_t = h[:, :3]
    t = h[:, 3]

    # Enforce orthogonality of R
    U, _, Vt = np.linalg.svd(R_t)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    return R, t


@check_input_shapes
def _solve_pnp_nonlinear(point3ds, point2ds, K, rvec, tvec):
    initial_guess = np.hstack((rvec.flatten(), tvec))

    def target_func(params, points_3d, points_2d, K):
        return reproj_error(points_3d, points_2d, K, R=rvec_to_matrix(params[:3]), tvec=params[3:6]).ravel()

    result = least_squares(target_func, initial_guess, args=(point3ds, point2ds, K))

    # optimized results
    rvec = result.x[:3][:, np.newaxis]
    tvec = result.x[3:6][:, np.newaxis]

    return result.success, rvec, tvec


@check_input_shapes
def solve_pnp(point3ds, point2ds, K, *args, **kwargs):
    R, tvec = _solve_pnp_linear(point3ds, point2ds, K)
    success, rvec, tvec = _solve_pnp_nonlinear(point3ds, point2ds, K, matrix_to_rvec(R), tvec)
    return success, rvec, tvec


if __name__ == "__main__":
    point3ds = np.array(
        [[-2.83510726, 0.35871423, 7.39620667], [-2.64956519, 1.26203440, 7.24983088],
         [-2.67895385, -0.01053622, 7.470429], [-2.49225235, 0.81401947, 7.36628308],
         [-2.31513876, -0.16642927, 7.48943918], [-2.40576114, -1.25461447, 8.02639394],
         [-2.12475213, 0.15831901, 7.43070466], [-1.88875539, 1.02978915, 8.18975227],
         [-1.70972510, -0.60470877, 8.44642688]])

    point2ds = np.array([[232.86323547, 1230.31469727], [294.15100098, 1622.02612305], [312.4229126, 1076.39611816],
                         [377.7600708, 1423.75268555], [470.68951416, 1014.42590332], [477.48834229, 604.61303711],
                         [544.51257324, 1148.95483398], [672.24969482, 1470.36743164], [754.92327881, 871.83178711]])

    K = np.array([[2.90588e+03, 0.00000e+00, 1.41600e+03],
                  [0.00000e+00, 2.90588e+03, 1.06400e+03],
                  [0.00000e+00, 0.00000e+00, 1.00000e+00]])
    success, rvec, tvec = solve_pnp(point3ds, point2ds, K)

    import cv2

    success2, rvec2, tvec2 = cv2.solvePnP(point3ds, point2ds, K, None)
    print(success == success2 and np.allclose(rvec, rvec2) and np.allclose(tvec, tvec2))
