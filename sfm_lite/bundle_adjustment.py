import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot


def create_sparsity_matrix(n_cameras, n_points, n_obs, camera_indices, point3d_indices, fixed_camera_indices=()):
    """Create a sparsity structure for the Jacobian matrix used in optimization."""
    assert len(camera_indices) == len(point3d_indices)
    J = lil_matrix((n_obs * 2, n_cameras * 6 + n_points * 3), dtype=int)

    for i, (point_idx, cam_idx) in enumerate(zip(point3d_indices, camera_indices)):
        row = i * 2
        if cam_idx not in fixed_camera_indices:
            J[row:row + 2, cam_idx * 6:cam_idx * 6 + 6] = 1
        J[row:row + 2, n_cameras * 6 + point_idx * 3:n_cameras * 6 + point_idx * 3 + 3] = 1

    return J


def project_points(points, camera_params, K):
    """Project 3D points onto camera image planes using the camera parameters and intrinsic matrix K."""
    results = []

    for point, params in zip(points, camera_params):
        R = Rot.from_rotvec(params[:3]).as_matrix()
        T = params[3:]
        M = K @ np.hstack((R, -R @ T[:, np.newaxis]))

        projected_point = M @ np.append(point, 1)
        results.append(projected_point[:2] / projected_point[2])

    return np.array(results)


def compute_residuals(x, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals for the optimization."""
    # cut the vector into two parts
    camera_params = x[:n_cameras * 6].reshape((n_cameras, 6))  # (n_cam, 6)
    points_3d = x[n_cameras * 6:].reshape((n_points, 3))  # (n_point, 3)
    projected_points = project_points(points_3d[point_indices], camera_params[camera_indices], K)
    output = (projected_points - points_2d).ravel()
    return output

