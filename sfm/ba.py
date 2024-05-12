import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
import time


def euler_from_matrix(R):
    return Rot.from_matrix(R).as_rotvec()


def rotation_from_euler(euler):
    return Rot.from_rotvec(euler).as_matrix()


def find_visible_points(X_found, filtered_feature_flag, nCam):
    """Identify indices and visibility matrix for 3D points visible in cameras."""
    visibility = np.any(filtered_feature_flag[:, :nCam + 1], axis=1)
    visible_indices = np.where(X_found & visibility)[0]
    visibility_matrix = filtered_feature_flag[visible_indices, :nCam + 1]
    return visible_indices, visibility_matrix


def extract_2D_points(indices, visibility_matrix, feature_x, feature_y):
    """Extract 2D points based on the visibility matrix and indices."""
    visible_features_x = feature_x[indices]
    visible_features_y = feature_y[indices]
    pts2D = []
    for i in range(len(indices)):
        for j in range(visibility_matrix.shape[1]):
            if visibility_matrix[i, j]:
                pts2D.append([visible_features_x[i, j], visible_features_y[i, j]])
    return np.array(pts2D)


def create_sparsity_matrix(n_cameras, n_points, visibility_matrix):
    """Create a sparsity structure for the Jacobian matrix used in optimization."""
    n_observations = np.sum(visibility_matrix)
    m = n_observations * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    obs_indices = np.nonzero(visibility_matrix)
    for i, (point_idx, cam_idx) in enumerate(zip(*obs_indices)):
        row = i * 2
        A[row:row + 2, cam_idx * 6:cam_idx * 6 + 6] = 1
        A[row:row + 2, n_cameras * 6 + point_idx * 3:n_cameras * 6 + point_idx * 3 + 3] = 1
    return A


def project(points, camera_params, K):
    """Project 3D points onto camera image planes."""
    results = []
    for cam_idx, (R, C) in enumerate(camera_params):
        P = K @ R @ np.hstack((np.eye(3), -C.reshape(-1, 1)))
        for point in points:
            x_hom = P @ np.hstack((point, 1))
            x_proj = x_hom[:2] / x_hom[2]
            results.append(x_proj)
    return np.array(results)


def compute_residuals(x, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals for the optimization."""
    camera_params = x[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = x[n_cameras * 6:].reshape((n_points, 3))
    projected_points = project(points_3d[point_indices], camera_params[camera_indices], K)
    return (projected_points - points_2d).ravel()


def get_camera_and_point_indices(visibility_matrix):
    """Extract indices for cameras and corresponding points based on visibility matrix."""
    camera_indices = []
    point_indices = []
    for i in range(visibility_matrix.shape[0]):
        for j in range(visibility_matrix.shape[1]):
            if visibility_matrix[i, j]:
                camera_indices.append(j)
                point_indices.append(i)
    return np.array(camera_indices), np.array(point_indices)


def bundle_adjustment(X_all, X_found, feature_x, feature_y, filtered_feature_flag, R_set, C_set, K, n_cam):
    """Perform bundle adjustment to optimize camera poses and 3D point locations."""
    indices, visibility_matrix = find_visible_points(X_found, filtered_feature_flag, n_cam)
    points_3d = X_all[indices]
    points_2d = extract_2D_points(indices, visibility_matrix, feature_x, feature_y)

    camera_params = [(euler_from_matrix(R), C) for R, C in zip(R_set, C_set)]
    initial_guess = np.hstack([np.hstack([e, C]).ravel() for e, C in camera_params] + [points_3d.ravel()])

    # Generate sparsity matrix and indices
    camera_indices, point_indices = get_camera_and_point_indices(visibility_matrix)
    sparsity_matrix = create_sparsity_matrix(n_cam + 1, len(points_3d), visibility_matrix)

    result = least_squares(compute_residuals, initial_guess, jac_sparsity=sparsity_matrix, verbose=2,
                           x_scale='jac', ftol=1e-10, method='trf',
                           args=(n_cam + 1, len(points_3d), camera_indices, point_indices, points_2d, K))

    optimized_params = result.x
    return optimized_params  # Further processing could be done to extract optimized camera poses and 3D points


# Usage of the function would involve setting up the initial parameters followed by a call to `bundle_adjustment`.
if __name__ == '__main__':
    # 3D points (shape [n_points, 3])
    X_all = np.array([
        [1.0, 2.0, 4.0],
        [2.0, 3.0, 5.0],
        [3.0, 4.0, 6.0],
        [4.0, 5.0, 7.0]
    ])

    # Found points flag (shape [n_points])
    X_found = np.array([True, False, True, True])

    # 2D feature coordinates (shape [n_points, n_cameras])
    feature_x = np.array([
        [100, 200],
        [150, 250],
        [130, 230],
        [120, 220]
    ])
    feature_y = np.array([
        [110, 210],
        [160, 260],
        [140, 240],
        [125, 225]
    ])

    # Visibility flags (shape [n_points, n_cameras])
    filtered_feature_flag = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 1]
    ])

    # Camera rotations and translations
    R_set = [np.eye(3), np.eye(3)]  # Identity matrices for simplicity
    C_set = [np.array([0, 0, 0]), np.array([1, 1, 1])]  # Simple translations

    # Camera intrinsic matrix
    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])

    # Number of cameras
    n_cam = 2

    bundle_adjustment(X_all, X_found, feature_x, feature_y, filtered_feature_flag, R_set, C_set, K, n_cam)