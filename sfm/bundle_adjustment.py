import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from sfm.data_stucture import X3D


def find_visible_points(X_found, filtered_feature_flag, nCam):
    """Identify indices and visibility matrix for 3D points visible in cameras."""
    visibility = np.any(filtered_feature_flag[:, :nCam + 1], axis=1)
    visible_indices = np.where(X_found & visibility)[0]
    visibility_matrix = filtered_feature_flag[visible_indices, :nCam + 1]
    return visible_indices, visibility_matrix


def extract_2D_points(visibility_matrix, feature_x, feature_y):
    """Extract 2D points based on the visibility matrix and indices."""
    pts2D = []
    for i in range(visibility_matrix.shape[0]):
        for j in range(visibility_matrix.shape[1]):
            if visibility_matrix[i, j]:
                pts2D.append([feature_x[i, j], feature_y[i, j]])
    return np.array(pts2D)


def create_sparsity_matrix(n_cameras, n_points, camera_indices, point_indices):
    """Create a sparsity structure for the Jacobian matrix used in optimization."""
    assert len(camera_indices) == len(point_indices)
    n_observations = len(camera_indices)
    m = n_observations * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    for i, (point_idx, cam_idx) in enumerate(zip(point_indices, camera_indices)):
        row = i * 2
        A[row:row + 2, cam_idx * 6:cam_idx * 6 + 6] = 1
        A[row:row + 2, n_cameras * 6 + point_idx * 3:n_cameras * 6 + point_idx * 3 + 3] = 1
    return A


def project(points, camera_params, K):
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
    camera_params = x[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = x[n_cameras * 6:].reshape((n_points, 3))
    projected_points = project(points_3d[point_indices], camera_params[camera_indices], K)
    output = (projected_points - points_2d).ravel()
    return output


def decode_optimized_parameters(x, n_cam, n_points):
    """Decode optimized parameters from the result of least squares optimization."""
    # Camera parameters are the first n_cam * 6 elements
    camera_params = x[:n_cam * 6].reshape((n_cam, 6))
    optimized_R_set = []
    optimized_C_set = []
    for params in camera_params:
        euler_angles = params[:3]
        translation = params[3:]
        rotation_matrix = Rot.from_rotvec(euler_angles).as_matrix()
        optimized_R_set.append(rotation_matrix)
        optimized_C_set.append(translation.reshape(3, 1))

    # Remaining elements are the 3D point coordinates
    optimized_points_3d = x[n_cam * 6:].reshape((n_points, 3))

    return (optimized_R_set, optimized_C_set), optimized_points_3d


def bundle_adjustment(X_all: X3D, R_set, C_set, K, node_indices, tol=1e-10, verbose=0):
    """
    Perform bundle adjustment to optimize camera poses and 3D point locations.

    Args:
        X_all (numpy.ndarray): Array of all 3D points in the scene, shape (n_points, 3).
        R_set (list of numpy.ndarray): List of rotation matrices for each camera, each shape (3, 3).
        C_set (list of numpy.ndarray): List of translation vectors for each camera, each shape (3,).
        K (numpy.ndarray): Camera intrinsic matrix, shape (3, 3).
        node_indices (list):

    Returns:
        tuple: A tuple containing:
            - tuple: A tuple of lists, where the first list contains rotation matrices (each shape (3, 3)) and the second list contains translation vectors (each shape (3, 1)) for each camera.
            - numpy.ndarray: Array of optimized 3D points, shape (n_filtered_points, 3), where n_filtered_points is the number of 3D points visible in at least one camera.

    This function optimizes the camera poses and 3D point positions by minimizing the reprojection error using the least squares method. The optimization adjusts both the camera parameters (rotation and translation) and the coordinates of the 3D points based on the observed 2D points in the camera image planes.
    """
    points_2d, camera_indices, point_indices = X_all.points2d()
    camera_ids = np.unique(camera_indices)
    continuous_indices = {item: i for i, item in enumerate(camera_ids)}
    n_cam = len(continuous_indices)
    assert n_cam == len(R_set) == len(C_set)
    camera_indices = np.array([continuous_indices[item] for item in camera_indices])
    camera_params = [(Rot.from_matrix(R).as_rotvec(), C.flatten()) for R, C in zip(R_set, C_set)]
    initial_guess = np.hstack([np.hstack([e, C]).ravel() for e, C in camera_params] + [X_all.data.ravel()])  # (696,)

    # Generate sparsity matrix and indices
    jac_sparsity = create_sparsity_matrix(n_cam, len(X_all), camera_indices, point_indices)  # (8, 21)

    result = least_squares(compute_residuals, initial_guess, jac_sparsity=jac_sparsity, verbose=verbose,
                           x_scale='jac', ftol=tol, method='trf',
                           args=(n_cam, len(X_all), camera_indices, point_indices, points_2d, K))

    (optimized_R_set, optimized_C_set), optimized_points_3d = decode_optimized_parameters(result.x, n_cam, len(X_all))
    return (optimized_R_set, optimized_C_set), optimized_points_3d, camera_ids
