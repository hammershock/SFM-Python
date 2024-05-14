import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from .data_stucture import X3D
from .transforms import H_from_RT, RT_from_H


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


def bundle_adjustment(G: nx.DiGraph, X_all: X3D, K, tol=1e-10, verbose=0):
    """
    Perform bundle adjustment to optimize camera poses and 3D point locations.
    This function optimizes the camera poses and 3D point positions by minimizing the reprojection error using the least squares method. The optimization adjusts both the camera parameters (rotation and translation) and the coordinates of the 3D points based on the observed 2D points in the camera image planes.
    """
    # {2: (R2, T2), 3: (R3, T3), 5: (R5, T5), ...}
    registered_cameras = {node: RT_from_H(data['H']) for node, data in G.nodes(data=True) if 'H' in data}
    initial_camera = next(node for node, data in G.nodes(data=True) if 'base' in data)
    point_2ds, camera_nodes, point_3d_indices = X_all.points2d(registered_cameras)
    # [n_cam * 6,                  | n_points * 3      ]
    # [e1, t1, e2, t2, e3, t3, ..., X1, X2, X3, ..., Xn]

    camera_node2index = {idx: i for i, idx in enumerate(registered_cameras.keys())}
    camera_indices = np.array([camera_node2index[node] for node in camera_nodes])
    camera_params = [(Rot.from_matrix(R).as_rotvec(), T.flatten()) for R, T in registered_cameras.values()]
    initial_guess = np.hstack([np.hstack([e, t]).ravel() for e, t in camera_params] + [X_all.data.ravel()])
    # [n_observations,                                 ]
    # [x1, x2, x3, x4, x5, ....                    , xn]
    # [c1, c2, c3, c4, c5, ....                    , cn]  # camera ids
    # [X1, X2, X3, X4, X5, ....                    , Xn]  # points ids
    n_cam = len(registered_cameras)
    n_points = len(X_all)
    n_observations = len(camera_indices)

    # Generate sparsity matrix and indices
    jac_sparsity = create_sparsity_matrix(n_cam, n_points, n_observations, camera_indices, point_3d_indices, {initial_camera})  # (8, 21)

    result = least_squares(compute_residuals, initial_guess, jac_sparsity=jac_sparsity, verbose=verbose,
                           x_scale='jac', ftol=tol, method='trf',
                           args=(n_cam, n_points, camera_indices, point_3d_indices, point_2ds, K))

    # Camera parameters are the first n_cam * 6 elements
    camera_params = result.x[:n_cam * 6].reshape((n_cam, 6))

    for n, params in zip(registered_cameras.keys(), camera_params):
        euler_angles = params[:3]
        T = params[3:]
        R = Rot.from_rotvec(euler_angles).as_matrix()
        G.nodes[n]['H'] = H_from_RT(R, T)

    # Remaining elements are the 3D point coordinates
    X_all.data = result.x[n_cam * 6:].reshape((n_points, 3))
