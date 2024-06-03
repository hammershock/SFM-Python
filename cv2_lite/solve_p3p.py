import numpy as np


def solve_p3p(points_2d, points_3d, K):
    """
    解P3P,返回候选的4组解
    Solve the P3P problem using three 2D-3D point correspondences.

    Parameters:
    points_2d (ndarray): Array of 2D points (3x2).
    points_3d (ndarray): Array of corresponding 3D points (3x3).
    K (ndarray): Camera intrinsic matrix (3x3).

    Returns:
    list: List of possible rotation matrices (3x3).
    list: List of possible translation vectors (3x1).
    """

    def calculate_distances(points_3d):
        d12 = np.linalg.norm(points_3d[0] - points_3d[1])
        d23 = np.linalg.norm(points_3d[1] - points_3d[2])
        d31 = np.linalg.norm(points_3d[2] - points_3d[0])
        return d12, d23, d31

    # Step 1: Normalize 2D points
    points_2d_hom = np.hstack((points_2d, np.ones((points_2d.shape[0], 1))))
    normalized_points = (np.linalg.inv(K) @ points_2d_hom.T).T[:, :2]

    # Step 2: Calculate distances between 3D points
    d12, d23, d31 = calculate_distances(points_3d)

    # Step 3: Calculate cosines of angles between the lines connecting the normalized points
    cos_alpha = np.dot(normalized_points[1], normalized_points[2])
    cos_beta = np.dot(normalized_points[0], normalized_points[2])
    cos_gamma = np.dot(normalized_points[0], normalized_points[1])

    # Step 4: Use P3P equations to solve for distances from the camera to the points
    a = (d12 ** 2 / d23 ** 2) * ((cos_beta - cos_alpha * cos_gamma) / (1 - cos_alpha ** 2))
    b = (d12 ** 2 / d31 ** 2) * ((cos_gamma - cos_alpha * cos_beta) / (1 - cos_alpha ** 2))
    c = (d12 ** 2 - a - b) / (1 - cos_alpha ** 2)

    if a < 0 or b < 0 or c < 0:
        return []  # No real results

    q = np.sqrt(a)
    r = np.sqrt(b)
    s = np.sqrt(c)

    # Step 5: Generate possible results
    results = []
    for sign1, sign2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        A = sign1 * q * normalized_points[0]
        B = sign2 * r * normalized_points[1]
        C = s * normalized_points[2]

        # Constructing the 3x3 rotation matrix
        R_approx = np.vstack((A, B, C)).T
        R = np.vstack([R_approx, np.cross(R_approx[0], R_approx[1])])

        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R

        t = points_3d[0] - R @ np.append(normalized_points[0], 1)
        tvec = t[:, np.newaxis]
        results.append((R, tvec))

    return results

