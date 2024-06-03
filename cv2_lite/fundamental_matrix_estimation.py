from typing import Tuple

import numpy as np

from .transforms import euclidean_to_homogeneous


def estimate_fundamental_matrix(pts1, pts2) -> np.ndarray:
    assert pts1.shape == pts2.shape, "Point sets must have the same shape"
    N = pts1.shape[0]
    assert N >= 8, "At least 8 points are required to estimate the fundamental matrix"

    # Normalize points
    pts1_mean = np.mean(pts1, axis=0)
    pts2_mean = np.mean(pts2, axis=0)

    pts1_std = np.std(pts1)
    pts2_std = np.std(pts2)

    T1 = np.array([[1 / pts1_std, 0, -pts1_mean[0] / pts1_std],
                   [0, 1 / pts1_std, -pts1_mean[1] / pts1_std],
                   [0, 0, 1]])

    T2 = np.array([[1 / pts2_std, 0, -pts2_mean[0] / pts2_std],
                   [0, 1 / pts2_std, -pts2_mean[1] / pts2_std],
                   [0, 0, 1]])

    pts1_normalized = (pts1 - pts1_mean) / pts1_std
    pts2_normalized = (pts2 - pts2_mean) / pts2_std

    # Construct matrix A for Af = 0
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1_normalized[i]
        x2, y2 = pts2_normalized[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Solve for f using SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))

    # Denormalize F
    F = np.dot(T2.T, np.dot(F, T1))

    return F


def estimate_fundamental_matrix_ransac(pts1, pts2, method='RANSAC', threshold=1.0, confidence=0.99, maxIters=1000) -> \
Tuple[np.ndarray, np.ndarray, bool]:
    """
    Estimate the fundamental matrix using RANSAC to handle outliers.

    Args:
        pts1: A numpy array of shape (N, 2) containing N 2D points from the first image.
        pts2: A numpy array of shape (N, 2) containing N 2D points from the second image.
        method: Method for computing fundamental matrix (default 'RANSAC').
        threshold: The distance threshold to determine inliers.
        confidence: The confidence level, between 0 and 1, to keep inliers.
        maxIters: The maximum number of RANSAC iterations.

    Returns:
        F: A numpy array of shape (3, 3) representing the estimated fundamental matrix.
        inlier_mask: A numpy array of shape (N, ) with boolean values indicating the inliers.
        success: A boolean indicating if the estimation was successful.
    """
    assert pts1.shape == pts2.shape, "Point sets must have the same shape"
    N = pts1.shape[0]
    assert N >= 8, "At least 8 points are required to estimate the fundamental matrix"

    def generate_samples():
        for _ in range(maxIters):
            indices = np.random.choice(N, 8, replace=False)
            sample_pts1 = pts1[indices]
            sample_pts2 = pts2[indices]
            yield sample_pts1, sample_pts2

    def compute_score(pts1_sample, pts2_sample):
        F = estimate_fundamental_matrix(pts1_sample, pts2_sample)
        pts1_hom = euclidean_to_homogeneous(pts1)
        pts2_hom = euclidean_to_homogeneous(pts2)
        lines2 = np.dot(F, pts1_hom.T).T

        numerators = np.abs(np.sum(lines2 * pts2_hom, axis=1))
        denominators = np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2)
        distances = numerators / denominators

        inlier_mask = distances < threshold
        return F, inlier_mask

    scores_generator = (compute_score(pts1_sample, pts2_sample) for pts1_sample, pts2_sample in generate_samples())
    best_F, best_inlier_mask = max(scores_generator, key=lambda x: np.sum(x[1]))

    success = np.sum(best_inlier_mask) / N >= confidence
    return best_F, best_inlier_mask, success
