from typing import Tuple

import numpy as np

from .transforms import euclidean_to_homogeneous


def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    T = np.array([
        [1 / std, 0, -mean[0] / std],
        [0, 1 / std, -mean[1] / std],
        [0, 0, 1]
    ])
    pts_normalized = (pts - mean) / std
    return pts_normalized, T


def construct_matrix_A(pts1, pts2):
    return np.column_stack([
        pts2[:, 0] * pts1[:, 0], pts2[:, 0] * pts1[:, 1], pts2[:, 0],
        pts2[:, 1] * pts1[:, 0], pts2[:, 1] * pts1[:, 1], pts2[:, 1],
        pts1[:, 0], pts1[:, 1], np.ones(len(pts1))
    ])


def estimate_fundamental_matrix(pts1, pts2) -> np.ndarray:
    if pts1.shape != pts2.shape:
        raise ValueError("Point sets must have the same shape")
    if pts1.shape[0] < 8:
        raise ValueError("At least 8 points are required to estimate the fundamental matrix")

    pts1_normalized, T1 = normalize_points(pts1)
    pts2_normalized, T2 = normalize_points(pts2)

    A = construct_matrix_A(pts1_normalized, pts2_normalized)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    return T2.T @ F @ T1


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
