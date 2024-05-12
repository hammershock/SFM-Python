import numpy as np


def calc_angle(vec1, vec2):
    dot_product = np.sum(vec1 * vec2, axis=0)
    norms = np.linalg.norm(vec1, axis=0) * np.linalg.norm(vec2, axis=0)
    cosine_angle = dot_product / norms
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle