import os
from typing import Generator

import cv2
import numpy as np


image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')


def load_calibration_data(txt_path):
    """

    :param txt_path:
    :return:
    """
    # K, dist_coeffs = load_calibration_data('camera_info.json')
    # E = K.T @ F @ K
    K = []
    with open(txt_path, 'r') as f:
        for line in f:
            K.append(list(map(float, line.split())))
    K = np.array(K)
    assert K.shape == (3, 3), K.shape
    return K


if __name__ == "__main__":
    load_calibration_data(f'../ImageDataset_SceauxCastle/images/K.txt')
