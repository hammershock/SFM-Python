# -*- coding: utf-8 -*-
"""
A simple Camera Calibration Tool using OpenCV

This script performs camera calibration using a chessboard pattern.
The pattern can be generated online: https://calib.io/pages/camera-calibration-pattern-generator
It allows for capturing multiple calibration images automatically at set intervals and
calculates the camera's intrinsic matrix and distortion coefficients. The results are saved to a JSON file.

Features:
- Automatic image capture at user-defined intervals.
- Real-time display of the camera feed.
- Audio feedback on successful chessboard detection.
- Outputs calibration data in JSON format.

Requirements:
- OpenCV library
- NumPy library
- pygame library for audio feedback
- tqdm: For displaying progress during batch operations.

Usage:
Run the script in a directory with a 'data' folder containing initial calibration images (optional).
Place the chessboard in view of the camera and ensure good lighting.
Press 'c' to manually capture images or let the automatic capture handle it.

@author: hammershock
@email: hammershock@163.com
@date: 2024.4.28
"""
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
import time


def video_stream(source):
    cam = cv2.VideoCapture(source)
    if not cam.isOpened():
        raise RuntimeError("Failed to open camera")
    try:
        while True:
            retval, frame = cam.read()
            if not retval:
                raise RuntimeError("Failed to grab frame")
            yield frame  # Yield the frame to the caller
    finally:
        cam.release()  # Ensure the camera is released when done or on exception


def tick(capture_interval=0, last_time=[0.0]):
    """here it is a trick, we use mutable argument as global variable, but it is not recommended to do so."""
    current_time = time.time()
    if capture_interval > 0 and (current_time - last_time[0] >= capture_interval):
        last_time.clear()
        last_time.append(current_time)
        return True
    return False


def calib(*, w, h, size_grid, capture_interval=0, data_dir='./data', save_path='camera_info.json'):
    global frame
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]
    results = [cv2.findChessboardCorners(img, (w, h), None) for img in tqdm(images)]
    points_pixel = [corner_point for ret, corner_point in results if ret]
    print(f'Found {len(points_pixel)} valid corners')

    # for frame in video_stream(0):
    #     gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('Camera Feed', frame)
    #     key = cv2.waitKey(1)
    #
    #     if key in [27, ord('q')]:  # exit the loop and start to calibrate
    #         break
    #
    #     elif key == ord('c') or tick(capture_interval):  # Trigger capture
    #         ret, corner_point_image = cv2.findChessboardCorners(gray_img, (w, h), None)
    #         if ret:
    #             points_pixel.append(corner_point_image)
    #             frame_output = frame.copy()
    #             cv2.drawChessboardCorners(frame_output, (w, h), corner_point_image, ret)
    #             print(f'{len(points_pixel)} images collected ')
    #             cv2.imwrite(os.path.join(output_dir, f'{len(points_pixel)}.png'), frame_output)
    #             cv2.imwrite(os.path.join(data_dir, f'{len(points_pixel)}.png'), frame)

    print(f'{len(points_pixel)} valid images found.')
    if len(points_pixel) > 0:
        corner_point = np.zeros((w * h, 3), np.float32)
        corner_point[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        corner_point *= size_grid

        corner_points = [corner_point] * len(points_pixel)
        # h_px, w_px, _ = frame.shape
        h_px, w_px = images[0].shape[:2]
        ret, camera_matrix, coff_dis, v_rot, v_trans = (
            cv2.calibrateCamera(corner_points, points_pixel, (w_px, h_px), None, None))
        # Save calibration data to JSON
        calibration_data = {
            "intrinsic_matrix": camera_matrix.tolist(),
            "distortion_coefficients": coff_dis.tolist(),
        }
        with open(save_path, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        print(calibration_data)
        return camera_matrix, coff_dis


if __name__ == "__main__":
    calib(w=11, h=8, size_grid=0.010, capture_interval=3, data_dir='./calibration_data', save_path='./camera_info_mate40.json')  # Capture every 3 seconds
