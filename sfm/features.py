import cv2


sift = cv2.SIFT_create()


def extract_sift(image):
    key_points, descriptors = sift.detectAndCompute(image, None)
    return key_points, descriptors
