from typing import Sequence

import cv2


class KeyPoint:
    def __init__(self, kp):
        self.pt = kp.pt
        self.x = kp.pt[0]            # x坐标
        self.y = kp.pt[1]            # y坐标
        self.size = kp.size          # 关键点的直径
        self.angle = kp.angle        # 关键点的角度
        self.response = kp.response  # 关键点的响应强度，由特征检测算法计算
        self.octave = kp.octave      # 关键点所在的图像金字塔的层级
        self.class_id = kp.class_id  # 关键点的类别ID

    def cv(self):
        """将自定义KeyPoint对象转换回cv2.KeyPoint对象"""
        return cv2.KeyPoint(self.x, self.y, self.size, self.angle, self.response, self.octave, self.class_id)

    def __repr__(self):
        return (f"KeyPoint(x={self.x:.2f}, y={self.y:.2f}, size={self.size:.2f}, "
                f"angle={self.angle:.2f}, response={self.response:.2f}, "
                f"octave={self.octave}, class_id={self.class_id})")


class DMatch:
    def __init__(self, match):
        self.queryIdx = match.queryIdx
        self.trainIdx = match.trainIdx
        self.imgIdx = match.imgIdx
        self.distance = match.distance

    def cv(self):
        return cv2.DMatch(self.queryIdx, self.trainIdx, self.imgIdx, self.distance)

    def __repr__(self):
        return f"DMatch({self.queryIdx}, {self.trainIdx}, {self.imgIdx}, {self.distance:.2f})"


class Vertex:
    def __init__(self, image, kps, descriptors):
        self.image = image
        self.kps = kps
        self.descriptors = descriptors


class Edge:
    def __init__(self, F, source_idx, target_idx, matches: Sequence[DMatch], pts1, pts2):
        self.F = F
        self.source_vertex_idx = source_idx
        self.target_vertex_idx = target_idx
        self.matches = matches
        self.pts1 = pts1
        self.pts2 = pts2

    def calc_E(self, K):
        self.E = K.T @ self.F @ K
