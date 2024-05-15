import logging
import os
from collections import defaultdict
from itertools import product
from typing import List, Tuple, Sequence, Generator, Optional, Iterator

import cv2
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d.art3d import _path_to_3d_segment
from tqdm import tqdm

from .features import extract_sift
from .io import image_extensions, load_image_gray
from .metrics import calc_angle
from .structure import generate_visibility_mask
from .transforms import H_from_RT, normalize_homogeneous


class X3D:
    def __init__(self, G):
        self.G: nx.DiGraph = G
        self.data = np.empty((0, 3))
        self._increment_id = 0
        self.increment_mask: list = []
        self._colors: list = []
        self._tree = defaultdict(list)

    def add_points(self, X3d_new):
        self.data = np.vstack((self.data, X3d_new))
        self.increment_mask.extend([self._increment_id] * len(X3d_new))
        self._increment_id += 1

    def add_colors(self, colors):
        self._colors.extend(colors)

    @property
    def colors(self):
        return np.array(self._colors)

    def add_track(self, x3d_index, camera_id, feature_id, x, y):
        self._tree[x3d_index].append((camera_id, feature_id, x, y))

    def points2d(self, registered_cameras):
        point_2ds = []
        camera_indices = []
        point_3d_indices = []

        for idx, track_lst in self._tree.items():
            for cam_id, feature_id, x, y in track_lst:
                if cam_id in registered_cameras:
                    point_2ds.append(np.array([x, y]))
                    camera_indices.append(cam_id)
                    point_3d_indices.append(idx)

        point_2ds = np.array(point_2ds)
        camera_indices = np.array(camera_indices)
        point_3d_indices = np.array(point_3d_indices)
        return point_2ds, camera_indices, point_3d_indices

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Node:
    """
    A node of the CoVisMap, which means an Image or a camera view
    """
    def __init__(self, filepath: str):
        image = load_image_gray(filepath)
        kps, self.desc = extract_sift(image)
        self.pts = np.array([kp.pt for kp in kps])
        self.colors = np.array([image[y, x] for x, y in self.pts.astype(int)])

        self.tracks = defaultdict(set)
        self.is_initial_camera = False
        self.H = None  # camera pose
        self.constructed = {}

    def set_initial_camera(self):
        self.is_initial_camera = True
        self.H = np.eye(4)

    def set_camera_pose(self, H):
        self.H = H

    def mark_constructed(self, feat_idx, pt3d_idx):
        self.constructed[feat_idx] = pt3d_idx

    def is_constructed(self, feat_idx):
        return feat_idx in self.constructed

    def num_constructs(self):
        return len(self.constructed)


class Edge:
    def __init__(self, src: int, dst: int, F, E, pairs):
        self.src = src
        self.dst = dst
        self.F = F
        self.E = E
        self.pairs = pairs

        self.tracks = None
        self.dirty = False
        self.angle = None

    def __repr__(self):
        return f'Edge({self.src} - {self.dst})'

    def vertices(self, G) -> Tuple[Node, ...]:
        n1 = G.nodes[self.src]['data']
        n2 = G.nodes[self.dst]['data']
        return n1, n2

    def pts(self, G) -> Tuple[np.ndarray, np.ndarray]:
        n1, n2 = self.vertices(G)
        pts1 = n1.pts[self.pairs[:, 0]]
        pts2 = n2.pts[self.pairs[:, 1]]
        return pts1, pts2

    def constructed_mask(self, G):
        mask = []
        n1, n2 = self.vertices(G)
        for i, j in self.pairs:
            mask.append([n1.is_constructed(i), n2.is_constructed(j)])
        return np.array(mask)

    def triangulate(self, G, K, H1=None, H2=None, mask=None):
        if H1 is None or H2 is None:
            n1, n2 = self.vertices(G)
            H1, H2 = n1.H, n2.H
            assert H1 is not None and H2 is not None

        pts1, pts2 = self.pts(G)
        if mask is not None:
            pts1 = pts1[mask]
            pts2 = pts2[mask]

        # triangulate points
        M1, M2 = K @ H1[:3], K @ H2[:3]  # projection matrix
        X3d_H = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)  # (4, N)
        X3d_H /= X3d_H[-1]
        vis_mask = generate_visibility_mask(H1, H2, X3d_H)
        X3d = X3d_H[:3, :]
        return X3d.T[vis_mask]

    def tracker_nodes(self, pair: Tuple[int, int], G: nx.DiGraph) -> Iterator[Tuple[Node, int]]:
        for cam_id, feat_idx in self.tracks[tuple(pair)]:
            node: Node = G.nodes[cam_id]['data']
            yield node, cam_id, feat_idx

    def mid_angle(self, G, K):
        pts1, pts2 = self.pts(G)
        # initialize camera pose with Essential Matrix Decomposition
        _, R, T, mask = cv2.recoverPose(self.E, pts1, pts2, K)

        H1 = np.eye(4)
        H2 = H_from_RT(R, T)
        # triangulate points
        X3d = self.triangulate(G, K, H1, H2)  # (N, 3)
        M1, M2 = (K @ H1[:3])[:, :3], (K @ H2[:3])[:, :3]  # projection matrix

        # calculate the median angle, select the best edge to initialize.
        O1 = -np.linalg.inv(M1) @ M1
        O2 = -np.linalg.inv(M2) @ M2
        ray1 = X3d - O1[np.newaxis, :]
        ray2 = X3d - O2[np.newaxis, :]
        angle = np.median(calc_angle(ray1, ray2))
        self.angle = angle

        return angle if 10 <= angle < 60 else 60

    def vote(self, G: nx.DiGraph):
        sum = 0
        for pair in self.pairs:
            sum += any(node.is_constructed(feat_idx) for node, cam_id, feat_idx in self.tracker_nodes(pair, G))
        return sum

    def mark_dirty(self):
        self.dirty = True
