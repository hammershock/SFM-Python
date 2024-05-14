from collections import defaultdict
from typing import List, Tuple

import networkx as nx
import numpy as np


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
