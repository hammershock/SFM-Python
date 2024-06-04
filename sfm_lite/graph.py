"""
Key Data Structure of SFM
including co-vis-map, map Node, map edge
provides simple and easy-to-use APIs
"""
from collections import defaultdict
from itertools import product

import cv2
import networkx as nx
import numpy as np
from typing import Union, overload, Tuple, Optional, Set, DefaultDict, Generator, List, Type


class Node:
    """camera node"""
    def __init__(self):
        self.idx: Optional[int] = None
        self.parent: Optional['Graph'] = None

        self.desc = np.empty((0, 128))  # (N, 128)
        self.pts = np.empty((0, 2))  # (N, 2)
        self.H = np.zeros((4, 4))
        self.registered = False
        self.image = None
        self.image_color = None
        self.is_initial_camera = False

        # {feat_idx: {(camera1_idx, feat_idx)}}
        self.tracks: DefaultDict[int, Set[Tuple[int, int]]] = defaultdict(set)
        self.feat2point_index = {}

    def register(self, H, *, initial_cam=False):
        self.registered = True
        self.H = H
        self.is_initial_camera = initial_cam
        if initial_cam:
            self.parent.initial_cam = self

    def load_image(self, image_path, extractor):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_color = cv2.imread(image_path)
        kps, self.desc = extractor.detectAndCompute(self.image, None)
        self.pts = np.array([kp.pt for kp in kps])

    def pts3d_pts2d(self):
        """for PnP use"""
        if len(self.feat2point_index) == 0:
            return np.empty((0, 2)), np.empty((0, 3))  # node has no points reconstructed
        feat_indices = np.array(list(self.feat2point_index.keys()))
        pt3d_indices = np.array(list(self.feat2point_index.values()))
        pt2ds = self.pts[feat_indices, :]
        pt3ds = self.parent.X3d[pt3d_indices, :]
        return pt3ds, pt2ds


class Edge:
    """matches edge"""
    n_constructed = 0

    def __init__(self, u, v, parent=None):
        self.u = u
        self.v = v
        self.parent: Optional['Graph'] = parent
        self.pairs = np.empty((0, 2), int)
        self._tracks = None
        self.dirty = False
        self.F = None
        self.E = None

    def set_pairs(self, pairs, E=None, F=None):
        self.pairs = pairs
        if E is not None:
            self.E = E
        if F is not None:
            self.F = F

    def nodes(self) -> Tuple[Node, Node]:
        return self.parent[self.u], self.parent[self.v]

    @property
    def tracks(self) -> DefaultDict[Tuple[int, int], Set[Tuple[int, int]]]:
        if self._tracks is None:
            assert len(self.pairs), "please match features before building tracks"
            n1, n2 = self.nodes()
            self._tracks = {(i, j): n1.tracks[i] | n2.tracks[j] for i, j in self.pairs}

        return self._tracks

    def pt2ds_pt2ds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """pt2ds1, pt2ds2, filtered pairs"""
        assert len(self.pairs), f'pairs should have set here!'
        filtered_pairs = []
        n1, n2 = self.nodes()
        for i, j in self.pairs:
            if i not in n1.feat2point_index and j not in n2.feat2point_index:
                filtered_pairs.append((i, j))
        filtered_pairs = np.array(filtered_pairs)
        return n1.pts[filtered_pairs[:, 0]], n2.pts[filtered_pairs[:, 1]], filtered_pairs

    def construct_3d(self, new_X3d, pairs):
        assert len(pairs) == len(new_X3d)
        n1, n2 = self.nodes()
        assert n1.registered and n2.registered
        k = len(self.parent.X3d)
        self.dirty = True

        for m, (i, j) in enumerate(pairs):
            for cam_id, feat_idx in self.tracks[(i, j)]:  # feat -> pt3d
                node = self.parent[cam_id]
                node.feat2point_index[feat_idx] = m + k
                x, y = node.pts[feat_idx].astype(int)
                self.parent.tracks[m + k].append((cam_id, feat_idx, x, y))
                self.parent.color_tree[m + k].append(node.image_color[y, x])

        self.parent.X3d = np.vstack((self.parent.X3d, new_X3d))
        self.parent.increment_mask.extend([Edge.n_constructed] * len(new_X3d))
        self.parent._color_mapping[Edge.n_constructed] = np.random.randint(255, size=(3, ))
        Edge.n_constructed += 1


class Graph:
    def __init__(self):
        self._G = nx.DiGraph()
        self._node_idx = 0
        self._color_mapping = {}

        self.X3d = np.empty((0, 3))
        self.increment_mask = []
        self.color_tree = defaultdict(list)
        self.tracks = defaultdict(list)
        self.initial_cam = None

    @overload
    def __getitem__(self, item: int) -> Node:
        ...

    @overload
    def __getitem__(self, item: Tuple[int, int]) -> Edge:
        ...

    def __getitem__(self, item: Union[int, Tuple[int, int]]) -> Union[Node, Edge]:
        if isinstance(item, int):
            return self._G.nodes[item]['data']
        elif isinstance(item, tuple):
            u, v = item[:2]

            return self._G[u][v]['data']

    def add_node(self, node: Node):
        node.idx = self._node_idx
        node.parent = self
        self._G.add_node(self._node_idx, data=node)
        self._node_idx += 1

    def add_edge(self, u, v, edge: Edge):
        edge.u, edge.v = u, v
        edge.parent = self
        self._G.add_edge(u, v, data=edge)

    @property
    def edges(self) -> List['Edge']:
        """return edges available"""
        return [data['data'] for u, v, data in self._G.edges(data=True) if not data['data'].dirty]

    @property
    def nodes(self) -> Generator[Node, None, None]:
        for node, data in self._G.nodes(data=True):
            yield data["data"]

    @property
    def colors(self):
        assert len(self.X3d) == len(self.color_tree)
        colors = np.array([np.max(self.color_tree[idx], axis=0) for idx in range(len(self.X3d))])
        return colors[:, [2, 1, 0]]

    @property
    def increment_colors(self):
        colors = [self._color_mapping[i] for i in self.increment_mask]
        return np.array(colors)

    @property
    def camera_poses(self):
        return [node.H for node in self.nodes if node.registered]


