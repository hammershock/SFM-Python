import logging
import os
from collections import defaultdict
from typing import Generator

import cv2
import networkx as nx
import numpy as np
from tqdm import tqdm

from sfm.transforms import H_from_RT


class Graph:
    def __init__(self, image_dir, K):
        self._G = nx.DiGraph()
        self.image_dir = image_dir
        self.K = K

        self.X = np.empty((0, 3))
        self._tree = defaultdict(list)

    @property
    def edges(self) -> Generator['Edge', None, None]:
        for u, v in self._G.edges:
            yield self._G[u][v]['data']

    def build(self, min_matches=80):
        assert min_matches >= 8, "The Fundamental Matrix be estimated only when 8 more pairs are available"
        filenames = [filename for filename in os.listdir(self.image_dir) if filename.endswith(image_extensions)]
        for idx, filename in enumerate(tqdm(filenames)):
            filepath = os.path.join(self.image_dir, filename)
            self._G.add_node(idx, data=Node(filepath))

        combinations = [(i, j) for i, j in product(self._G.nodes, repeat=2) if i < j]

        bf = cv2.BFMatcher(cv2.NORM_L2)
        for i, j in tqdm(combinations, desc="matching desc"):
            n1: Node = self._G.nodes[i]['data']
            n2: Node = self._G.nodes[j]['data']

            matches = bf.knnMatch(n1.desc, n2.desc, k=2)
            matches = np.array(matches, dtype=object)

            # apply Lowe's ratio test
            pairs = np.array([(m.queryIdx, m.trainIdx) for m, n in matches if m.distance < 0.5 * n.distance], dtype=int)

            if len(pairs) < min_matches:
                continue

            pts1 = n1.pts[pairs[:, 0]]
            pts2 = n2.pts[pairs[:, 1]]

            # Estimate Fundamental Matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)
            inlier_pairs = pairs[mask.ravel() == 1]

            if len(inlier_pairs) < min_matches:
                continue

            edge = Edge(i, j, F=F, E=self.K.T @ F @ self.K, pairs=pairs)
            self._G.add_edge(i, j, data=edge)  # use pair

        # compute tracks
        self._compute_tracks()

    def construct(self):
        # initial register
        self._initial_register()

        while True:
            ...

    def _compute_tracks(self):
        for u, v in self._G.edges:
            edge: Edge = self._G[u][v]['data']
            for i, j in edge.pairs:
                n1, n2 = edge.vertices(self._G)
                n1.tracks[i].add((v, j))
                n2.tracks[j].add((u, i))

        # step 2: compute the track of edges
        for u, v in self._G.edges:
            edge: Edge = self._G[u][v]['data']
            n1, n2 = edge.vertices(self._G)
            edge.tracks = {(i, j): n1.tracks[i] | n2.tracks[j] for i, j in edge.pairs}

    def _initial_register(self) -> None:
        gen = (e for e in self.edges)
        init_edge = min(gen, key=lambda x: x.mid_angle(self._G, self.K))

        assert init_edge.angle < 60, "Failed in Initial Register"
        logging.info(f'Initial Register Complete! medium angle: {init_edge.angle}')

        pts1, pts2 = init_edge.pts(self._G)
        n1, n2 = init_edge.vertices(self._G)
        # initialize camera pose with Essential Matrix Decomposition
        _, R, T, mask = cv2.recoverPose(init_edge.E, pts1, pts2, self.K)
        n1.set_initial_camera()
        n2.set_camera_pose(H_from_RT(R, T))
        init_edge.mark_dirty()
        X3d = init_edge.triangulate(self._G, self.K)
        self.X = np.vstack((self.X, X3d))

        # mark rebuilt
        for n, pair in enumerate(init_edge.pairs[mask]):
            for node, cam_id, feat_idx in init_edge.tracker_nodes(pair, self._G):
                node.mark_constructed(feat_idx, n)
                x, y = node.pts[feat_idx]
                self._tree[n].append((cam_id, feat_idx, x, y))

    def _apply_increment(self):
        # TODO: finish this!
        edge = max((edge for edge in self.edges), key=lambda edge: edge.vote(self._G))
        logging.info(f'choose {edge}, votes: {edge.vote(self._G)}')
        pts1, pts2 = edge.pts(self._G)  # points2d
        constructed_mask = edge.constructed_mask(self._G)
        if np.any(np.sum(constructed_mask, axis=0) < 6):
            return

        n1, n2 = edge.vertices(self._G)
        self._tree