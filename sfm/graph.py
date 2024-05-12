from itertools import product
from typing import Tuple

import cv2
import networkx as nx
import numpy as np
from tqdm import tqdm


def mark_edge_constructed(G, edge, i, j, pos3d_index):
    """
    标记这条边中i-j匹配点所在的track已经被重建过
    """
    u, v = edge
    edge_data = G[u][v]
    for node_idx, feat_idx in edge_data["tracks"][(i, j)]:
        G.nodes[node_idx]["constructed"][feat_idx] = pos3d_index


def generate_edges(G: nx.DiGraph, K, min_matches=80):
    """
    the key of the edge data: E, F, matches, mask
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)

    combinations = [(i, j) for i, j in product(G.nodes, repeat=2) if i < j]
    for i, j in tqdm(combinations, desc="matching key points"):
        v1, v2 = G.nodes[i], G.nodes[j]
        matches = bf.knnMatch(v1['desc'], v2['desc'], k=2)
        # apply Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
        # The Fundamental Matrix be estimated only when 8 more pairs are available
        if len(good_matches) > min_matches:
            pts1 = np.float32([v1['kps'][m.queryIdx].pt for m in good_matches])  # positions (N, 2)
            pts2 = np.float32([v2['kps'][m.trainIdx].pt for m in good_matches])  # positions (N, 2)
            # Estimate Fundamental Matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)
            inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]
            if len(inlier_matches) > min_matches:
                G.add_edge(i, j, F=F, E=K.T @ F @ K, matches=inlier_matches, mask=mask)

    return G


def select_edge(G: nx.DiGraph) -> Tuple[int, int, float]:
    """
    选择一条边，使得含有track中已经被重建过的比例最大
    """
    def votes(u, v, data):
        cnt = sum(any(i in G.nodes[n]["constructed"] for n, i in track_set) for track_set in data["tracks"].values())
        return cnt / len(data["tracks"]), (u, v)

    ratio, (u, v) = max((votes(u, v, data) for u, v, data in G.edges(data=True) if not G[u][v].get('dirty')), key=lambda x: x[0])
    return u, v, ratio

