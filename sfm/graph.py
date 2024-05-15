from itertools import product
from typing import Tuple

import cv2
import networkx as nx
import numpy as np
from tqdm import tqdm


from .data_stucture import X3D


def mark_edge_constructed(G, X3d: X3D, edge, i, j, pos3d_index):
    """
    标记这条边中i-j匹配点所在的track已经被重建过,
    同时给P3d数据结构该三维点对应的所有二维点信息。
    """
    u, v = edge
    edge_data = G[u][v]
    for node_idx, feat_idx in edge_data["tracks"][(i, j)]:
        G.nodes[node_idx]["constructed"][feat_idx] = pos3d_index
        x, y = np.array(G.nodes[node_idx]["kps"][feat_idx].pt, dtype=int)
        X3d.add_track(pos3d_index, node_idx, feat_idx, x, y)


def add_to_color(G, u, v, i, j, colors):
    n1, n2 = G.nodes[u], G.nodes[v]
    x1, y1 = np.array(n1["kps"][i].pt, dtype=int)
    color1 = n1['image'][y1, x1, :]
    x2, y2 = np.array(n2["kps"][j].pt, dtype=int)
    color2 = n2['image'][y2, x2, :]
    color = (color1 + color2) / 2
    colors.append(color)


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
            print(f'RANSAC filtered: {len(inlier_matches)}/{len(good_matches)}/{len(matches)}')
            if len(inlier_matches) > min_matches:
                pairs = np.array([(m.queryIdx, m.trainIdx) for m in inlier_matches])
                G.add_edge(i, j, F=F, E=K.T @ F @ K, pairs=pairs, mask=mask, matches=inlier_matches)  # use pair

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

