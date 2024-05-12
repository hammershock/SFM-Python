import os
import warnings
from builtins import len
from collections import defaultdict
from itertools import product
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
import networkx as nx

from .graph import mark_edge_constructed
from .io import image_extensions
from .features import extract_sift
from .cache_utils import memory, serialize_graph
from .utils import timeit
from .transforms import H_from_RT, Homogeneous2Euler, Euler2Homogeneous, H_from_rtvec


def triangulate_points(P1, P2, pts1, pts2):
    """使用OpenCV的triangulatePoints函数三角化点对"""
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    # 转换为齐次坐标
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T


def calc_angle(vec1, vec2):
    dot_product = np.sum(vec1 * vec2, axis=0)
    norms = np.linalg.norm(vec1, axis=0) * np.linalg.norm(vec2, axis=0)
    cosine_angle = dot_product / norms
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


@timeit
@memory.cache
def build_graph(image_dir, K):
    G = nx.DiGraph()

    image_files = [filename for filename in os.listdir(image_dir) if filename.endswith(image_extensions)]
    # generate vertices
    for idx, filename in enumerate(tqdm(image_files, desc='extracting sift features')):
        if filename.endswith(image_extensions):
            filepath = os.path.join(image_dir, filename)
            image = cv2.imread(filepath)
            kps, desc = extract_sift(image)
            G.add_node(idx, kps=kps, desc=desc, image=image, constructed={})

    # generate edges
    G = generate_edges(G, K)

    return serialize_graph(G)  # notice: we should transform cv2.DMatch into dict to pickle


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


@timeit
def compute_tracks(G: nx.DiGraph, min_track_len=3) -> nx.DiGraph:
    """
    created node_data["tracks"] (dict[set[tuple[int, int]])

    created edge_data["tracks"] (dict[set[tuple[int, int]]])
    create edge_data["mask_enough_tracks"] (np.array)
    """
    # create empty tracks
    for node, data in G.nodes(data=True):
        data["tracks"] = defaultdict(set)

    # step 1: compute the track of nodes
    for node, node_data in G.nodes(data=True):  # for each vertex of the Graph:
        # compute the track of key points of the vertex ...
        for neighbor in G.neighbors(node):  # from the matches with its neighbours
            edge_data = G[node][neighbor]
            for match in edge_data['matches']:
                assert isinstance(match, cv2.DMatch)
                i, j = match.queryIdx, match.trainIdx
                node_data['tracks'][i].add((neighbor, j))
                G.nodes[neighbor]['tracks'][j].add((node, i))

    # step 2: compute the track of edges
    for u, v, data in G.edges(data=True):
        n1, n2 = G.nodes[u], G.nodes[v]
        data["tracks"] = {}

        # 具有足够长度的轨迹track
        data["mask_enough_tracks"] = np.zeros(len(data["matches"]), dtype=bool)
        for n, match in enumerate(data['matches']):
            assert isinstance(match, cv2.DMatch)
            i, j = match.queryIdx, match.trainIdx
            data["tracks"][(i, j)] = n1['tracks'][i] | n2['tracks'][j]
            if len(data["tracks"][(i, j)]) >= min_track_len:
                data["mask_enough_tracks"][n] = True

    return G


def triangulate_edge(G, K, edge, H1=None, H2=None):
    """
    输入: 两个相机的位姿
    输出: 三维世界坐标

    G: 共视图
    K: 3 * 3 相机内参
    edge: tuple[int, int] 三角化的边
    H1: 4 * 4 相机外参1
    H2: 4 * 4 相机外参2
    """
    u, v = edge
    edge_data = G[u][v]

    # Essential Matrix Decomposition.
    pts1 = np.float32([G.nodes[u]['kps'][m.queryIdx].pt for m in edge_data['matches']])  # p2d in matches of image u
    pts2 = np.float32([G.nodes[v]['kps'][m.trainIdx].pt for m in edge_data['matches']])  # p2d in matches of image v

    if H1 is None or H2 is None:
        # initialize camera pose with Essential Matrix Decomposition
        _, R, T, mask = cv2.recoverPose(edge_data['E'], pts1, pts2, K)
        mask = mask.astype(bool).flatten()
        H1 = np.eye(4)  # build the coord on the first Image
        H2 = H_from_RT(R, T)

    # projection matrices
    M1, M2 = K @ H1[:3], K @ H2[:3]

    # triangulate points
    X3d_H = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)  # (4, N)
    X3d_E = Homogeneous2Euler(X3d_H)  # (3, N)
    X3d_H = Euler2Homogeneous(X3d_E)  # (4, N)

    # Check if points are in front of both cameras
    # Transform points back to each camera coordinate system
    P1 = np.linalg.inv(H1) @ X3d_H
    P2 = np.linalg.inv(H2) @ X3d_H

    # Create masks where Z values are positive in both camera coordinate systems
    mask = (P1[2, :] > 0) & (P2[2, :] > 0)
    edge_data['mask_inliers'] = mask

    return M1, M2, H1[:3], H2[:3], X3d_E, mask


@timeit
def initial_register(G, K):
    def mid_angle(edge):
        M1, M2, H1, H2, X3d_E, mask = triangulate_edge(G, K, edge)

        # calculate the median angle, select the best edge to initialize.
        O1 = -np.linalg.inv(M1[:, :3]) @ M1[:, 3]
        O2 = -np.linalg.inv(M2[:, :3]) @ M2[:, 3]
        ray1 = X3d_E - O1[:, np.newaxis]
        ray2 = X3d_E - O2[:, np.newaxis]
        angles = calc_angle(ray1, ray2)
        angle = np.median(angles)
        angle = angle if 10 <= angle < 60 else 60
        return angle, H1, H2, X3d_E, mask, edge

    # choose the best edge from the Graph according to the median angle...
    angle, H1, H2, X3d, mask, (u, v) = min((mid_angle(edge) for edge in G.edges), key=lambda x: x[0])

    assert angle < 60, 'failed to find an edge to init.'
    print(f'Initial Register Complete! medium angle: {angle}')

    # Register Initial two Cameras.登记初始两个相机的位姿
    G.nodes[u]['H'] = H1
    G.nodes[v]['H'] = H2

    edge_data = G[u][v]
    # 标记这个边为：已经使用过
    edge_data['dirty'] = True  # this edge has been used!

    # 对两幅图像的所有成功重建的匹配点，标记他们的track为：已被重建过
    pairs = [(match.queryIdx, match.trainIdx) for match, available in zip(edge_data['matches'], mask) if available]
    for n, (i, j) in enumerate(pairs):
        mark_edge_constructed(G, (u, v), i, j, n)

    return X3d.T[mask]


def select_edge(G: nx.DiGraph) -> Tuple[int, int, float]:
    """
    选择一条边，使得含有track中已经被重建过的比例最大
    """
    # TODO: is it correct? 还是两个图像中出现过的keypoint已经被重建的比例最大？
    def votes(u, v, data):
        cnt = sum(any(i in G.nodes[n]["constructed"] for n, i in track_set) for track_set in data["tracks"].values())
        return cnt / len(data["tracks"]), (u, v)

    ratio, (u, v) = max((votes(u, v, data) for u, v, data in G.edges(data=True) if not G[u][v].get('dirty')), key=lambda x: x[0])
    return u, v, ratio


def apply_increment(G, K, X3d, increment_mask=None):
    if increment_mask is None or len(increment_mask) == 0:
        increment_mask = [0] * len(X3d)

    # select the best edge to begin.
    u, v, ratio = select_edge(G)
    print(f'choose {(u, v)}, votes: {ratio}')

    # data structure of PnP
    pt3ds = {'left': [], 'right': []}
    pt2ds = {'left': [], 'right': []}

    # Data Structure of Triangulation
    pts1 = []
    pts2 = []

    # 对于这条边所有的配对
    for match in G[u][v]['matches']:
        i, j = match.queryIdx, match.trainIdx

        pt3d_left_idx = G.nodes[u]['constructed'].get(i, None)
        pt3d_right_idx = G.nodes[v]['constructed'].get(j, None)

        # 检查左侧点是否被重建过，如果三维坐标已经存在，则可以用于解算左相机位姿
        if pt3d_left_idx is not None:
            pt3ds['left'].append(X3d[pt3d_left_idx])
            pt2ds['left'].append(G.nodes[u]['kps'][i].pt)

        # 检查右侧点是否已经被重建过，如果三维坐标已经存在，则可以用于解算右相机位姿
        if pt3d_right_idx is not None:
            pt3ds['right'].append(X3d[pt3d_right_idx])
            pt2ds['right'].append(G.nodes[v]['kps'][j].pt)

        # 在这里将其他匹配点三角化，补充新的三维点
        if pt3d_left_idx is None and pt3d_right_idx is None:
            # New Construction, new 3d point
            pts1.append(G.nodes[u]['kps'][i].pt)
            pts2.append(G.nodes[v]['kps'][j].pt)

    # 如果没有足够的三维点-二维点对用于解算PnP（少于6对），则直接结束，因为没有解算的相机位姿提供给三角化，将没有办法继续添加新产生的三维点
    if len(pt3ds['left']) < 6 or len(pt3ds['right']) < 6:  # not enough points to PnP
        G[u][v]['dirty'] = True  # This edge has been used! # 不要忘记标记这个边已经用过了不能重复使用
        return X3d, increment_mask

    # 这里使用PnP解算出左右两个相机的位姿
    retval1, r_vec_left, t_vec_left = cv2.solvePnP(np.array(pt3ds['left']), np.array(pt2ds['left']), K, np.zeros((1, 5)))
    retval2, r_vec_right, t_vec_right = cv2.solvePnP(np.array(pt3ds['right']), np.array(pt2ds['right']), K, np.zeros((1, 5)))
    H1 = H_from_rtvec(r_vec_left, t_vec_left)
    H2 = H_from_rtvec(r_vec_right, t_vec_right)

    # 将解出来的位姿用于三角化，添加新的三维点
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    if len(pts2) > 0 and len(pts1) > 0:
        M1, M2, H1, H2, X3d_new, visible_mask = triangulate_edge(G, K, (u, v), H1=H1, H2=H2)
        X3d_new = X3d_new.T[visible_mask]
        # Register new two Cameras.
        G.nodes[u]['H'] = H1
        G.nodes[v]['H'] = H2
    else:
        warnings.warn('no more point3ds added...')
        return X3d, increment_mask

    # 标记新的三角化出来的三维点的track为“已经被重建出来”
    edge_data = G[u][v]
    k = len(X3d)
    pairs = [(match.queryIdx, match.trainIdx) for (match, available) in zip(G[u][v]['matches'], visible_mask) if available]
    for n, (i, j) in enumerate(pairs):
        mark_edge_constructed(G, (u, v), i, j, n + k)

    # 将新的三维点附加在原来的后面
    increment_mask.extend([increment_mask[-1] + 1] * len(X3d_new))
    X3d = np.vstack((X3d, X3d_new))

    G[u][v]['dirty'] = True  # This edge has been used!
    return X3d, increment_mask
