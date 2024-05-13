import os
import warnings
import logging
from builtins import len
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import networkx as nx

from .data_stucture import X3D
from .graph import mark_edge_constructed, generate_edges, select_edge
from .io import image_extensions
from .features import extract_sift
from .cache_utils import memory, serialize_graph, restore_graph
from .metrics import calc_angle
from .structure import triangulate_edge
from .utils import timeit
from .transforms import H_from_rtvec, H_from_RT
from .bundle_adjustment import bundle_adjustment


@timeit
def build_graph(image_dir, K):
    G = build_graph_(image_dir, K)
    return restore_graph(G)  # restore the cv2.KeyPoint and cv2.DMatch from pickleable dict


@memory.cache
def build_graph_(image_dir, K):
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


@timeit
def initial_register(G, K) -> X3D:
    def mid_angle(edge):
        u, v = edge
        edge_data = G[u][v]

        # Essential Matrix Decomposition.
        pts1 = np.float32([G.nodes[u]['kps'][m.queryIdx].pt for m in edge_data['matches']])  # p2d in matches of image u
        pts2 = np.float32([G.nodes[v]['kps'][m.trainIdx].pt for m in edge_data['matches']])  # p2d in matches of image v

        # initialize camera pose with Essential Matrix Decomposition
        _, R, T, mask = cv2.recoverPose(edge_data['E'], pts1, pts2, K)
        H1 = np.eye(4)  # build the coord on the first Image  # (3, 4)
        H2 = H_from_RT(R, T)  # (3, 4)
        M1, M2 = K @ H1[:3], K @ H2[:3]

        X3d_E, mask = triangulate_edge(pts1, pts2, K, H1, H2)

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
    angle, H1, H2, points3d, mask, (u, v) = min((mid_angle(edge) for edge in G.edges), key=lambda x: x[0])

    assert angle < 60, 'failed to find an edge to init.'
    logging.info(f'Initial Register Complete! medium angle: {angle}')

    # Register Initial two Cameras.登记初始两个相机的位姿
    G.nodes[u]['H'] = H1
    G.nodes[v]['H'] = H2

    edge_data = G[u][v]
    # 标记这个边为：已经使用过
    edge_data['dirty'] = True  # this edge has been used!

    X3d = X3D(G)
    # 对两幅图像的所有成功重建的匹配点，标记他们的track为：已被重建过
    pairs = [(match.queryIdx, match.trainIdx) for match, available in zip(edge_data['matches'], mask) if available]
    colors = []
    for n, (i, j) in enumerate(pairs):
        mark_edge_constructed(G, X3d, (u, v), i, j, n)
        add_to_color(G, u, v, i, j, colors)

    X3d.add_points(points3d.T[mask])
    X3d.add_colors(colors)
    return X3d


def add_to_color(G, u, v, i, j, colors):
    n1, n2 = G.nodes[u], G.nodes[v]
    x1, y1 = np.array(n1["kps"][i].pt, dtype=int)
    color1 = n1['image'][y1, x1, :]
    x2, y2 = np.array(n2["kps"][j].pt, dtype=int)
    color2 = n2['image'][y2, x2, :]
    color = (color1 + color2) / 2
    colors.append(color)


@timeit
def apply_increment(G, K, X3d, min_ratio=0.05):
    # select the best edge to begin.
    u, v, ratio = select_edge(G)
    print(f'choose {(u, v)}, votes: {ratio}')

    # data structure of PnP
    pt3ds = {'left': [], 'right': []}
    pt2ds = {'left': [], 'right': []}

    # Data Structure of Triangulation
    pts1 = []
    pts2 = []
    pairs_constructed = []
    ret = not (all(G[u][v].get('dirty') for u, v in G.edges) or ratio < min_ratio)

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
            pairs_constructed.append((i, j))

    # 如果没有足够的三维点-二维点对用于解算PnP（少于6对），则直接结束，因为没有解算的相机位姿提供给三角化，将没有办法继续添加新产生的三维点
    if len(pt3ds['left']) < 6 or len(pt3ds['right']) < 6:  # not enough points to PnP
        G[u][v]['dirty'] = True  # This edge has been used! # 不要忘记标记这个边已经用过了不能重复使用
        return ret, X3d

    # 这里使用PnP解算出左右两个相机的位姿
    retval1, r_l, t_l = cv2.solvePnP(np.array(pt3ds['left']), np.array(pt2ds['left']), K, np.zeros((1, 5)))
    retval2, r_r, t_r = cv2.solvePnP(np.array(pt3ds['right']), np.array(pt2ds['right']), K, np.zeros((1, 5)))

    # 将解出来的位姿用于三角化，添加新的三维点
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # 如果存在没有被重建的多余的匹配点，就将他们三角化，添加到重建好的三维点集中
    if len(pts2) > 0 and len(pts1) > 0:
        H1 = H_from_rtvec(r_l, t_l)
        H2 = H_from_rtvec(r_r, t_r)

        X3d_new, visible_mask = triangulate_edge(pts1, pts2, K, H1, H2)

        X3d_new = X3d_new.T[visible_mask]
        # Register new two Cameras.
        G.nodes[u]['H'] = H1
        G.nodes[v]['H'] = H2

        # 标记新的三角化出来的三维点的track为“已经被重建出来”
        k = len(X3d)
        pairs = [pair for pair, available in zip(pairs_constructed, visible_mask) if available]

        colors = []
        for n, (i, j) in enumerate(pairs):
            mark_edge_constructed(G, X3d, (u, v), i, j, n + k)
            add_to_color(G, u, v, i, j, colors)

        # 将新的三维点附加在原来的后面
        X3d.add_points(X3d_new)
        X3d.add_colors(colors)

        G[u][v]['dirty'] = True  # This edge has been used!
    else:
        warnings.warn('no more point3ds added...')
    return ret, X3d


@timeit
def apply_bundle_adjustment(G: nx.DiGraph, K, X3d: X3D, tol=1e-10):
    # step 1; build the R_set and C_set
    # R_set, C_set: the pose of all cameras registered, from the vertex of the Graph
    R_set, C_set, node_indices = [], [], []
    for node, data in G.nodes(data=True):  # check if the node is registered
        H = data.get('H')  # with shape 4 * 4
        if H is not None:
            R = H[:3, :3]
            T = H[:3, 3:]
            R_set.append(R)
            C_set.append(T)
            node_indices.append(node)

    # apply bundle adjustment
    (optimized_R_set, optimized_C_set), optimized_points_3d, camera_ids = bundle_adjustment(X3d, R_set, C_set, K, node_indices, tol=1e-10)
    X3d.data = optimized_points_3d
    for camera_id, R, T in zip(camera_ids, optimized_R_set, optimized_C_set):
        G.nodes[camera_id]['H'] = H_from_RT(R, T)
    return X3d
