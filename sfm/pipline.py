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
from .transforms import H_from_rtvec, H_from_RT, RT_from_H
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
            pts = np.array([kp.pt for kp in kps])
            colors = np.array([image[y, x] for x, y in pts.astype(int)])
            G.add_node(idx, kps=kps, desc=desc, image=image, constructed={}, pts=pts, colors=colors)

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
            for i, j in edge_data["pairs"]:
                node_data['tracks'][i].add((neighbor, j))
                G.nodes[neighbor]['tracks'][j].add((node, i))

    # step 2: compute the track of edges
    for u, v, data in G.edges(data=True):
        n1, n2 = G.nodes[u], G.nodes[v]
        data["tracks"] = {(i, j): n1['tracks'][i] | n2['tracks'][j] for i, j in data["pairs"]}
        # filter tracks

    return G


@timeit
def initial_register(G, K) -> X3D:
    def mid_angle(edge):
        u, v = edge
        edge_data = G[u][v]
        n1, n2 = G.nodes[u], G.nodes[v]

        # Essential Matrix Decomposition.
        pts1 = np.array([n1['pts'][i] for i, _ in edge_data["pairs"]])
        pts2 = np.array([n2['pts'][j] for _, j in edge_data["pairs"]])

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
    angle, H1, H2, points3d, mask, edge = min((mid_angle(edge) for edge in G.edges), key=lambda x: x[0])

    assert angle < 60, 'failed to find an edge to init.'
    logging.info(f'Initial Register Complete! medium angle: {angle}')
    u, v = edge
    edge_data = G[u][v]
    n1, n2 = G.nodes[u], G.nodes[v]

    # Initial 3D points
    X3d = X3D(G)

    # register cameras
    n1['H'] = H1
    n2['H'] = H2
    n1['base'] = True
    # mark this edge has been used.
    edge_data['dirty'] = True

    # mark rebuilt
    mask_enough_tracks = np.array([len(edge_data["tracks"][(i, j)]) >= 3 for i, j in edge_data["pairs"]])
    pairs = edge_data["pairs"][mask]

    for n, (i, j) in enumerate(pairs):
        for cam_id, feat_idx in edge_data["tracks"][(i, j)]:
            G.nodes[cam_id]["constructed"][feat_idx] = n
            x, y = G.nodes[cam_id]['pts'][feat_idx]
            X3d.add_track(n, cam_id, feat_idx, x, y)

    X3d.add_points(points3d.T[mask])
    return X3d


@timeit
def apply_increment(G, K, X3d, min_ratio=0.05):
    # Select the best edge to begin.
    u, v, ratio = select_edge(G)
    edge_data = G[u][v]
    n1, n2 = G.nodes[u], G.nodes[v]
    logging.info(f'choose {(u, v)}, votes: {ratio}')

    # Masks for constructed points
    left_visible_mask = np.array([i in n1['constructed'] for i, _ in edge_data["pairs"]])
    right_visible_mask = np.array([j in n2['constructed'] for _, j in edge_data["pairs"]])

    left_2d_indices = np.array([i for i, _ in edge_data["pairs"][left_visible_mask]])
    right_2d_indices = np.array([j for _, j in edge_data["pairs"][right_visible_mask]])

    # If not enough points for PnP
    if len(left_2d_indices) < 6 or len(right_2d_indices) < 6:
        G[u][v]['dirty'] = True
        ret = not (all(G[u][v].get('dirty') for u, v in G.edges) or ratio < min_ratio)
        return ret, X3d

    pt2ds = {
        'left': n1['pts'][left_2d_indices],
        'right': n2['pts'][right_2d_indices]
    }

    left_3d_indices = np.array([n1['constructed'][i] for i, _ in edge_data["pairs"][left_visible_mask]])
    right_3d_indices = np.array([n2['constructed'][j] for _, j in edge_data["pairs"][right_visible_mask]])

    pt3ds = {
        'left': X3d.data[left_3d_indices],
        'right': X3d.data[right_3d_indices]
    }

    # Solve PnP
    retval1, r_l, t_l = cv2.solvePnP(pt3ds['left'], pt2ds['left'], K, np.zeros((1, 5)))
    retval2, r_r, t_r = cv2.solvePnP(pt3ds['right'], pt2ds['right'], K, np.zeros((1, 5)))

    # New construction
    mask = ~(left_visible_mask | right_visible_mask)
    pairs_constructed = np.array(edge_data['pairs'])[mask]
    left_unconstructed_indices = np.array([i for i, _ in pairs_constructed])
    right_unconstructed_indices = np.array([j for _, j in pairs_constructed])
    pt2d1 = n1['pts'][left_unconstructed_indices]
    pt2d2 = n2['pts'][right_unconstructed_indices]

    # Triangulate new 3D points
    if len(pt2d1) > 0 and len(pt2d2) > 0:
        H1 = H_from_rtvec(r_l, t_l)
        H2 = H_from_rtvec(r_r, t_r)

        X3d_new, visible_mask = triangulate_edge(pt2d1, pt2d2, K, H1, H2)
        X3d_new = X3d_new.T[visible_mask]

        # Register new cameras
        G.nodes[u]['H'] = H1
        G.nodes[v]['H'] = H2

        mask_enough_tracks = np.array([len(edge_data["tracks"][(i, j)]) >= 3 for i, j in pairs_constructed])
        pairs = pairs_constructed[visible_mask]

        # Mark new triangulated 3D points as constructed
        k = len(X3d)
        for n, (i, j) in enumerate(pairs):
            mark_edge_constructed(G, X3d, (u, v), i, j, n + k)

        X3d.add_points(X3d_new)
        G[u][v]['dirty'] = True
    else:
        warnings.warn('no more point3ds added...')

    ret = not (all(G[u][v].get('dirty') for u, v in G.edges) or ratio < min_ratio)
    return ret, X3d


@timeit
def apply_bundle_adjustment(G: nx.DiGraph, K, X3d: X3D, tol=1e-10, verbose=0):
    # apply bundle adjustment
    bundle_adjustment(G, X3d, K, tol=tol, verbose=verbose)

    return X3d
