import logging
import os
import time
from itertools import product
from pathlib import Path
from typing import Optional

from scipy.optimize import least_squares

import cv2
import joblib
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

from .transforms import H_from_RT, H_from_rtvec, RT_from_H
from .visualize import visualize_points3d
from .graph import Graph, Node, Edge
from .utils import timeit
from .bundle_adjustment import create_sparsity_matrix, compute_residuals
from cv2_lite.solve_pnp import reproj_error

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

memory = joblib.Memory(CACHE_DIR, verbose=2)

# from cv2_lite import recoverPose, triangulatePoints, findFundamentalMat, solvePnP


@memory.cache
def _sfm_build_graph(image_dir, K, min_matches=80):
    graph = Graph()
    graph = SFM._load_images(graph, image_dir)  # load images, extract features
    graph = SFM._match_features(graph, K, min_matches=min_matches)  # match features
    return graph


def calc_reproj_error(points3d, points2d, K, R, tvec):
    errors = reproj_error(points3d, points2d, K, R, tvec)
    err = np.linalg.norm(errors, axis=1).mean()
    return err


class SFM:
    _sift = cv2.SIFT_create()
    _bf = cv2.BFMatcher(cv2.NORM_L2)

    def __init__(self, image_dir, K):
        self.graph: Optional[Graph] = None
        self.image_dir = image_dir
        self.K = K

    @timeit
    def construct(self, min_matches=80, use_ba=False, ba_tol=1e-10, verbose=2, callback=None, interval=0.0):
        self.graph = _sfm_build_graph(self.image_dir, self.K, min_matches=min_matches)
        self.graph = self._build_tracks()  # build tracks
        self._initial_register(verbose=verbose)  # initial register

        for i in range(len(self.graph.edges)):
            result = self._select_edge()
            if result is None:
                logging.info(f'finished after fusing {i + 1} edges')
                break
            edge, _, (pt3ds_l, pt2ds_l), (pt3ds_r, pt2ds_r) = result
            self._apply_increment(edge, pt3ds_l, pt2ds_l, pt3ds_r, pt2ds_r, verbose=verbose)
            if use_ba:
                self._apply_bundle_adjustment(tol=ba_tol, verbose=verbose)
            if callback:
                callback()
                time.sleep(interval)
            logging.info(f'edge {i} finished.')

    @timeit
    @staticmethod
    def _load_images(graph, image_dir):
        """load images from directory and add them to the graph"""
        image_extensions = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')
        image_filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(image_extensions)]
        for filename in tqdm(image_filenames, "loading images"):
            image_path = os.path.join(image_dir, filename)
            node = Node()
            node.load_image(image_path, SFM._sift)
            graph.add_node(node)
        return graph

    @timeit
    @staticmethod
    def _match_features(graph, K, min_matches=80):
        """match features of each edge"""
        full_edges = [(u, v) for u, v in product(graph._G.nodes, repeat=2) if u > v]

        for u, v in tqdm(full_edges, "matching features"):
            edge = Edge(u, v, graph)
            matches = SFM._bf.knnMatch(graph[u].desc, graph[v].desc, k=2)
            # apply Lowe's ratio test
            good_pairs = np.array([(m.queryIdx, m.trainIdx) for m, n in matches if m.distance < 0.5 * n.distance], dtype=int)
            if len(good_pairs) > 8:
                edge.set_pairs(good_pairs)
                pts1, pts2, _ = edge.pt2ds_pt2ds()
                # Estimate Fundamental Matrix
                F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)
                inlier_pairs = good_pairs[inlier_mask.ravel() > 0]  # inlier good_pairs
                logging.debug(f'{len(inlier_pairs)}/{len(good_pairs)}/{len(matches)}')
                if len(inlier_pairs) > min_matches:
                    edge.set_pairs(inlier_pairs, E=K.T @ F @ K, F=F)
                    graph.add_edge(u, v, edge)
        return graph

    @timeit
    def _build_tracks(self):
        """build tracks of each node"""
        for edge in tqdm(self.graph.edges, "building tracks"):
            n1, n2 = edge.nodes()
            for i, j in edge.pairs:
                n1.tracks[i].add((edge.v, j))
                n2.tracks[j].add((edge.u, i))
        return self.graph

    @timeit
    def _initial_register(self, verbose=0):
        """initial register the first edge and generate initial 3D Points"""
        # choose edge
        initial_edge = None
        best_angle = float('inf')
        initial_X3d = None
        initial_pairs = None
        initial_H2 = None
        for edge in tqdm(self.graph.edges, "choosing edge"):
            pts1, pts2, pairs = edge.pt2ds_pt2ds()
            # Essential Matrix Decomposition
            _, R, t, mask = cv2.recoverPose(edge.E, pts1, pts2, self.K)

            # Initial Pose
            H1 = np.eye(4)
            H2 = H_from_RT(R, t)

            # Triangulate points
            M1, M2 = self.K @ H1[:3], self.K @ H2[:3]  # projection matrix

            X3d_H = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)  # (4, N)
            X3d_H /= X3d_H[-1]
            visible_mask = mask.ravel() > 0

            X3d = X3d_H[:3].T[visible_mask]

            # calculate median angle
            O1 = -np.linalg.inv(M1[:, :3]) @ M1[:, 3]
            O2 = -np.linalg.inv(M2[:, :3]) @ M2[:, 3]
            ray1 = X3d - O1[np.newaxis, :]
            ray2 = X3d - O2[np.newaxis, :]

            # angle between two rays
            dot_product = np.sum(ray1 * ray2, axis=0)
            norms = np.linalg.norm(ray1, axis=0) * np.linalg.norm(ray2, axis=0)
            cosine_angle = dot_product / norms
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            mid_angle = np.median(angle)

            if verbose:
                err1 = calc_reproj_error(X3d, pts1[visible_mask], self.K, R=np.eye(3), tvec=np.zeros(3))
                err2 = calc_reproj_error(X3d, pts2[visible_mask], self.K, R=R, tvec=t)
                print(f"selecting initial edge: mid_angle: {mid_angle}mean reproj errs: {(err1+err2)/2}")

            if 3 < mid_angle < 60 and mid_angle < best_angle:
                best_angle = mid_angle
                initial_edge = edge
                initial_X3d = X3d
                initial_pairs = pairs[visible_mask]
                initial_H2 = H2

        # ensure the initial edge exists
        assert initial_edge is not None, "failed to find initial edge"

        # register camera
        n1, n2 = initial_edge.nodes()
        n1.register(np.eye(4), initial_cam=True)
        n2.register(initial_H2)

        # first construction
        initial_edge.construct_3d(initial_X3d, initial_pairs)

    def _select_edge(self):
        """select a remaining edge to reconstruct"""
        edges = self.graph.edges
        # select the best edge
        if len(edges) == 0:
            return

        best_edge = None
        best_score = 0
        map1, map2 = None, None
        for edge in self.graph.edges:
            n1, n2 = edge.nodes()
            map1_, map2_ = n1.pts3d_pts2d(), n2.pts3d_pts2d()
            score = min(len(map1_[0]), len(map2_[0])) / len(edge.pairs)
            if score > best_score:
                best_edge, best_score = edge, score
                map1, map2 = map1_, map2_

        # enough pairs to solve PnP
        if best_score >= 0.05 and len(map1[0]) > 6 and len(map2[0]) > 6:
            logging.debug(f'ratio: {best_score}')
            return best_edge, best_score, map1, map2

    def _apply_increment(self, edge, pt3ds_l, pt2ds_l, pt3ds_r, pt2ds_r, verbose=0):
        """triangulate new edge to construct 3D points incrementally """
        ret1, r1, t1 = cv2.solvePnP(pt3ds_l, pt2ds_l, self.K, np.zeros((1, 5)))
        ret2, r2, t2 = cv2.solvePnP(pt3ds_r, pt2ds_r, self.K, np.zeros((1, 5)))

        # Register camera
        n1, n2 = edge.nodes()
        n1.register(H_from_rtvec(r1, t1))
        n2.register(H_from_rtvec(r2, t2))

        # Triangulate Points
        pts1, pts2, pairs = edge.pt2ds_pt2ds()
        M1, M2 = self.K @ n1.H[:3], self.K @ n2.H[:3]
        X3d_H = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)  # (4, N)
        X3d_H /= X3d_H[-1]
        # Create visibility mask
        P1 = np.linalg.inv(n1.H) @ X3d_H
        P2 = np.linalg.inv(n2.H) @ X3d_H
        visibility_mask = (P1[2, :] > 0) & (P2[2, :] > 0)

        X3d = X3d_H[:3].T[visibility_mask]

        if verbose:
            err1 = calc_reproj_error(X3d, pts1[visibility_mask], self.K, *RT_from_H(n1.H))
            err2 = calc_reproj_error(X3d, pts2[visibility_mask], self.K, *RT_from_H(n2.H))
            print(f"incremental err: {(err1 + err2) / 2}")

        edge.construct_3d(X3d, pairs[visibility_mask])

        errs = 0.0
        for n, (_, point3d, cam_id, feat_id, point2d) in enumerate(self.graph.pt3ds_pt2ds()):
            point3d = point3d.reshape((1, 3))
            point2d = point2d.reshape((1, 2))
            errs += calc_reproj_error(point3d, point2d, self.K, *RT_from_H(self.graph[cam_id].H))

        if verbose:
            print(f"total mean reproj err: {errs}/{n}={errs/n}")

    def _apply_bundle_adjustment(self, tol=1e-10, verbose=2):
        """
        apply bundle adjustment,
        optimize all registered camera poses and 3D points constructed to minimize the reprojection error
        """
        data = [(idx, cam_id, point2d) for idx, _, cam_id, _, point2d in self.graph.pt3ds_pt2ds()]
        pt_indices, cam_indices, pt2ds = map(np.array, zip(*data))

        cameras_registered = [node for node in self.graph.nodes if node.registered]
        RTs = [RT_from_H(node.H) for node in cameras_registered]
        camera_map = {node.idx: i for i, node in enumerate(cameras_registered)}

        camera_params = [(Rot.from_matrix(R).as_rotvec(), T.flatten()) for R, T in RTs]
        camera_indices = np.array([camera_map[camera_idx] for camera_idx in cam_indices])
        initial_guess = np.hstack([np.hstack([e, t]).ravel() for e, t in camera_params] + [self.graph.X3d.ravel()])

        n_cam = len(RTs)
        n_points = len(self.graph.X3d)
        n_observations = len(pt_indices)

        # Generate sparsity matrix and indices
        jac_sparsity = create_sparsity_matrix(n_cam, n_points, n_observations, camera_indices, pt_indices)  # (8, 21)

        result = least_squares(compute_residuals, initial_guess, jac_sparsity=jac_sparsity, verbose=verbose,
                               x_scale='jac', ftol=tol, method='trf',
                               args=(n_cam, n_points, camera_indices, pt_indices, pt2ds, self.K))

        # Camera parameters are the first n_cam * 6 elements
        camera_params = result.x[:n_cam * 6].reshape((n_cam, 6))

        for cam_id, n in camera_map.items():
            params = camera_params[n]  # (6, )
            euler_angles = params[:3]
            T = params[3:]
            R = Rot.from_rotvec(euler_angles).as_matrix()
            self.graph[cam_id].H = H_from_RT(R, T)

        # Remaining elements are the 3D point coordinates
        self.graph.X3d = result.x[n_cam * 6:].reshape((n_points, 3))
