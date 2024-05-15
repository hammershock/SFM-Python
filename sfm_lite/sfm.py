import logging
import os
from pathlib import Path
from typing import Optional

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from sfm.utils import timeit

import cv2
import joblib
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

from sfm.metrics import calc_angle
from sfm.transforms import H_from_RT, H_from_rtvec, RT_from_H
from sfm.visualize import visualize_points3d
from .graph import Graph, Node, Edge


CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

memory = joblib.Memory(CACHE_DIR, verbose=0)


@memory.cache
def _sfm_build_graph(image_dir, K, min_matches=80):
    graph = Graph()
    graph = SFM._load_images(graph, image_dir)  # load images, extract features
    graph = SFM._match_features(graph, K, min_matches=min_matches)  # match features
    return graph


def create_sparsity_matrix(n_cameras, n_points, n_obs, camera_indices, point3d_indices, fixed_camera_indices=()):
    """Create a sparsity structure for the Jacobian matrix used in optimization."""
    assert len(camera_indices) == len(point3d_indices)
    J = lil_matrix((n_obs * 2, n_cameras * 6 + n_points * 3), dtype=int)

    for i, (point_idx, cam_idx) in enumerate(zip(point3d_indices, camera_indices)):
        row = i * 2
        if cam_idx not in fixed_camera_indices:
            J[row:row + 2, cam_idx * 6:cam_idx * 6 + 6] = 1
        J[row:row + 2, n_cameras * 6 + point_idx * 3:n_cameras * 6 + point_idx * 3 + 3] = 1

    return J


def project_points(points, camera_params, K):
    """Project 3D points onto camera image planes using the camera parameters and intrinsic matrix K."""
    results = []

    for point, params in zip(points, camera_params):
        R = Rot.from_rotvec(params[:3]).as_matrix()
        T = params[3:]
        M = K @ np.hstack((R, -R @ T[:, np.newaxis]))

        projected_point = M @ np.append(point, 1)
        results.append(projected_point[:2] / projected_point[2])

    return np.array(results)


def compute_residuals(x, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals for the optimization."""
    # cut the vector into two parts
    camera_params = x[:n_cameras * 6].reshape((n_cameras, 6))  # (n_cam, 6)
    points_3d = x[n_cameras * 6:].reshape((n_points, 3))  # (n_point, 3)
    projected_points = project_points(points_3d[point_indices], camera_params[camera_indices], K)
    output = (projected_points - points_2d).ravel()
    return output


class SFM:
    _sift = cv2.SIFT_create()
    _bf = cv2.BFMatcher(cv2.NORM_L2)

    def __init__(self, image_dir, K):
        self.graph : Optional[Graph] = None
        self.image_dir = image_dir
        self.K = K

    @timeit
    def construct(self):
        self.graph = _sfm_build_graph(self.image_dir, self.K, min_matches=80)
        self.graph = self._build_tracks()  # build tracks
        self._initial_register()  # initial register

        for i in range(len(self.graph.edges)):
            result = self._select_edge()
            if result is None:
                logging.info(f'finished after fusing {i + 1} edges')
                break
            edge, _, (pt3ds_l, pt2ds_l), (pt3ds_r, pt2ds_r) = result
            self._apply_increment(edge, pt3ds_l, pt2ds_l, pt3ds_r, pt2ds_r)
            self._apply_bundle_adjustment(tol=1e-10, verbose=2)
            visualize_points3d(self.graph.X3d)
            logging.info(f'edge {i} finished.')

    @staticmethod
    def _load_images(graph, image_dir):
        image_extensions = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')
        image_filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(image_extensions)]
        for filename in tqdm(image_filenames, "loading images"):
            image_path = os.path.join(image_dir, filename)
            node = Node()
            node.load_image(image_path, SFM._sift)
            graph.add_node(node)
        return graph

    @staticmethod
    def _match_features(graph, K, min_matches=80):
        for u, v in tqdm(graph.full_edges(), "matching features"):
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
                if len(inlier_pairs) > min_matches:
                    edge.set_pairs(inlier_pairs, E=K.T @ F @ K, F=F)
                    graph.add_edge(u, v, edge)
        return graph

    def _build_tracks(self):
        for edge in tqdm(self.graph.edges, "building tracks"):
            n1, n2 = edge.nodes()
            for i, j in edge.pairs:
                n1.tracks[i].add((edge.v, j))
                n2.tracks[j].add((edge.u, i))
        return self.graph

    def _initial_register(self):
        # choose edge
        initial_edge = None
        best_angle = float('inf')
        initial_X3d = None
        initial_pairs = None
        initial_H2 = None
        for edge in self.graph.edges:
            pts1, pts2, pairs = edge.pt2ds_pt2ds()
            # Essential Matrix Decomposition
            _, R, T, mask = cv2.recoverPose(edge.E, pts1, pts2, self.K)

            # Initial Pose
            H1 = np.eye(4)
            H2 = H_from_RT(R, T)

            # Triangulate points
            M1, M2 = self.K @ H1[:3], self.K @ H2[:3]  # projection matrix

            X3d_H = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)  # (4, N)
            X3d_H /= X3d_H[-1]
            X3d = X3d_H[:3].T[mask.ravel() > 0]

            # calculate median angle
            O1 = -np.linalg.inv(M1[:, :3]) @ M1[:, 3]
            O2 = -np.linalg.inv(M2[:, :3]) @ M2[:, 3]
            ray1 = X3d - O1[np.newaxis, :]
            ray2 = X3d - O2[np.newaxis, :]
            angle = np.median(calc_angle(ray1, ray2))
            if 3 < angle < 60 and angle < best_angle:
                best_angle = angle
                initial_edge = edge
                initial_X3d = X3d
                initial_pairs = pairs
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
            print(f'ratio: {best_score}')
            return best_edge, best_score, map1, map2

    def _apply_increment(self, edge, pt3ds_l, pt2ds_l, pt3ds_r, pt2ds_r):
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
        mask = (P1[2, :] > 0) & (P2[2, :] > 0)

        edge.construct_3d(X3d_H[:3].T[mask], pairs[mask])

    def _apply_bundle_adjustment(self, tol=1e-10, verbose=2):
        pt_indices = []
        cam_indices = []
        pt2ds = []
        RTs = []
        camera_map = {}
        for point_idx, lst in self.graph.tracks.items():
            for cam_id, feat_idx, x, y in lst:
                pt_indices.append(point_idx)
                pt2ds.append([x, y])
                if cam_id not in camera_map:
                    camera_map[cam_id] = len(camera_map)
                    RTs.append(RT_from_H(self.graph[cam_id].H))
                    assert len(RTs) == len(camera_map)
                cam_indices.append(camera_map[cam_id])

        pt2ds = np.array(pt2ds)
        cam_indices = np.array(cam_indices)

        camera_params = [(Rot.from_matrix(R).as_rotvec(), T.flatten()) for R, T in RTs]

        initial_guess = np.hstack([np.hstack([e, t]).ravel() for e, t in camera_params] + [self.graph.X3d.ravel()])

        n_cam = len(camera_map)
        n_points = len(self.graph.X3d)
        n_observations = len(pt_indices)

        # Generate sparsity matrix and indices
        jac_sparsity = create_sparsity_matrix(n_cam, n_points, n_observations, cam_indices, pt_indices, {camera_map[self.graph.initial_cam.idx]})  # (8, 21)

        result = least_squares(compute_residuals, initial_guess, jac_sparsity=jac_sparsity, verbose=verbose,
                               x_scale='jac', ftol=tol, method='trf',
                               args=(n_cam, n_points, cam_indices, pt_indices, pt2ds, self.K))

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
