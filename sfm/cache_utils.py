from pathlib import Path

import cv2
import joblib

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

memory = joblib.Memory(CACHE_DIR, verbose=0)


cv_keypoint_attributes = ['pt', 'angle', 'octave', 'response', 'size', 'class_id']
cv_d_match_attributes = ['distance', 'trainIdx', 'queryIdx', 'imgIdx']


def serialize_graph(G):
    def cv2_keypoint2dict(keypoint: cv2.KeyPoint):
        kp_dict = {attr: getattr(keypoint, attr) for attr in cv_keypoint_attributes}
        return kp_dict

    def cv2_d_match2dict(d_match: cv2.DMatch):
        dm_dict = {attr: getattr(d_match, attr) for attr in cv_d_match_attributes}
        return dm_dict

    for i in G.nodes:
        G.nodes[i]['kps'] = [cv2_keypoint2dict(kp) for kp in G.nodes[i]['kps']]

    for u, v in G.edges:
        G[u][v]['matches'] = [cv2_d_match2dict(dm) for dm in G[u][v]['matches']]

    return G


def restore_graph(G):
    def dict2cv2kp(kp_dict: dict):
        x, y = kp_dict['pt']
        del kp_dict['pt']
        return cv2.KeyPoint(x=x, y=y, **kp_dict)

    def dic2cv2dm(dm_dict: dict):
        params = {f'_{key}': value for key, value in dm_dict.items()}
        return cv2.DMatch(**params)

    for i in G.nodes:
        G.nodes[i]['kps'] = [dict2cv2kp(kp) for kp in G.nodes[i]['kps']]

    for u, v in G.edges:
        G[u][v]['matches'] = [dic2cv2dm(dm) for dm in G[u][v]['matches']]

    return G

