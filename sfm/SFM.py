from typing import Optional

import numpy as np

from .pipline import build_graph, compute_tracks, initial_register, apply_increment, apply_bundle_adjustment


class SFM(object):
    def __init__(self, image_dir, K, callback_group: Optional[dict] = None):
        self.K = K
        self.increment_mask = []

        self.G = build_graph(image_dir, K)
        self.G = compute_tracks(self.G)
        self.callback_group = callback_group

    def reconstruct(self, use_ba=False, ba_tol=1e-10, verbose=0):
        X3d = initial_register(self.G, self.K)  # (N, 3)
        if self.callback_group and "after_init" in self.callback_group:
            self.callback_group["after_init"](X3d.data)

        while True:
            ret, X3d = apply_increment(self.G, self.K, X3d, min_ratio=0.05)
            if self.callback_group and "after_increment" in self.callback_group:
                self.callback_group["after_increment"](X3d.data)

            if use_ba:
                X3d = apply_bundle_adjustment(self.G, self.K, X3d, tol=ba_tol, verbose=verbose)
                if self.callback_group and "after_ba" in self.callback_group:
                    self.callback_group["after_ba"](X3d.data)
            if not ret:
                break

        # for u, v, data in self.G.edges(data=True):
        #     display_edge_depth(self.G, (u, v), self.K)
        print(f'reconstruction complete!')
        return X3d.data, np.array(X3d.colors)

