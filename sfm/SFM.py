from .pipline import build_graph, compute_tracks, initial_register, apply_increment, apply_bundle_adjustment
from .visualize import visualize_points3d


class SFM(object):
    def __init__(self, image_dir, K):
        self.K = K
        self.increment_mask = []

        self.G = build_graph(image_dir, K)
        self.G = compute_tracks(self.G)

    def reconstruct(self, use_ba=False):
        X3d = initial_register(self.G, self.K)  # (N, 3)

        while True:
            ret, X3d = apply_increment(self.G, self.K, X3d, min_ratio=0.05)
            # visualize_points3d(X3d.data, color_indices=X3d.increment_mask)
            if use_ba: X3d = apply_bundle_adjustment(self.G, self.K, X3d, tol=1e-10)
            visualize_points3d(X3d.data, colors=X3d.colors)
            if not ret: break

        print(f'reconstruct done!')
        return X3d.data, X3d.colors

