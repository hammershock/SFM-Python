from .pipline import build_graph, compute_tracks, initial_register, apply_increment, apply_bundle_adjustment
from .visualize import visualize_points3d


class SFM(object):
    def __init__(self, image_dir, K):
        self.image_dir = image_dir
        self.K = K
        self.increment_mask = []

        self.G = build_graph('./ImageDataset_SceauxCastle/images', K)
        self.G = compute_tracks(self.G)

    def reconstruct(self):
        X3d = initial_register(self.G, self.K)  # (N, 3)

        while True:
            ret, X3d = apply_increment(self.G, self.K, X3d, min_ratio=0.05)
            X3d = apply_bundle_adjustment(self.G, self.K, X3d)
            visualize_points3d(X3d.data, colors=X3d.colors)
            if not ret:
                break

        print(f'reconstruct done!')
        return X3d
