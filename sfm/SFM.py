from .pipline import build_graph, compute_tracks, initial_register, apply_increment


class SFM(object):
    def __init__(self, image_dir, K):
        self.image_dir = image_dir
        self.K = K
        self.increment_mask = []

        self.G = build_graph('./ImageDataset_SceauxCastle/images', K)
        self.G = compute_tracks(self.G)

    def reconstruct(self, return_colors=False):
        X3d, colors = initial_register(self.G, self.K)  # (N, 3)

        while True:
            ret, X3d, increment_mask, colors = apply_increment(self.G, self.K, X3d,
                                                               increment_mask=self.increment_mask,
                                                               colors=colors,
                                                               min_ratio=0.05)
            if not ret:
                break

        if return_colors:
            return X3d, colors
        return X3d
