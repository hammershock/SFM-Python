from sfm_lite import SFM
from sfm import load_calibration_data
from sfm.visualize import visualize_points3d

if __name__ == '__main__':
    K = load_calibration_data('./ImageDataset_SceauxCastle/images/K.txt')
    sfm = SFM("./ImageDataset_SceauxCastle/images", K)
    sfm.construct()
    visualize_points3d(sfm.graph.X3d)
