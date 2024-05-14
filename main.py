"""
Incremental MultiView Structure From Motion
         ---- A simple Practice in Python

Requirements:
opencv-python
numpy
matplotlib
joblib
tqdm
networkx
scipy(for Bundle Adjustment)

@author: Hanmo Zhang
@email: zhanghanmo@bupt.edu.cn
"""
import argparse
from sfm import load_calibration_data, SFM
from sfm.visualize import visualize_edge, visualize_points3d, visualize_graph


def main(image_dir, calibration_file, use_ba, ba_tol, verbose):
    K = load_calibration_data(calibration_file)
    sfm = SFM(image_dir, K, callback_group={"after_ba": visualize_points3d})
    X3d, colors = sfm.reconstruct(use_ba=use_ba, ba_tol=ba_tol, verbose=verbose)
    visualize_points3d(X3d, s=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SFM reconstruction on a set of images.")
    parser.add_argument('--image_dir', type=str, default="./ImageDataset_SceauxCastle/images", help='Directory containing images for reconstruction.')
    parser.add_argument('--calibration_file', type=str, default="./ImageDataset_SceauxCastle/images/K.txt", help='File containing camera calibration data.')
    parser.add_argument('--use_ba', action='store_true', help='Whether to use bundle adjustment. Default is False.')
    parser.add_argument('--ba_tol', type=float, default=1e-10, help='Tolerance for bundle adjustment. Default is 1e-10.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose of output. Default is 0.')

    args = parser.parse_args()
    main(args.image_dir, args.calibration_file, args.use_ba, args.ba_tol, args.verbose)
