# SFM-Python
Incremental MultiView Structure From Motion          ---- A simple Practice in Python

![opencv-python](https://img.shields.io/badge/opencv--python-4.9.0.80-blue)
![numpy](https://img.shields.io/badge/numpy-1.26.4-orange)
![networkx](https://img.shields.io/badge/networkx-3.3-yellow)
![tqdm](https://img.shields.io/badge/tqdm-4.66.4-green)
![matplotlib](https://img.shields.io/badge/matplotlib-3.8.4-red)
![joblib](https://img.shields.io/badge/joblib-1.4.2-purple)
![scipy](https://img.shields.io/badge/scipy-1.13.0-lightgrey)

[中文版文档](README_ZH.md)

![Peek 2024-05-15 20-11](https://github.com/hammershock/SFM_Python/assets/109429530/ff11f797-2908-4f52-9696-47a0f6b7d1ff)

This project implements a simple practice of Incremental MultiView Structure From Motion (SfM) in Python. The primary objective is to reconstruct 3D structures from a set of 2D images. The implementation utilizes several libraries including OpenCV, NumPy, Matplotlib, Joblib, TQDM, NetworkX, and SciPy for Bundle Adjustment.

## Requirements & Installation

To install the required libraries, you can use pip:

```bash
pip install opencv-python numpy matplotlib joblib tqdm networkx scipy
```

## Cloning the Repository

The repository contains a submodule for the ImageDataset_SceauxCastle dataset, which is necessary for running the example code. When cloning the repository, make sure to clone the submodule as well.

To clone the repository along with the submodule, use the following command:

```bash
git clone --recurse-submodules git@github.com:hammershock/SFM_Python.git
```

Alternatively, if you have already cloned the repository without the submodule, you can initialize and update the submodule using:

```bash
git submodule update --init --recursive
```

SceauxCastle Dataset：
![SceauxCastle Dataset](https://github.com/hammershock/SFM_Python/assets/109429530/30685b90-966f-4b18-bcc3-1518f86d2a11)

## Usage

To run the Structure From Motion reconstruction, use the following command:

```bash
python main.py --image_dir <path_to_image_directory> --calibration_file <path_to_calibration_file> [--min_matches <minimum_pairs_of_matches>] [--use_ba] [--ba_tol <bundle_adjustment_tolerance>] [--verbose <verbosity_level>]
```

### Arguments

- `--image_dir`: Directory containing images for reconstruction. (default: `./ImageDataset_SceauxCastle/images`)
- `--calibration_file`: File containing camera calibration data. (default: `./ImageDataset_SceauxCastle/images/K.txt`)
- `--min_matches`: Minimum pairs of matches. (default: 80)
- `--use_ba`: Whether to use bundle adjustment. (default: False)
- `--ba_tol`: Tolerance for bundle adjustment. (default: `1e-10`)
- `--verbose`: Verbosity level of output. (default: 0)

## Example

```bash
python main.py --image_dir ./ImageDataset_SceauxCastle/images --calibration_file ./ImageDataset_SceauxCastle/images/K.txt --min_matches 80
```

```bash
python app.py
```

## Author

Hanmo Zhang  
Email: zhanghanmo@bupt.edu.cn

## License

This project is licensed under the MIT License. See the LICENSE file for details.
