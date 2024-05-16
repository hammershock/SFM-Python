# SFM_Python
增量式多视图结构重建（Structure From Motion）——Python 实现的简单练习
——CV Course大作业
![Peek 2024-05-15 20-11](https://github.com/hammershock/SFM_Python/assets/109429530/ff11f797-2908-4f52-9696-47a0f6b7d1ff)

本项目实现了一个简单的增量式多视图结构重建（SfM）练习，主要目标是从一组二维图像中重建三维结构。该实现使用了多个库，包括 OpenCV、NumPy、Matplotlib、Joblib、TQDM、NetworkX 和用于束调整的 SciPy。

## 依赖库

要安装所需的库：

```bash
pip install opencv-python numpy matplotlib joblib tqdm networkx scipy
```

## 克隆仓库

该仓库包含一个 ImageDataset_SceauxCastle 数据集的子模块，这是运行示例代码所必需的。在克隆仓库时，请确保也克隆子模块。

要克隆包含子模块的仓库，请使用以下命令：

```bash
git clone --recurse-submodules git@github.com:hammershock/SFM_Python.git
```

或者，如果您已经克隆了不包含子模块的仓库，可以使用以下命令初始化并更新子模块：

```bash
git submodule update --init --recursive
```

## 使用方法

```bash
python main.py --image_dir <图像目录路径> --calibration_file <校准文件路径> [--min_matches <最小匹配对数>] [--use_ba] [--ba_tol <束调整容差>] [--verbose <输出详细级别>]
```

## 示例

```bash
python main.py --image_dir ./ImageDataset_SceauxCastle/images --calibration_file ./ImageDataset_SceauxCastle/images/K.txt --min_matches 80
```

```bash
python app.py
```
