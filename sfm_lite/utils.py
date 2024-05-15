import datetime

import numpy as np


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()  # 获取开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = datetime.datetime.now()  # 获取结束时间
        duration = end_time - start_time  # 计算执行时间
        # 打印函数名和执行时间，包括分和秒
        print(f"Function '{func.__name__}' executed in {duration.total_seconds():.6f}")
        return result
    return wrapper


def load_calibration_data(txt_path):
    """
    :param txt_path:
    :return:
    """
    K = []
    with open(txt_path, 'r') as f:
        for line in f:
            K.append(list(map(float, line.split())))
    K = np.array(K)
    assert K.shape == (3, 3), K.shape
    return K
