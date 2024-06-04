import datetime

import numpy as np


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        duration = end_time - start_time

        # ANSI escape sequences for colors
        GREEN = '\033[92m'
        CYAN = '\033[96m'
        RESET = '\033[0m'
        # print(f"Function '{func.__name__}' executed in {duration.total_seconds():.6f}")
        print(f"\nFunction {CYAN}'{func.__name__}'{RESET} executed in {CYAN}{duration.total_seconds():.6f}{RESET} seconds", flush=True)
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
