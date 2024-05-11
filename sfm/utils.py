import datetime
import time


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
