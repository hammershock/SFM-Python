import numpy as np


def jacobian(func, x0, eps=1e-8, *args):
    n = x0.size
    f0 = func(x0, *args)
    m = f0.size
    J = np.zeros((m, n))
    for i in range(n):
        x1 = np.array(x0, dtype=float)
        x1[i] += eps
        f1 = func(x1, *args)
        J[:, i] = (f1 - f0) / eps
    return J


def least_squares(func, x0, args=(), max_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        f = func(x, *args)
        J = jacobian(func, x, 1e-8, *args)
        delta = np.linalg.lstsq(J, -f, rcond=None)[0]
        x += delta
        if np.linalg.norm(delta) < tol:
            break
    return {'x': x, 'fun': f, 'jac': J}


