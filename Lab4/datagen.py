import numpy as np


def getdata(size=100):
    mean = [1, 2, 3]
    cov = [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 0.2]]
    data = np.random.multivariate_normal(mean, cov, size)
    return data
