import numpy as np
import matplotlib.pyplot as plt


def getdata(datasize = 100):
    mean1 = [-2, 2]
    mean2 = [6.5, 8]
    mean3 = [10, -2]
    # cov = [[1, 0.8], [0.8, 1]]
    cov = [[0.5, 0], [0, 0.5]]
    x1 = np.random.multivariate_normal(mean1, cov, size=datasize)
    x2 = np.random.multivariate_normal(mean2, cov, size=datasize)
    x3 = np.random.multivariate_normal(mean3, cov, size=datasize)
    x = np.row_stack((x1, x2))
    x = np.row_stack((x, x3))
    np.random.shuffle(x)
    return x, 3
    # return x1, x2, x3


# if __name__ == "__main__":
    # x1, x2, x3 = getdata()
    # plt.scatter(x1[:, 0:1], x1[:, -1], marker='.', color='r')
    # plt.scatter(x2[:, 0:1], x2[:, -1], marker='.', color='b')
    # plt.scatter(x3[:, 0:1], x3[:, -1], marker='.', color='y')
    # plt.show()
