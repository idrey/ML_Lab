import numpy as np
import random
import matplotlib.pyplot as plt

def getdata(begin, end, num, dim):
    """
    在一个区间内生成y = sin(2πx)函数的坐标
    :param begin: 区间左端点
    :param end: 区间右端点
    :param num: 生成数据点的数量
    :param dim: x的阶数
    :return: x的多项式Numpy矩阵和y坐标组成的列向量
    """
    x = np.linspace(begin,end, num)
    y = np.sin(2 * np.pi * x)
    mu = 0
    sigma = 0.016
    for i in range(x.size):
        x[i] += random.gauss(mu, sigma)
        y[i] += random.gauss(mu, sigma)

    x_array = np.zeros((num, dim + 1))

    for i in range(num):
        x_array[i][0] = 1
        for j in range(1, dim + 1):
            x_array[i][j] = x_array[i][j - 1] * x[i]
    y = np.array([y]).T
    return x_array, y

if __name__ == '__main__':
    x_array, y = getdata(0, 1, 40, 4)
    x = x_array[:, 1:2]
    plt.plot(x,y,'o')
    plt.show()

