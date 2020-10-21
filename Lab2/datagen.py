import random

import numpy as np


def getdata(data_size, data_loc, data_sca, dimension):
    x_list = []
    medium = np.sum(data_loc)
    y = np.empty((data_size, 1))
    for i in range(dimension):
        x_list.append(np.random.normal(loc=data_loc[i], scale=data_sca[i], size=(data_size, 1)))
    x = x_list[0]
    for i in range(1, dimension):
        x = np.column_stack((x, x_list[i]))
    for i in range(data_size):
        if (np.sum(x[i]) >= medium):
            y[i][0] = 1
        else:
            y[i][0] = 0
    # print(np.column_stack((x, y)))
    return x, y

def getcordata(data_mean, data_cov,data_size):
    x = np.random.multivariate_normal(data_mean, data_cov, data_size)
    medium = np.sum(data_mean)
    y = np.empty((data_size,1))
    for i in range(data_size):
        if (np.sum(x[i]) >= medium):
            y[i][0] = 1
        else:
            y[i][0] = 0
    # print(np.column_stack((x, y)))
    return x,y

if __name__ == '__main__':
    data_size = 10
    data_loc = [random.uniform(0,5.0) for i in range(4)]
    data_sca = [0.2, 0.2, 0.2, 0.2]
    dimension = 4
    getdata(data_size, data_loc, data_sca, dimension)
