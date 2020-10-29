import numpy as np
import matplotlib.pyplot as plt
import random
import datagen as dg


def init_points(x, k):
    data_size, data_dim = x.shape
    numbers = random.sample(range(0, data_size - 1), k)
    center_points = np.empty([k, data_dim])
    for i in range(k):
        center_points[i] = x[numbers[i], :]
    return center_points


def clustering(x, center_points):
    data_size, data_dim = x.shape
    k = center_points.shape[0]
    distance = np.empty([data_size, k])
    for i in range(data_size):
        for j in range(k):
            distance[i][j] = np.linalg.norm(x[i] - center_points[j])
    cluster = np.argmin(distance, axis=1)
    return cluster


def calc_center_points(x, cluster, k):
    data_size, data_dim = x.shape
    new_center_points = np.zeros((k, data_dim))
    for i in range(data_size):
        for j in range(data_dim):
            new_center_points[cluster[i]][j] = new_center_points[cluster[i]][j] + x[i][j]
    new_center_points = new_center_points / data_size
    return new_center_points


def kmeans(x, k):
    global cluster
    center_points = init_points(x, k)
    for m in range(10000):
        cluster = clustering(x, center_points)
        new_center_points = calc_center_points(x, cluster, k)
        if (new_center_points == center_points).all():
            break
        center_points = new_center_points
    return cluster


if __name__ == '__main__':
    x, k = dg.getdata()
    cluster = kmeans(x, k)
    print(cluster)
    color = ['r', 'b', 'y']
    for i in range(x.shape[0]):
        plt.scatter(x[i, 0], x[i, 1], marker='.', color=color[cluster[i]])
    plt.show()
