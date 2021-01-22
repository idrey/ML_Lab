import numpy as np
import matplotlib.pyplot as plt
import datagen as dg


def pca(x, k):
    x_mean = np.mean(x)
    x_norm = x - x_mean
    s_Cov = np.dot(np.transpose(x_norm), x_norm)
    eig_val, eig_vec = np.linalg.eig(s_Cov)
    index = np.argsort(-eig_val)
    index = index[0:k]
    pc = eig_vec[:, index]
    new_data = np.dot(np.dot(x - x_mean, pc), pc.T) + x_mean
    return new_data, pc, x_mean


def calc_psnr(original, compressed):
    mse = np.sqrt(np.mean((original - compressed) ** 2))
    return 20 * np.log10(255.0 / mse)


if __name__ == '__main__':
    mean = [2, 5]
    cov = [[3.0, 0], [0, 0.1]]
    data = np.random.multivariate_normal(mean, cov, 50)
    plt.ylim((3.5, 6))
    plt.scatter(data[:, 0:1], data[:, -1], marker='.', color='r')
    new_data, pc, mean = pca(data, 1)
    plt.scatter(new_data[:, 0:1], new_data[:, -1], marker='.', color='b')
    plt.show()
    # new_data, pc, mean = pca(dg.getdata(), 3)
    # print(pc)
