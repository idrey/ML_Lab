import numpy as np
import matplotlib.pyplot as plt
import datagen as dg


def init_params(k, dim):
    # means = np.random.randint(-5, 15, size=(k, dim))
    means = np.random.rand(k, dim).reshape(k, dim)
    var = np.array([np.identity(dim)] * k)
    pi = np.ones((k, 1)) * (1.0 / k)
    return means, var, pi


def revise_var(var_k):
    dim = var_k.shape[0]
    delta = 1e-8
    for i in range(dim):
        var_k[i][i] = var_k[i][i] + delta
    return var_k


def calc_resp_molecular(pi, x, means, var):
    n, dim = x.shape
    k = means.shape[0]
    resp_molecular = np.empty((n, k))
    for i in range(n):
        for j in range(k):
            resp_molecular[i][j] = pi[j][0] * calc_prob(x[i], means[j], var[j])

    return resp_molecular


def calc_prob(x_n, means_k, var_k):
    D = x_n.shape[0]
    x_n_column = x_n.reshape((D, -1))
    means_k_column = means_k.reshape((D, -1))
    if np.linalg.det(var_k) == 0:
        revise_var(var_k)
    var_k_det = np.linalg.det(var_k)
    x_minus_mean = x_n_column - means_k_column
    x_minus_mean_t = np.transpose(x_minus_mean)
    var_k_inv = np.linalg.pinv(var_k)
    prob = np.exp(np.dot(np.dot(x_minus_mean_t, var_k_inv), x_minus_mean) * (-0.5))
    prob = prob / np.power(2 * np.pi, D / 2) / np.power(var_k_det, 0.5)
    return prob


def update_params(x, resp, N):
    k = resp.shape[1]
    n, dim = x.shape
    new_means = np.zeros((k, dim))
    # for j in range(k):
    #     for l in range(n):
    #         new_means[k] = new_means[k] + resp[l][k] * x[l]
    for j in range(k):
        new_means[j] = np.dot(resp[:, j].reshape(1, n), x)
    new_means = new_means / N
    new_var = np.empty((k, dim, dim))

    for j in range(k):
        x_minus_means = x - new_means[j]
        x_minus_means_t = np.transpose(x_minus_means)
        new_var[j] = np.dot(resp[:, j] * x_minus_means_t, x_minus_means) / N[j]
    new_pi = N / k
    return new_means, new_var, new_pi


def em(x, k):
    data_size, dim = x.shape
    means, var, pi = init_params(k, dim)
    pre_log_lld = 1e9
    resp = np.empty((data_size, k))
    t = 0
    for i in range(1000):
        resp_molecular = calc_resp_molecular(pi, x, means, var)
        log_lld = np.sum(np.log(np.sum(resp_molecular, axis=1)))
        print(log_lld)
        if np.abs(log_lld - pre_log_lld) < 1e-10:
            t += 1
        else:
            t = 0
        if t == 10:
            print("converged")
            break
        pre_log_lld = log_lld
        for j in range(data_size):
            resp[j] = resp_molecular[j] / np.sum(resp_molecular[j])
        N = np.sum(resp, axis=0).reshape(k, 1)
        means, var, pi = update_params(x, resp, N)
    cluster = np.argmax(resp, axis=1)
    return cluster


if __name__ == '__main__':
    x, k = dg.getdata()
    cluster = em(x, k)
    color = ['r', 'b', 'y']
    for i in range(x.shape[0]):
        plt.scatter(x[i, 0], x[i, 1], marker='.', color=color[cluster[i]])
    plt.show()

