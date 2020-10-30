import numpy as np


def get_uci_data(filename='bezdekIris.data'):
    iris = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    a = np.loadtxt(filename, dtype=np.str, delimiter=',')
    np.random.shuffle(a)
    n, dim = a.shape
    X = a[:, 0:dim - 1]
    Y = a[:, -1]
    X = X.astype(np.float)
    for i in range(n):
        Y[i] = iris[Y[i]]
    Y = Y.astype(np.int)
    k = len(iris)
    return X, Y, k


if __name__ == '__main__':
    X, Y, k = get_uci_data()
