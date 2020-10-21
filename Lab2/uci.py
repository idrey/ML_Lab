import numpy as np

def getucidata(filename='DataSet/Banknote.txt'):
    dataset_raw = np.loadtxt(filename, delimiter=",")
    np.random.shuffle(dataset_raw)
    _x, _y = dataset_raw.shape
    Y = dataset_raw[:, -1].reshape(_x, 1)
    X = dataset_raw[:, 0:_y - 1]
    train_size = int(0.7 * _x)
    train_X = X[0:train_size, :]
    train_Y = Y[0:train_size, :]
    # train_Y = train_Y - 1
    test_X = X[train_size:, :]
    test_Y = Y[train_size:, :]
    # test_Y = test_Y - 1

    return train_X, train_Y, test_X,test_Y

if __name__ == '__main__':
    getucidata()


