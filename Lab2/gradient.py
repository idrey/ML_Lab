import numpy as np
import datagen as dg
import loss
import calcacc
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient(mode, data_size, lamb = 0.0):
    dimension = 4
    alpha = 0.1
    times = 100000
    cur_loss = 100000
    data_loc = [1, 2, 3, 4]
    data_sca = [0.2, 0.2, 0.2, 0.2]
    data_mean = [1, 2, 3, 4]
    data_cov = [[0.2, 0.1, 0.1, 0.1], [0.1, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1], [0.1, 0.1, 0.1, 0.2]]

    if mode == 0:
        X, Y = dg.getdata(data_size, data_loc, data_sca, dimension)
    else:
        X, Y = dg.getcordata(data_mean, data_cov, data_size)
    vector1 = np.ones((data_size, 1))
    X = np.column_stack((vector1, X))
    w = np.zeros((dimension + 1, 1))

    while times > 0:
        times -= 1
        w = (1 - alpha * lamb / data_size) * w - alpha / data_size * np.dot(np.transpose(X), sigmoid(np.dot(X, w)) - Y)
        preloss = loss.getloss(X, Y, w)
        print(preloss)
        if preloss > cur_loss:
            break
        cur_loss = preloss
    print(w)

    test_data_size = 100
    test_data_loc = [1, 2, 3, 4]
    test_data_sca = [0.2 for i in range(dimension)]
    test_data_mean = [1, 2, 3, 4]
    test_data_cov = [[0.2, 0.1, 0.1, 0.1], [0.1, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1], [0.1, 0.1, 0.1, 0.2]]

    if mode == 0:
        X_test, Y_test = dg.getdata(test_data_size, test_data_loc, test_data_sca, dimension)
    else:
        X_test, Y_test = dg.getcordata(test_data_mean, test_data_cov, test_data_size)

    test_vector1 = np.ones((test_data_size, 1))
    X_test = np.column_stack((test_vector1, X_test))
    return calcacc.getacc(X_test, Y_test, w)


if __name__ == '__main__':
    gradient(0, 30, 0.0001)

