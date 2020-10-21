import numpy as np
import datagen as dg
import loss
import random
import calcacc
import uci
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

alpha = 0.0001
lamb = 0.0
times = 10000
cur_loss = 100000

X, Y, X_test, Y_test = uci.getucidata()
n, dimension = X.shape
print(n, dimension)
vector1 = np.ones((n, 1))
X = np.column_stack((vector1, X))
w = np.zeros((dimension + 1, 1))

while times > 0:
    times -= 1
    w = (1 - alpha * lamb / n) * w - alpha / n * np.dot(np.transpose(X), sigmoid(np.dot(X, w)) - Y)
    preloss = loss.getloss(X, Y, w)
    print(preloss)
    if preloss > cur_loss:
        alpha = alpha / 2
    if preloss < 1e-7:
        break
    cur_loss = preloss
print(w)

test_data_size, _y = X_test.shape
test_vector1 = np.ones((test_data_size, 1))
X_test = np.column_stack((test_vector1, X_test))
calcacc.getacc(X_test, Y_test, w)



