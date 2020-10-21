import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def getloss(X, Y, w):
    n = Y.size
    Yt = np.transpose(Y)
    s = sigmoid(np.dot(X,w))
    return - 1 / n * np.sum((np.dot(Yt, np.log(s)) + np.dot((1 - Yt), np.log(1 - s))))