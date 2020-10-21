import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getacc(X,Y,w):
    yhat = sigmoid(np.dot(X, w))
    y_array = Y.reshape(-1)
    yhat_array = yhat.reshape(-1)
    right = 0
    for i in range(y_array.size):
        if ((yhat_array[i] >= 0.5) and (y_array[i] == 1)) or ((yhat_array[i] < 0.5) and (y_array[i] == 0)):
            right += 1
    accuracy = right / y_array.size * 100
    print('Accuracy:' + str(accuracy) + '%')
    return accuracy