import numpy as np
import matplotlib.pyplot as plt
import uci

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
train_X, train_Y, test_X,test_Y = uci.getucidata()
X = np.row_stack((train_X, test_X))
Y = np.row_stack((train_Y, test_Y))
_x, _y = X.shape

t_x1 = []
t_x2 = []
t_x3 = []
f_x1 = []
f_x2 = []
f_x3 = []

for i in range(_x):
    if Y[i][0] == 0:
        f_x1.append(X[i][0])
        f_x2.append(X[i][1])
        f_x3.append(X[i][2])
    elif Y[i][0] == 1:
        t_x1.append(X[i][0])
        t_x2.append(X[i][1])
        t_x3.append(X[i][2])

ax.scatter(t_x1, t_x2, t_x3, c='r', marker='o')
ax.scatter(f_x1, f_x2, f_x3, c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
