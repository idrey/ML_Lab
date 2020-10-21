import numpy as np
import matplotlib.pyplot as plt
import datagen as dg
import curveplot as cp
import time
start = time.perf_counter()
dim = 12  # 阶数
n = 60  # 数据点数
times = 60  # 迭代次数
m = dim + 1
x_array, y = dg.getdata(0, 1, n, dim)

xT = np.transpose(x_array)
A = np.dot(xT, x_array)
x = np.zeros(dim + 1).reshape(dim + 1, 1)
b = np.dot(xT, y)
ri = b - np.dot(A, x)  # 残差
di = np.copy(ri)
lamb = 0.00000000001
for i in range(times):
    alphai = np.sum(np.dot(np.transpose(ri), ri) / np.dot(np.dot(np.transpose(di), A), di))
    xi1 = (1 - alphai * lamb) * x+ alphai * di
    ri1 = ri - alphai * np.dot(A, di)
    betai1 = np.dot(np.transpose(ri1), ri1) / np.dot(np.transpose(ri), ri)
    di1 = ri1 + betai1 * di
    ri = np.copy(ri1)
    di = np.copy(di1)
    x = xi1

cp.curveplot(x)
x_pre = x_array[:, 1:2]
plt.title("Conjugate gradient dimension:" + str(dim) + " " + "n:" + str(n))
plt.plot(x_pre, y, 'o', label='Training data')
plt.show()
end = time.perf_counter()
print("Time:", end - start)