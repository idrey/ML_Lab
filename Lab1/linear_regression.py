import numpy as np
import datagen as dg
import matplotlib.pyplot as plt
import curveplot as cp
import time

start = time.perf_counter()

lamb = 0.001  # 正则化项系数
dim = 12  # 阶数
n = 60  # 数据点数
x_array, y = dg.getdata(0, 1, n, dim)
Xt = np.transpose(x_array)
XtX = np.dot(Xt, x_array)
Xty = np.dot(Xt, y)

w = np.linalg.solve(XtX + lamb * np.eye(dim + 1), Xty)

x_pre = x_array[:, 1:2]

cp.curveplot(w)
plt.title("dimension:" + str(dim) + " " + "lambda:" + str(lamb) + " " + "n:" + str(n))
plt.plot(x_pre, y, 'o', label='Training data')  # 实际数据点
# plt.plot(x_point, y_cos, 'o', label='Validating data')  # 实际数据点
plt.legend()
plt.show()

end = time.perf_counter()

print("Time:", end - start)

