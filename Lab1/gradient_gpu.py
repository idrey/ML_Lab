import numpy as np
import datagen as dg
import matplotlib.pyplot as plt
import loss
import curveplot as cp
import cupy as cc
import time

start = time.perf_counter()
l_rate = 0.05  # 学习率
dim = 12  # 阶数
n = 60  # 数据点个数
lamb = 0  # 正则化项系数
times = 500000  # 迭代次数
preloss = 100000  # 损失，初始为一个充分大的值
x_array, y = dg.getdata(0, 1, n, dim)
w = np.zeros(dim + 1).reshape(dim + 1, 1)
Xt = np.transpose(x_array)

while times > 0:
    times -= 1
    xw = cc.dot(x_array, w)
    xw_y = xw - y
    tmp = (1 - lamb * l_rate / n) * w - l_rate / n * cc.dot(Xt, cc.dot(x_array, w) - y)
    w = tmp
    tmploss = loss.getloss(x_array, y, w)
    if tmploss == preloss:
        break
    preloss = tmploss
    print(preloss)
print("---")
print(w)

plt.title("l_rate:" + str(l_rate) + " " +"dimension:" + str(dim) + " " + "lambda:" + str(lamb) + " " + "n:" + str(n))
cp.curveplot(w,0,1)
x_pre = x_array[:, 1:2]
plt.plot(x_pre, y, 'o', label='Training data')
plt.show()
end = time.perf_counter()
print("Time:", end - start)