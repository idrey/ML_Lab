import gradient
import matplotlib.pyplot as plt

x = []
acc_20 = []
acc_60 = []
times = 20
for i in range(times):
    x.append(i)
    acc_60.append(gradient.gradient(1, data_size=50, lamb=0))
    acc_20.append(gradient.gradient(0, data_size=50, lamb=0))



plt.xlabel('times')
plt.ylabel('acc')
plt.ylim((0,100))
plt.plot(x,acc_20, marker='o', label='data_size = 50, idd')
plt.plot(x,acc_60, marker='o', label='data_size = 50, not-idd')
plt.legend()
plt.show()