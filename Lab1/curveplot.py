import matplotlib.pyplot as plt
import numpy as np

def curveplot(w, begin = 0, end = 1, num = 1000):
    """
    绘制拟合曲线
    :param w: 学习所得参数
    :param begin:
    :param end:
    :param num:
    :return:
    """
    func = np.poly1d(np.array(w.T)[0][::-1])
    x1 = np.linspace(begin, end, num)
    y1 = func(x1)
    plt.plot(x1, y1, label="Curve fitting")