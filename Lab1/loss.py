import cupy as cc


def getloss(x, y, w):
    """
    计算损失函数，为Xw - y的二范数
    :param x: X矩阵
    :param y: y向量
    :param w: 参数向量
    :return: 损失函数
    """
    re = cc.dot(x, w) - y
    loss = cc.linalg.norm(re)
    return loss
