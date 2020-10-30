import GMM_EM
import uci
import numpy as np

X, Y, k = uci.get_uci_data()
cluster = GMM_EM.em(X, k)
data_size = cluster.shape[0]
train_result = np.zeros(k)
data_result = np.zeros(k)
for i in range(data_size):
    train_result[cluster[i]] += 1
    data_result[Y[i]] += 1
print("Train result:")
print(train_result)
print("Data result:")
print(data_result)