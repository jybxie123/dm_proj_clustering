from sklearn.datasets import load_wine,load_digits
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

# # 加载Wine数据集
# wine = load_wine()
# X = wine.data

digits = load_digits()
X = digits.data

# 使用NearestNeighbors计算最近邻距离
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

# 对距离进行排序，并绘制k-距离图
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.xlabel('Points')
plt.ylabel('Distance')
plt.title('k-Distance Graph')
plt.show()