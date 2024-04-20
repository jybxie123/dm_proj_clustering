import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('D:\\CS\\Data Science\\Data Mining\\project\\dm_proj_clustering\\k-prototype\\Automobile.csv')
data = data.dropna()

features = data.iloc[:, 5:].values  # Use the 6th column and afterward as data
labels = data.iloc[:, 4].values  # The 5th colum is the target label, which is "body-style".
categorical_indices = [0, 1, 7, 8, 10]  # The 6、7、13、14、16th column is catagorical data.

scaler = MinMaxScaler()
non_categorical_features = data.iloc[:, 5:].drop(data.columns[[5, 6, 12, 13, 15]], axis=1).values
non_categorical_features_normalized = scaler.fit_transform(non_categorical_features)
categorical_features = data.iloc[:, categorical_indices].values
combined_features = np.concatenate((categorical_features, non_categorical_features_normalized), axis=1)

sse = []
for k in range(1, 20):  # Possible K
    kmeans = KMeans(n_clusters=11, init='k-means++', random_state=0)
    kmeans.fit_predict(non_categorical_features)
    sse.append(kmeans.inertia_)  # SSE for each K


plt.figure(figsize=(8, 6))
plt.plot(range(1, 20), sse, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.show()