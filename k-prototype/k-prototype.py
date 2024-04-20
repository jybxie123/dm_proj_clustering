import pandas as pd
import numpy as np
import random
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

random.seed(42)
data = pd.read_csv('D:\\CS\\Data Science\\Data Mining\\project\\dm_proj_clustering\\k-prototype\\Automobile.csv')
data = data.dropna()
# unique_values = data.iloc[:, 0].unique()
# num_unique_values = len(unique_values)
# print(num_unique_values)

features = data.iloc[:, 5:].values  # Use the 6th column and afterward as data
labels = data.iloc[:, 0].values  # The 1th colum is the target label, which is "make".
categorical_indices = [5, 6, 12, 13, 15]  # The 6、7、13、14、16th column is catagorical data.

scaler = MinMaxScaler()
non_categorical_features = data.iloc[:, 5:].drop(data.columns[categorical_indices], axis=1).values
non_categorical_features_normalized = scaler.fit_transform(non_categorical_features)
categorical_features = data.iloc[:, categorical_indices].values
# print(categorical_features)
combined_features = np.concatenate((categorical_features, non_categorical_features_normalized), axis=1)


# features = data.iloc[:, 1:].values  # Use the 6th column and afterward as data
# labels = data.iloc[:, 0].values  # The t1h colum is the target label, which is "make".
# categorical_indices = [1, 2, 3, 4, 5, 6, 12, 13, 15]  # The 6、7、13、14、16th column is catagorical data.

# scaler = MinMaxScaler()
# non_categorical_features = data.iloc[:, 1:].drop(data.columns[categorical_indices], axis=1).values
# non_categorical_features_normalized = scaler.fit_transform(non_categorical_features)
# categorical_features = data.iloc[:, categorical_indices].values
# # print(categorical_features)
# combined_features = np.concatenate((categorical_features, non_categorical_features_normalized), axis=1)
# print("ok")
# print(features)
exception = 1
i = 0
while exception:
    try:
        print(i)
        i += 1
        kproto = KPrototypes(n_clusters=20, init='Cao', n_init=1) # If it fails, try again.
        clusters = kproto.fit_predict(combined_features, categorical=list(range(len(categorical_indices))))
        exception = 0
    except:
        pass


kmeans = KMeans(n_clusters=20, init='k-means++', random_state=0)
clusters_non_categorical = kmeans.fit_predict(non_categorical_features)

ari_non_categorical = adjusted_rand_score(labels, clusters_non_categorical)

ari = adjusted_rand_score(labels, clusters)
print("Adjusted Rand Index (ARI) =", ari)
print("Adjusted Rand Index (ARI) for non-categorical data with K-Means =", ari_non_categorical)