import sklearn
import sklearn.datasets as ds
import pandas as pd
from sklearn.datasets import make_blobs
import numpy as np

class SimpleDataset:
    def __init__(self, data, target=None):
        self.data = data
        if target is not None:
            self.target = target

def get_iris():
    return ds.load_iris()

def get_digits():
    return ds.load_digits()

def get_wine():
    return ds.load_wine()

def get_mall_cust():
    url = "/Users/jiangyanbo/working/course/data_mining/project/dataset/Mall_Customers.csv"
    data = pd.read_csv(url)
    X = data.iloc[:, [3, 4]].values
    simple_x = SimpleDataset(X)
    return simple_x

def get_blobs():
    X, y = make_blobs(n_samples=1000, centers=3, cluster_std=0.5, random_state=0) 
    simple_x = SimpleDataset(X, y)
    return simple_x

def get_blobs_with_outlier():
    X, y = make_blobs(n_samples=1000, centers=3, cluster_std=0.60, random_state=0)
    np.random.seed(42) 
    n_outliers = 100
    outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
    outlier_labels = -1 * np.ones((n_outliers,))
    X = np.vstack([X, outliers])
    y = np.concatenate([y, outlier_labels])
    simple_x = SimpleDataset(X, y)
    return simple_x

def get_two_diamonds():
    url = "/Users/jiangyanbo/working/course/data_mining/project/dataset/TwoDiamonds.csv"
    data = pd.read_csv(url)
    X = data[['X1','X2']].values
    y = data['Class'].values
    simple_x = SimpleDataset(X,y)
    return simple_x

def get_engy_time():
    url = "/Users/jiangyanbo/working/course/data_mining/project/dataset/EngyTime.csv"
    data = pd.read_csv(url)
    X = data[['X1','X2']].values
    y = data['Class'].values
    simple_x = SimpleDataset(X,y)
    return simple_x

if __name__ == '__main__':
    get_two_diamonds()
    get_engy_time()
    print("done")