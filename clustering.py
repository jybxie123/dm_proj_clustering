import utils.method as m
import utils.dataset as d
import utils.plot as p

CLUSTERING_METHODS = {
    'kmeans': m.kmeans,
    'kmeans_plus_plus': m.kmeans_plus_plus,
    'dbscan': m.dbscan,
}

DATASETS = {
    'iris': d.get_iris,
    'wine': d.get_wine,
    'digits': d.get_digits,
}


class Clustering():
    def __init__(self, method: str, dataset: str, c_num: int, eps: float, min_samples: int) -> None:
        self.dataset_name = dataset
        self.dataset = DATASETS[dataset]()
        self.method_name = method
        self.method = CLUSTERING_METHODS[method](self.dataset,c_num=c_num,eps=eps, min_samples=min_samples)
        self.plot = p.plot_dataset

    def train(self):
        model = self.method
        X = self.dataset.data  # sepal length 和 sepal width
        if hasattr(model, 'predict'):
            y = model.predict(X)
        else:
            y = model.labels_
        if hasattr(model, 'cluster_centers_'):
            centers = model.cluster_centers_
        else:
            centers = None
        self.plot(X, y, centers, ds_name=self.dataset_name, method_name=self.method_name, is_dim_reduced=True)

