import utils.method as m
import utils.dataset as d
import utils.plot as p

CLUSTERING_METHODS = {
    'kmeans': m.kmeans,
    'kmeans_plus_plus': m.kmeans_plus_plus,
}

DATASETS = {
    'iris': d.get_iris,
    'wine': d.get_wine,
    'digits': d.get_digits,
    'mall_cust': d.get_mall_cust,
    'blobs': d.get_blobs,
    'blobs_with_outlier': d.get_blobs_with_outlier,
    'engy_time' : d.get_engy_time,
    'two_diamonds': d.get_two_diamonds,
}

def calculate_accu(y_true, y_pred):
    if y_true is None:
        return None
    from sklearn import metrics
    adjusted_rand_index = metrics.adjusted_rand_score(y_true, y_pred)
    return adjusted_rand_index

class Clustering():
    def __init__(self, method: str, dataset: str, c_num: int) -> None:
        self.dataset_name = dataset
        self.dataset = DATASETS[dataset]()
        self.method_name = method
        self.method = CLUSTERING_METHODS[method](self.dataset,c_num=c_num)
        self.plot = p.plot_dataset

    def train(self):
        model = self.method
        X = self.dataset.data  # sepal length 和 sepal width
        if hasattr(model, 'predict'):
            y = model.predict(X)
        else:
            y = model.labels_
        # calculate accu
        y_true = self.dataset.target if hasattr(self.dataset, 'target') else None
        accu = calculate_accu(y_true, y)
        if accu is not None:
            print(f"Accuracy: {accu}")
        if hasattr(model, 'cluster_centers_'):
            centers = model.cluster_centers_
        else:
            centers = None
        self.plot(X, y, centers, ds_name=self.dataset_name, method_name=self.method_name, is_dim_reduced=False)

