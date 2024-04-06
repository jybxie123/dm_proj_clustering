def kmeans(dataset, c_num, eps, min_samples):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=c_num)
    kmeans.fit(dataset.data)
    return kmeans


def kmeans_plus_plus(dataset, c_num, eps, min_samples):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=c_num, init='k-means++')
    kmeans.fit(dataset.data)
    return kmeans

def dbscan(dataset, c_num, eps, min_samples):
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(dataset.data)
    return dbscan
