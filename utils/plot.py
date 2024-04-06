import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# not implemented yet
# def dr_tsne(X):
#     perplexity_value = min(30, len(X) - 1)
#     tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
#     X_tsne = tsne.fit_transform(X)
#     return X_tsne

def dr_umap(X):
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation', random_state=42)
    X_umap = umap_model.fit_transform(X)
    return X_umap

DIMENSION_REDUCTION_METHODS = {
    # 't-SNE':dr_tsne, 
    'UMAP':dr_umap
    }

def plot_dataset(X_ori, y, centers_ori, ds_name="Iris", method_name = 'K-means', is_dim_reduced = False):
    if is_dim_reduced:
        umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation', random_state=42)
        X = umap_model.fit_transform(X_ori)
        centers = umap_model.transform(centers_ori) if centers_ori is not None else None
    else: 
        X = X_ori
        centers = centers_ori
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title(f"{method_name} Clustering on {ds_name} Dataset")
    plt.show()

