from sklearn.decomposition import PCA

def compute_pca(X):
    pca=PCA(n_components=2)
    return pca.fit_transform(X)