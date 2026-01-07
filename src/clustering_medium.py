from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def cluster_features(X, k=6):
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return labels, sil, ch
