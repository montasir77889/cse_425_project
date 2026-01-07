import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os

def evaluate_and_save(X, z):
    os.makedirs("results/tables", exist_ok=True)

    X_pca = PCA(n_components=16).fit_transform(X)

    labels_pca = KMeans(5, random_state=42).fit_predict(X_pca)
    labels_vae = KMeans(5, random_state=42).fit_predict(z)

    df = pd.DataFrame({
        "Method": ["PCA + KMeans", "VAE + KMeans"],
        "Silhouette Score": [
            silhouette_score(X_pca, labels_pca),
            silhouette_score(z, labels_vae)
        ],
        "Calinski-Harabasz Index": [
            calinski_harabasz_score(X_pca, labels_pca),
            calinski_harabasz_score(z, labels_vae)
        ]
    })

    df.to_csv("results/tables/easy_task_metrics.csv", index=False)
