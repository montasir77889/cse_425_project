import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os

def cluster_and_save(z, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    clusters = KMeans(n_clusters=5, random_state=42).fit_predict(z)
    z_2d = TSNE(n_components=2, random_state=42).fit_transform(z)

    plt.figure(figsize=(8,6))
    plt.scatter(z_2d[:,0], z_2d[:,1], c=clusters, cmap="tab10")
    plt.title("VAE Latent Space Clustering (t-SNE)")
    plt.savefig(save_path, dpi=300)
    plt.close()
