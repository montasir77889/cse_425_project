import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_latent(z, labels, path):
    z2 = TSNE(n_components=2, random_state=42).fit_transform(z)
    plt.figure(figsize=(6,5))
    plt.scatter(z2[:,0], z2[:,1], c=labels, s=6)
    plt.savefig(path, dpi=300)
    plt.close()

def plot_distribution(labels, names, path):
    plt.figure()
    plt.bar(names, np.bincount(labels))
    plt.savefig(path, dpi=300)
    plt.close()
