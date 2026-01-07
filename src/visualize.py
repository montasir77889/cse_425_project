import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(X, labels, save_path):
    X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, s=8)
    plt.title("Medium Task: Audio + Lyrics Clustering")
    plt.savefig(save_path, dpi=300)
    plt.close()
