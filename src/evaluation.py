import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

def evaluate(X, cluster_dict, true_labels):
    rows = []

    for name, labels in cluster_dict.items():
        rows.append({
            "Method": name,
            "Silhouette": silhouette_score(X, labels),
            "Davies-Bouldin": davies_bouldin_score(X, labels),
            "ARI": adjusted_rand_score(true_labels, labels)
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/medium_task_metrics.csv", index=False)
