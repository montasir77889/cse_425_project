import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import Counter

def purity_score(y_true, y_pred):
    clusters = np.unique(y_pred)
    correct = 0
    for c in clusters:
        idx = np.where(y_pred == c)[0]
        labels = y_true[idx]
        correct += Counter(labels).most_common(1)[0][1]
    return correct / len(y_true)

def evaluate(y_true, y_pred):
    return {
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "Purity": purity_score(y_true, y_pred)
    }
