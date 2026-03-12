
import numpy as np

def split_cluster(X):
    var = np.var(X, axis=0)
    dim = np.argmax(var)
    order = np.argsort(X[:,dim])
    mid = len(order)//2
    return X[order[:mid]], X[order[mid:]]

def top_down_cluster(X, k=4):
    clusters = [X]

    while len(clusters) < k:
        idx = np.argmax([np.var(c) for c in clusters])
        target = clusters.pop(idx)
        left,right = split_cluster(target)
        clusters.append(left)
        clusters.append(right)

    return clusters
