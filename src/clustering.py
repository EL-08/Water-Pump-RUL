import numpy as np

# Split a cluster by the column with the highest variance
def split_cluster(X, indices):
    var = np.var(X, axis=0)
    dim = np.argmax(var)
    order = np.argsort(X[:, dim])
    mid = len(order) // 2

    left_X = X[order[:mid]]
    right_X = X[order[mid:]]

    left_indices = indices[order[:mid]]
    right_indices = indices[order[mid:]]

    return (left_X, left_indices), (right_X, right_indices)

# Recursively split until k clusters are created
def top_down_cluster(X, k=4):
    clusters = [(X, np.arange(len(X)))]

    while len(clusters) < k:
        idx = np.argmax([np.var(cluster[0]) for cluster in clusters])
        X_target, indices_target = clusters.pop(idx)

        left, right = split_cluster(X_target, indices_target)

        clusters.append(left)
        clusters.append(right)

    return clusters