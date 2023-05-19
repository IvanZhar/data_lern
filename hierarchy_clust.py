import numpy as np


class HierarchyClust:
    def __init__(self, n_clusters, metric='euclidean', linkage='ward'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.clusters = {}

    def get_euclidean_clusters(self, x):
        dist_array = np.full((int(0.5 * (x.shape[0] ** 2 - x.shape[0])), 3), np.inf)
        k = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                if j <= i:
                    continue
                else:
                    dist_array[k] = np.array([i, j, np.linalg.norm(x[i] - x[j], axis=0)])
                    k += 1

        dist_array = dist_array[dist_array[:, -1].argsort()]
        for i in range(int(len(x) / 2)):
            self.clusters[frozenset({dist_array[0, 0], dist_array[0, 1]})] = dist_array[0, 2]
            dist_array = dist_array[
                np.logical_and(
                    dist_array[:, 0] != dist_array[0, 0],
                    dist_array[:, 1] != dist_array[0, 1]
                )
            ]



    def fit_predict(self, x):
        dist_mat = distance_matrix(x, x)


a = np.array([
    [1, 2],
    [2, 3],
    [1, 5],
    [2, 5],
    [4, 5],
    [5, 6],
    [4, 2],
    [5, 1]
])

hc = HierarchyClust(5)
hc.get_euclidean_clusters(a)
print(hc.clusters)

# a = {frozenset({1, 2}): 9, frozenset({1, 4}): 0, frozenset({2, 3}): 5, frozenset({2, 7}): 5}
# print(a.keys())
#
# print([a.get(key) for key in a.keys() if 1 in key])


