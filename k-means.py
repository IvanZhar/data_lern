import numpy as np
import pandas as pd
import random


class KMeans:
    def __init__(self, k, repeat=10, loss='wcss'):
        self.k = k
        self.repeat=repeat
        self.loss = loss
        self.x_in_cluster_dict = {}
        self.centers = None
        self.clusters = None
        self.losses = []
        self.max_itiration = 100

    @staticmethod
    def _distance(arr1, arr2):
        assert arr1.shape[1] == arr2.shape[-1], 'arr1 colls nb != arr2 colls nb'
        return np.linalg.norm(arr1 - arr2, axis=1)

    def _wcss(self, clusters_centers):
        result = 0
        for cluster in self.x_in_cluster_dict:
            result += np.sum(self._distance(self.x_in_cluster_dict[cluster], clusters_centers[cluster]))
        return result

    def _get_new_centers(self, x):
        new_centers = np.zeros((self.k, x.shape[1]), dtype="float32")
        for i in range(self.k):
            x_in_cluster = x[self.clusters == i]
            self.x_in_cluster_dict[i] = x_in_cluster
            new_centers[i] = np.sum(x_in_cluster, axis=0) / x_in_cluster.shape[0] if x_in_cluster.shape[0] != 0 else self.centers
        return new_centers

    def _get_clusters(self, x, clust_cent):
        dist_clust_array = np.zeros((x.shape[0], clust_cent.shape[0]), dtype="float32")
        for i in range(self.k):
            dist_clust_array[:, i] = self._distance(x, clust_cent[i])
        self.clusters = np.argmin(dist_clust_array, axis=1)

    def single_fit_predict(self, X):
        dimension = X.shape[1]
        self.centers = np.zeros((self.k, dimension), dtype="float32")
        new_centers = np.zeros((self.k, dimension), dtype="float32")
        for i in range(self.k):
            new_centers[i] = np.array([
                random.uniform(np.min(X[:, j]), np.max(X[:, j])) for j in range(dimension)
            ])
        i = 0
        while (new_centers != self.centers).any() and i <= self.max_itiration:
            self.centers = new_centers
            self._get_clusters(X, new_centers)
            new_centers = self._get_new_centers(X)
            if self.loss == 'wcss':
                self.losses.append(self._wcss(new_centers))
            else:
                pass
            i += 1

        else:
            return {
                'clusters': self.clusters,
                'centers': new_centers,
                'loss': self.losses
            }

    def fit_predict(self, X):
        quality = float('inf')
        ultimate_results = {
            'clusters': None,
            'centers': None,
            'loss': None
        }
        for i in range(self.repeat):
            single_results = self.single_fit_predict(X)
            if single_results['loss'][-1] < quality:
                ultimate_results = single_results
        return ultimate_results


data = np.array([
    [2, 3],
    [5, 8],
    [3, 2],
    [5, 7],
    [3, 3],
    [4, 7],
    [2, 4],
    [3, 8],
    [3, 5],
    [7, 5],
    [6, 4],
    [6, 3],
    [7, 3],
    [7, 1]
])

km = KMeans(3)
result = km.fit_predict(data)
print(result['centers'])
print(result['loss'])

