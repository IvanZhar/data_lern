import numpy as np


class Cluster:
    def __init__(self, index, coordinates=None, child_1=None, child_2=None):
        self.index = index
        self.clusters = coordinates
        self.child_1 = child_1
        self.child_2 = child_2
        self.is_leaf = True


class HierarchyClust:
    def __init__(self, n_clusters, metric='euclidean', linkage='ward'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.clusters = {}
        self.x = None

    @staticmethod
    def get_euclidean_distance(clust_1, clust_2):
        return np.linalg.norm(clust_1 - clust_2, axis=0)

    def get_ward(self, clust_1, clust_2):
        pass


    def get_distance(self, x):
        distance_matrix = np.full((len(self.clusters), len(self.clusters)), np.inf)
        for i in range(distance_matrix.shape[1]):
            for j in range(i + 1, distance_matrix.shape[1]):
                pass
                #distance_matrix[i, j] = self.get_ward()



    def fit_predict(self, x):
        assert x.shape[0] > self.n_clusters, 'Objects number exceed clusters number'
        self.x = x
        for i in range(x.shape[0]):
            self.clusters[i] = [Cluster(i, coordinates=x[i])]
        while len(self.clusters) > self.n_clusters:
            pass



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

# hc = HierarchyClust(5)
# hc.get_euclidean_clusters(a)
# print(hc.clusters)

# a = {frozenset({1, 2}): 9, frozenset({1, 4}): 0, frozenset({2, 3}): 5, frozenset({2, 7}): 5}
# print(a.keys())
#
# print([a.get(key) for key in a.keys() if 1 in key])


