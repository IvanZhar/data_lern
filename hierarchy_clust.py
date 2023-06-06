import numpy as np
from functools import lru_cache



class Cluster:
    def __init__(self, index, is_leaf=False, coordinates=None, n_points=1, feature_indexes=[], child_1=None, child_2=None):
        self.index = index
        self.coordinate = coordinates
        self.n_points = n_points
        self.feature_indexes = feature_indexes
        #self.clusters = coordinates
        self.child_1 = child_1
        self.child_2 = child_2
        self.is_leaf = is_leaf


class HierarchyClust:
    def __init__(self, n_clusters, metric='euclidean', linkage='ward'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.clusters = {}
        self.cluster_creation_history = -1
        self.x = None

    @staticmethod
    @lru_cache
    def get_euclidean_distance(points_set):
        point_1, point_2 = points_set
        return np.linalg.norm(np.array(point_1) - np.array(point_2), axis=0)

    def get_wards_coefs(self, s, u, v):
        alpha_u = (s.n_points + u.n_points) / (s.n_points + (u.n_points + v.n_points))
        alpha_v = (s.n_points + v.n_points) / (s.n_points + (u.n_points + v.n_points))
        beta = -s.n_points / (s.n_points + (u.n_points + v.n_points))
        return alpha_u, alpha_v, beta

    @lru_cache
    def get_distance(self, cluster_set):
        cluster_1, cluster_2 = cluster_set
        if cluster_1.is_leaf and cluster_2.is_leaf:
            test = frozenset([cluster_1.coordinate, cluster_2.coordinate])
            return self.get_euclidean_distance(frozenset([cluster_1.coordinate, cluster_2.coordinate]))
        elif not cluster_1.is_leaf and not cluster_2.is_leaf:
            s_cluster = cluster_2
            w_cluster = cluster_1
            if cluster_1.index < cluster_2.index:
                s_cluster = cluster_1
                w_cluster = cluster_2
            al_u, al_v, b = self.get_wards_coefs(s_cluster, w_cluster.child_1, w_cluster.child_2)
            return al_u * self.get_distance(frozenset([w_cluster.child_1, s_cluster])) + \
                al_v * self.get_distance(frozenset([w_cluster.child_2, s_cluster])) + \
                b * self.get_distance(frozenset([w_cluster.child_1, w_cluster.child_2]))

    def get_distance_matrix(self):
        current_clusters_idxs = self.clusters.keys()
        max_clustr_idx = max(current_clusters_idxs)
        distance_matrix = np.full((max_clustr_idx, max_clustr_idx), np.inf)
        for i in range(distance_matrix.shape[1]):
            for j in range(i + 1, distance_matrix.shape[1]):
                if i not in current_clusters_idxs or j not in current_clusters_idxs:
                    pass
                else:
                    distance_matrix[i, j] = self.get_distance(frozenset([self.clusters[i], self.clusters[j]]))
        return distance_matrix

    def fit_predict(self, x):
        assert x.shape[0] > self.n_clusters, 'Objects number exceed clusters number'
        self.x = x
        for i in range(x.shape[0]):
            self.cluster_creation_history += 1
            self.clusters[self.cluster_creation_history] = Cluster(
                self.cluster_creation_history,
                coordinates=tuple(x[self.cluster_creation_history]),
                feature_indexes=[i],
                is_leaf=True)
        while len(self.clusters) > self.n_clusters:
            distance_matrix = self.get_distance_matrix()
            closest = np.argwhere(distance_matrix == np.min(distance_matrix))[0]
            self.cluster_creation_history += 1
            self.clusters[self.cluster_creation_history] = Cluster(
                self.cluster_creation_history,
                child_1=self.clusters[closest[0]],
                child_2=self.clusters[closest[-1]],
                n_points=self.clusters[closest[0]].n_points + self.clusters[closest[-1]].n_points,
                feature_indexes=self.clusters[closest[0]].feature_indexes + self.clusters[closest[-1]].feature_indexes
            )
            self.clusters.pop(closest[0])
            self.clusters.pop(closest[-1])







a = np.array([
    [4,   6],
    [2,   2],
    [7,   3],
    [2,   6],
    [7,   2],
    [2.8, 2],
    [1,   5]
])

hc = HierarchyClust(2)
print(hc.fit_predict(a))




# hc = HierarchyClust(5)
# hc.get_euclidean_clusters(a)
# print(hc.clusters)

# a = {frozenset({1, 2}): 9, frozenset({1, 4}): 0, frozenset({2, 3}): 5, frozenset({2, 7}): 5}
# print(a.keys())
#
# print([a.get(key) for key in a.keys() if 1 in key])


