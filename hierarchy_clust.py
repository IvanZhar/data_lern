import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.cluster import AgglomerativeClustering
import timeit
import cProfile


class Cluster:
    def __init__(self, index, is_leaf=False, coordinates=None, n_points=1, feature_indexes=(), child_1=None, child_2=None):
        self.index = index
        self.coordinate = coordinates
        self.n_points = n_points
        self.feature_indexes = feature_indexes
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
            return self.get_euclidean_distance(frozenset([cluster_1.coordinate, cluster_2.coordinate]))
        elif not cluster_1.is_leaf or not cluster_2.is_leaf:
            s_cluster = cluster_2
            w_cluster = cluster_1
            if cluster_1.index < cluster_2.index:
                s_cluster = cluster_1
                w_cluster = cluster_2
            al_u, al_v, b = self.get_wards_coefs(s_cluster, w_cluster.child_1, w_cluster.child_2)
            return al_u * self.get_distance(frozenset([w_cluster.child_1, s_cluster])) + \
                al_v * self.get_distance(frozenset([w_cluster.child_2, s_cluster])) + \
                b * self.get_distance(frozenset([w_cluster.child_1, w_cluster.child_2]))

    def get_best_link(self):
        total_clusters = list(self.clusters.keys())
        best_link = [np.nan, np.nan]
        best_dist = np.inf
        for i in range(len(total_clusters)):
            for j in range(i + 1, len(total_clusters)):
                distance = self.get_distance(frozenset([self.clusters[total_clusters[i]], self.clusters[total_clusters[j]]]))
                if distance < best_dist:
                    best_dist = distance
                    best_link[0], best_link[1] = [total_clusters[i], total_clusters[j]]
        return best_link

    def fit(self, x):
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
            closest = self.get_best_link()
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

house_df = pd.read_csv('kc_house_data.csv', sep=',')
coords = house_df[['lat', 'long']].drop_duplicates()
coords_np_cut = coords.to_numpy()[:100]

# hc = HierarchyClust(20)
# hc.fit(coords_np_cut)
mine = 'hc = HierarchyClust(20);hc.fit(coords_np_cut)'

# sk = AgglomerativeClustering(n_clusters=20)
# sk.fit(coords)
sks = 'sk = AgglomerativeClustering(n_clusters=20);sk.fit(coords_np_cut)'

#print('MINE time:', timeit.timeit(mine, globals=globals(), number=1))
#print('SKLEARN time: ', timeit.timeit(sks, globals=globals(), number=1))

cProfile.run(mine)
#cProfile.run(sks)
