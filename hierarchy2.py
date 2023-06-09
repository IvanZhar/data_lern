import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.cluster import AgglomerativeClustering
import timeit
import cProfile


class DynamicDistanceMatrix:
    def __init__(self, cluster_dict):
        self.cluster_dict = cluster_dict
        self.header = list(self.cluster_dict.keys())
        self.frame = self._make_frame(cluster_dict)
        self.root = None

    def _put_to_chain(self, new_node):
        if not self.root:
            self.root = new_node
        else:
            if new_node.value < self.root.value:
                if



        
    def _make_frame(self, cluster_dict):
        frame = {}
        # frame_min = {'link': float('inf'), 'metric': float('inf')}
        for i in range(len(self.header)):
            for j in range(i + 1, len(self.header)):
                self._put_to_chain(Node(frozenset([cluster_dict[i], cluster_dict[j]])))
                # frame[frozenset([
                #     self.header[i],
                #     self.header[j]
                # ])] = Node(frozenset([cluster_dict[i], cluster_dict[j]]))

                # if frame[frozenset([self.header[i], self.header[j]])].value < frame_min['metric']:
                #     frame_min['link'] = frozenset([
                #         self.cluster_dict[i],
                #         self.cluster_dict[j]
                #     ])
                #     frame_min['metric'] = frame[frozenset([self.header[i], self.header[j]])].value
        return frame

    def update(self, merged_cluster):
        merging_cluster_1_idx = merged_cluster.child_1.index
        merging_cluster_2_idx = merged_cluster.child_2.index
        print(self.frame.keys())
        for each in list(self.frame):
            if merging_cluster_1_idx in each or merging_cluster_2_idx in each:
                self.frame.pop(each)
        self.header.remove(merging_cluster_1_idx)
        self.header.remove(merging_cluster_2_idx)
        for each in self.header:
            self.frame[frozenset([
                self.cluster_dict[each].index,
                merged_cluster.index
            ])] = Node(
                frozenset([
                    self.cluster_dict[each],
                    merged_cluster
                ])
            )
            # if self.frame[frozenset([self.cluster_dict[each].index, merged_cluster.index])].value < self.frame_min['metric']:
            #     self.frame_min['link'] = frozenset([
            #         self.cluster_dict[each],
            #         merged_cluster
            #     ])
            #     self.frame_min['metric'] = self.frame[frozenset([
            #         self.cluster_dict[each].index,
            #         merged_cluster.index
            #     ])].value


class Node:
    def __init__(self, link):
        self.cluster_1, self.cluster_2 = link
        self.smaller = None
        self.greater = None
        if self.cluster_1.is_leaf and self.cluster_2.is_leaf:
            value = 0
            for i in range(len(self.cluster_1.coordinate)):
                value += (self.cluster_1.coordinate[i] - self.cluster_2.coordinate[i]) ** 2
            self.value = value ** 0.5
        else:
            self.value = self._get_distance(self.cluster_1, self.cluster_2)

    def _get_distance(self, cluster1, cluster2):
        if cluster1.index > cluster2.index:
            cluster_w = cluster1
            cluster_s = cluster2
        else:
            cluster_w = cluster2
            cluster_s = cluster1
        cluster_v = cluster_w.child_1
        cluster_u = cluster_w.child_2

        alpha_u = (cluster_s.n_points + cluster_u.n_points) / (cluster_s.n_points + (cluster_u.n_points + cluster_v.n_points))
        alpha_v = (cluster_s.n_points + cluster_v.n_points) / (cluster_s.n_points + (cluster_u.n_points + cluster_v.n_points))
        beta = -cluster_s.n_points / (cluster_s.n_points + (cluster_u.n_points + cluster_v.n_points))

        return alpha_u * Node(frozenset([cluster_u, cluster_s])).value + \
            alpha_v * Node(frozenset([cluster_v, cluster_s])).value + \
            beta * Node(frozenset([cluster_u, cluster_v])).value


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
        distance_matrix = DynamicDistanceMatrix(self.clusters)
        while True:
            closest_1, closest_2 = distance_matrix.frame_min['link']
            print('closest: ', closest_1.index, closest_2.index)
            self.cluster_creation_history += 1
            self.clusters[self.cluster_creation_history] = Cluster(
                self.cluster_creation_history,
                child_1=closest_1,
                child_2=closest_2,
                n_points=closest_1.n_points + closest_2.n_points,
                feature_indexes=closest_1.feature_indexes + closest_2.feature_indexes
            )
            self.clusters.pop(closest_1.index)
            self.clusters.pop(closest_2.index)
            if len(self.clusters) > self.n_clusters:
                distance_matrix.update(self.clusters[self.cluster_creation_history])
            else:
                break


a = np.array([
    [4,   6],
    [2,   2],
    [7,   3],
    [2,   6],
    [7,   2],
    [2.8, 2],
    [1,   5]
])

# house_df = pd.read_csv('kc_house_data.csv', sep=',')
# coords = house_df[['lat', 'long']].drop_duplicates()
# coords_np_cut = coords.to_numpy()[:100]

# hc = HierarchyClust(20)
# hc.fit(coords_np_cut)
mine = 'hc = HierarchyClust(2);hc.fit(a)'

# sk = AgglomerativeClustering(n_clusters=20)
# sk.fit(coords)
sks = 'sk = AgglomerativeClustering(n_clusters=20);sk.fit(coords_np_cut)'

print('MINE time:', timeit.timeit(mine, globals=globals(), number=10000))
#print('SKLEARN time: ', timeit.timeit(sks, globals=globals(), number=1))

#cProfile.run(mine)
#cProfile.run(sks)
