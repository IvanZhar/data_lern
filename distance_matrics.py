import numpy as np
from sys import getsizeof
import itertools


class DynamicDistanceMatrix:
    def __init__(self):
        self.header = None
        self.data = None

    def dataframe(self, cluster_dict):
        self.header = list(cluster_dict.keys())
        frame = {}
        frame_min = [float('inf'), float('inf')]
        for i in range(len(self.header)):
            for j in range(i + 1, len(self.header)):
                frame[frozenset([self.header[i], self.header[j]])] = Node(frozenset([cluster_dict[i], cluster_dict[j]]))
                if frame[frozenset([self.header[i], self.header[j]])].value < frame_min[1]:
                    frame_min[0] = frozenset([frame[frozenset([self.header[i], self.header[j]])].cluster_1,
                                              frame[frozenset([self.header[i], self.header[j]])].cluster_2])
                    frame_min[1] = frame[frozenset([self.header[i], self.header[j]])].value
        return frame


class Node:
    def __init__(self, link):
        self.cluster_1, self.cluster_2 = link
        if self.cluster_1.is_leaf and self.cluster_2.is_leaf:
            value = 0
            for i in range(len(self.cluster_1.coordinate)):
                value += (self.cluster_1[i] - self.cluster_2[i]) ** 2
            self.value = value ** 0.5
        else:
            self.value = self.get_distance(self.cluster_1, self.cluster_2)

    def get_distance(self, cluster1, cluster2):
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



