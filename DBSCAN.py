import numpy as np
import cProfile
import pandas as pd
import timeit


class DBSCAN:
    def __init__(self, eps, min_poins=3, metric='euclidean'):
        self.eps = eps
        self.min_poins = min_poins
        self.visited = np.array([])
        self.possibly_noise = np.array([])
        self.clusterred_points = np.array([])
        self.clusters = {}
        self.metric = metric
        self.current_cluster = -1
        self.cluster_vector = None

    @staticmethod
    def euclidean(x_1, x_2):
        return np.linalg.norm(x_1 - x_2, axis=0)

    def get_neighbors(self, point, x):
        if self.metric == 'euclidean':
            possibly_close = x
            for i in range(x.shape[1]):
                possibly_close = possibly_close[abs(possibly_close[:, i] - point[i]) <= self.eps]
            return np.array([obj for obj in possibly_close if self.euclidean(obj, point) <= self.eps])
        else:
            exit('any other metrics are not supported yet!')

    def expand_cluster(self, root, neighbors):
        self.clusterred_points.add(root)
        if self.current_cluster not in self.clusters:
            self.clusters[self.current_cluster] = root
        else:
            np.vstack((self.clusters[self.current_cluster], root))
        for each in neighbors:
            if root in self.visited and each not in self.possibly_noise:
                continue
            elif root in self.visited and each in self.possibly_noise:
                self.possibly_noise.remove(root)
            new_neighbors = self.get_neighbors(each, root)
            for new_neighbor in new_neighbors:
                self.expand_cluster(new_neighbor, new_neighbors)

    def fit_predict(self, x):
        for each in x:
            if each in self.visited:
                continue
            if self.visited.shape[0] != 0:
                np.vstack((self.visited, each))
            else:
                self.visited[0] = np.array([each])
            neighbors = self.get_neighbors(each, x)
            if neighbors.shape[0] >= self.min_poins:
                self.current_cluster += 1
                self.expand_cluster(each, neighbors)
        self.clusters[-1] = self.possibly_noise
        self.cluster_vector = np.zeros(x.shape)
        for key in self.clusters:
            for obj in self.clusters[key]:
                self.cluster_vector[x == obj] = key



house_df = pd.read_csv('kc_house_data.csv', sep=',')

mine_db = DBSCAN(0.01)
mine_db.fit_predict(house_df[['lat', 'long']].to_numpy())
