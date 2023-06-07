import numpy as np
import timeit
from sklearn.cluster import DBSCAN as sk_DBSCAN


class DBSCAN:
    def __init__(self, eps, min_poins=3, metric='euclidean'):
        self.eps = eps
        self.min_poins = min_poins
        self.visited = None
        self.clusters = {-1: []}
        self.clusterized = None
        self.features = None

    @staticmethod
    def euclidean(x_1, x_2):
        return np.linalg.norm(x_1 - x_2, axis=0)

    def get_neighbors(self, point, point_idx):
        all_closest = np.where(np.linalg.norm(self.features[:] - point, axis=1) <= self.eps)
        return np.delete(all_closest, np.argwhere(all_closest == point_idx))

    def expand_cluster(self, neighbors):
        for each in neighbors:
            if self.visited[each]:
                continue
            self.visited[each] = True
            self.clusterized.append(each)
            return self.expand_cluster(self.get_neighbors(self.features[each], each))

    def fit(self, x):
        grey_zone = []
        self.features = x
        self.visited = np.full((x.shape[0],), False)
        for i in range(self.features.shape[0]):
            if self.visited[i]:
                continue
            self.visited[i] = True
            neighbors = self.get_neighbors(self.features[i], i)
            self.clusterized = [i]
            self.expand_cluster(neighbors)
            if len(self.clusterized) < self.min_poins:
                grey_zone += self.clusterized
            else:
                self.clusters[max(self.clusters) + 1] = self.clusterized
        self.clusters[-1] = grey_zone



a = np.array([
    [5,  3],
    [2,  6],
    [10, 6],
    [10, 7],
    [6,  5],
    [3,  6],
    [5,  4],
    [10, 5],
    [6,  4],
    [7,  5],
    [10, 2],
    [7,  4],
    [10, 4],
    [10, 3]
])

# db = DBSCAN(2)
# db.fit(a)
mine = 'db = DBSCAN(2);db.fit(a)'


# sk = sk_DBSCAN(eps=2)
# sk.fit(a)
sks = 'sk = sk_DBSCAN(eps=2);sk.fit(a)'

print('MINE time: ', timeit.timeit(mine, globals=globals(), number=1000))
print('SKLEARN time: ', timeit.timeit(sks, globals=globals(), number=1000))

