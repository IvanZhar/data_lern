class DistanceMatrix:
    def __init__(self, num_clusters):
        self.matrix = [[Node(set(), float('inf')) for _ in range(num_clusters)] for _ in range(num_clusters)]
        self.heap = BinaryHeap()
        self.num_clusters = num_clusters

    def update_distance(self, cluster1, cluster2, distance):
        self.matrix[cluster1][cluster2].reset(frozenset([cluster1, cluster2]), distance)
        self.heap.insert(self.matrix[cluster1][cluster2])

    def merge_clusters(self, cluster1, cluster2):
        new_cluster = self.num_clusters
        self.num_clusters += 1

        for i in range(self.num_clusters):
            if i != cluster1 and i != cluster2:
                dist1 = self.matrix[i][cluster1].value
                dist2 = self.matrix[i][cluster2].value
                min_distance = min(dist1, dist2)

                self.update_distance(i, new_cluster, min_distance)

        self.heap = BinaryHeap()

        for i in range(self.num_clusters):
            if i != new_cluster:
                self.heap.insert(self.matrix[i][new_cluster])

        return new_cluster

    def find_closest_clusters(self):
        min_distance_node = self.heap.extract_min()
        return min_distance_node.link

    def perform_clustering(self, desired_num_clusters):
        while self.num_clusters > desired_num_clusters:
            closest_clusters = self.find_closest_clusters()
            new_cluster = self.merge_clusters(closest_clusters[0], closest_clusters[1])
            print(f"Merged clusters {closest_clusters[0]} and {closest_clusters[1]} into new cluster {new_cluster}")

    def print_matrix(self):
        for row in self.matrix:
            print([node.value for node in row])

class Node:
    def __init__(self, link, value):
        self.link = link
        self.value = value

    def reset(self, new_link, new_value):
        self.link = new_link
        self.value = new_value

class BinaryHeap:
    def __init__(self):
        self.heap = []

    def insert(self, node):
        self.heap.append(node)
        self._sift_up(len(self.heap) - 1)

    def extract_min(self):
        if len(self.heap) == 0:
            return None
        min_node = self.heap[0]
        last_node = self.heap.pop()
        if len(self.heap) > 0:
            self.heap[0] = last_node
            self._sift_down(0)
        return min_node

    def _sift_up(self, index):
        parent_idx = (index - 1) // 2
        while index > 0 and self.heap[parent_idx].value > self.heap[index].value:
            self.heap[parent_idx], self.heap[index] = self.heap[index], self.heap[parent_idx]
            index = parent_idx
            parent_idx = (index - 1) // 2

    def _sift_down(self, index):
        left_child_idx = 2 * index + 1
        right_child_idx = 2 * index + 2
        smallest = index
        if left_child_idx < len(self.heap) and self.heap[left_child_idx].value < self.heap[smallest].value:
            smallest = left_child_idx
        if right_child_idx < len(self.heap) and self.heap[right_child_idx].value < self.heap[smallest].value:
            smallest = right_child_idx
        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self._sift_down(smallest)