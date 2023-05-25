import numpy as np

distance_matrix = np.full((3, 3), np.inf)
distance_matrix[0, 0] = 1

print(distance_matrix)