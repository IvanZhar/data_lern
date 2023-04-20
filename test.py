import numpy as np

x = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4]
])

alf = np.array([
    [3, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 3, 0],
    [0, 0, 0, 3]
])

p = np.array([
    [5],
    [5],
    [5],
    [5]
])

print(np.dot(np.dot(x, alf), p))
print(np.dot(np.dot(x, p), alf))