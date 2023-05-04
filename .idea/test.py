import scipy.stats as at
import plotly.graph_objects as go
import numpy as np
import math as m
from sklearn.cluster import KMeans
from sklearn import datasets


dist = np.array([
    [0.14, 0.0037],
    [0.2, 0.0083],
    [0.28, 0.0174],
    [0.4, 0.0241],
    [0.57, 0.0321],
    [0.81, 0.0401],
    [1.15, 0.053],
    [1.63, 0.083],
    [2.31, 0.0993],
    [3.27, 0.0748],
    [4.63, 0.0359],
    [6.54, 0.0163],
    [9.25, 0.0092],
    [13.08, 0.006],
    [18.5, 0.0044],
])
kmeans = KMeans(n_clusters=10)
kmeans.fit(dist)
mu_k = kmeans.cluster_centers_
print(mu_k)