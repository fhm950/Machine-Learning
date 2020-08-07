import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans

X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40) 

clusters = len(np.unique(y))
k = KMeans(K=clusters, max_iters=150, plot_steps=False)
y_pred = k.predict(X)

k.plot()