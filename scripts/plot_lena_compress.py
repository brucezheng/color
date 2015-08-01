print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import array
from PIL import Image

from sklearn import cluster

n_clusters = 5
np.random.seed(0)

try:
    lena = array(Image.open('AUnroLL.jpg'))
except AttributeError:
    # Newer versions of scipy have lena in misc
    from scipy import misc
    lena = array(Image.open('AUnroLL.jpg'))
X = lena.reshape((-1, 1))  # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_

# create an array from labels and values
lena_compressed = np.choose(labels, values)
lena_compressed.shape = lena.shape

vmin = lena.min()
vmax = lena.max()

plt.figure(2, figsize=(3, 2.2))
plt.imshow(lena_compressed, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

plt.show()
