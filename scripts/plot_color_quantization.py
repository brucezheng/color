# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image
from numpy import array
from collections import Counter
import math

# Load the Summer Palace photo
pic = Image.open('colbert.jpg').convert('RGB')
china = array(pic)

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1]
china = np.array(china, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

count = Counter()
count.update(kmeans.labels_)

print(len(kmeans.cluster_centers_))
colors = list(map(lambda x: (kmeans.cluster_centers_[x[0]],x[1]), count.most_common(len(kmeans.cluster_centers_))))
#colors = map(lambda x: (x[0]*255,x[1]), colors)

def solid(v):
    w, h = 10,10
    image = np.zeros((w, h, 3))
    for i in range(w):
        for j in range(h):
            image[i][j] = v
    return image

def dist(c1,c2):
    return math.sqrt(math.pow(c1[0]-c2[0],2) + math.pow(c1[1]-c2[1],2) + math.pow(c1[2]-c2[2],2))

i = 1
for j in range(len(colors)):
    print(j)
    add = True
    for k in range(j):
        if dist(colors[k][0],colors[j][0]) < .2:
            print('Too close:' + str(colors[k][0]) + ' ' + str(colors[j][0]))
            add = False
            break
    if(add):
        plt.figure(i, figsize=(6,2))
        plt.clf()
        ax = plt.axes([0, 0, 1, 1])
        plt.axis('off')
        plt.imshow(solid(colors[j][0]))
        i += 1

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(i)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.show()
