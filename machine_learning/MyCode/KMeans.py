from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import datetime
import cv2


file_path = 'F:/suolong.jpg'
image = io.imread(file_path)
io.imshow(image)
io.show()

rows = image.shape[0]
cols = image.shape[1]

image = image.reshape(image.shape[0]*image.shape[1], 3)
print(image.shape)
kmeans = KMeans(n_clusters=128, n_init=10, max_iter=200)
print('正在聚类...')
print(datetime.datetime.now())
kmeans.fit(image)
print('聚类完成!')
print(datetime.datetime.now())
clusters = np.array(kmeans.cluster_centers_, dtype=np.uint8)
labels = np.array(kmeans.labels_, dtype=np.uint8)
labels = labels.reshape(rows, cols)

np.save('codebook_test.npy', clusters)
io.imsave('F:/test_suo.jpg', labels)


