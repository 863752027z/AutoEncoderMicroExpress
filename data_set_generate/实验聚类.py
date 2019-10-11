from sklearn.cluster import KMeans
import numpy as np
import datetime
import pandas as pd


def read_data_from_excel(path):
    df = pd.read_excel(path, 'page_1')
    data = np.array(df)
    data = np.delete(data, 0, axis=1)
    return data


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
