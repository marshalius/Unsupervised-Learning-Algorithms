import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('musteriler.csv')

x=df.iloc[:,3:]

from sklearn.cluster import KMeans

wcss = []
for i in range(1,10):
    km = KMeans(n_clusters=i, init="k-means++")
    km.fit(x)
    wcss.append(km.inertia_)
    
plt.plot(range(1,10), wcss)
plt.show()

km = KMeans(n_clusters=3, init="k-means++")
km.fit(x)

cc = km.cluster_centers_
plt.scatter(x.iloc[:,0].values, x.iloc[:,1].values, c=km.labels_, cmap='rainbow')
plt.scatter(cc[:,0], cc[:,1], s=100, c='black', marker='s')
plt.show()


