import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('musteriler.csv')

x = df.iloc[:, 3:].values

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
pred = ac.fit_predict(x)

plt.scatter(x[pred==0,0], x[pred==0,1])
plt.scatter(x[pred==1,0], x[pred==1,1])
plt.scatter(x[pred==2,0], x[pred==2,1])
plt.scatter(x[pred==3,0], x[pred==3,1])
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
