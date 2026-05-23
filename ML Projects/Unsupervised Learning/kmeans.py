import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = np.array([
    [150,45],[152,48],[155,50],
    [170,70],[172,72],[175,75],
    [190,90],[192,95],[195,100]
])


kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(data)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(data[:,0],data[:,1],c=labels,s=100,cmap='viridis')
plt.scatter(centers[:,0],centers[:,1],c='red',marker='x',s=300,label='Standard Size Specs')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('How AI Discover T-shirt Sizes')
plt.legend()
plt.show()