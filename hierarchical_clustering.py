import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check if dataset is properly scaled before dendrogram generation
print("Dataset mean after scaling:", np.mean(X_scaled, axis=0))
print("Dataset std deviation after scaling:", np.std(X_scaled, axis=0))

# Plot Dendrogram
plt.figure(figsize=(10, 5))
linkage_matrix = sch.linkage(X_scaled, method='ward')
sch.dendrogram(linkage_matrix)
plt.title('Dendrogram for Hierarchical Clustering on Iris Dataset')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# Determine optimal number of clusters from dendrogram
from scipy.cluster.hierarchy import fcluster
max_d = 7  # Adjust this threshold based on dendrogram visualization
optimal_clusters = len(set(fcluster(linkage_matrix, max_d, criterion='distance')))
print(f"Optimal number of clusters based on dendrogram: {optimal_clusters}")

# Implement Agglomerative Hierarchical Clustering with optimal clusters
hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters, metric='euclidean', linkage='ward')
labels = hierarchical.fit_predict(X_scaled)

# Plot the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
plt.title(f'Hierarchical Clustering with {optimal_clusters} Clusters on Iris Dataset')
plt.show()
