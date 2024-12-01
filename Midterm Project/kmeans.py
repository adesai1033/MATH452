import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from load_mnist import load
from pca import pca_alg


def plot_clusters(data, labels, centroids):
   
    data = pca_alg(data)
    centroids = pca_alg(centroids)

    k = len(centroids)
    for cluster_idx in range(k):

        cluster_points = data[labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'Cluster {cluster_idx+1}')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clusters and Centroids with PCA')
    plt.legend()
    plt.grid(True)
    plt.show()


def elbow_method(data, max_k):
    
    costs = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
        kmeans.fit(data)
        costs.append(kmeans.inertia_)  

    plt.plot(range(1, max_k + 1), costs, marker='o', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Clustering Cost (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()


def plot_centroids_and_samples(centroids, data, labels, n_samples=5):
  
    k = len(centroids)
    fig, axes = plt.subplots(k, n_samples + 1, figsize=(10, k * 2))

    for cluster_idx in range(k):

        centroid_img = centroids[cluster_idx].reshape(28, 28)
        axes[cluster_idx, 0].imshow(centroid_img, cmap='gray')
        axes[cluster_idx, 0].set_title(f'Centroid {cluster_idx+1}')
        axes[cluster_idx, 0].axis('off')
        cluster_samples = data[labels == cluster_idx]

        for i in range(min(n_samples, len(cluster_samples))):

            sample_img = cluster_samples[i].reshape(28, 28)
            axes[cluster_idx, i + 1].imshow(sample_img, cmap='gray')
            axes[cluster_idx, i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    data, _ = load()
    data = np.array(data)
  
    elbow_method(pca_alg(data, num_dim=3), max_k=15)
  
    chosen_k = 4  
    kmeans = KMeans(n_clusters=chosen_k, init='k-means++', n_init=10, random_state=0)
    kmeans.fit(data)
  
    centroids = kmeans.cluster_centers_

    plot_clusters(data, kmeans.labels_, centroids)
    plot_centroids_and_samples(centroids, data, kmeans.labels_, n_samples=5)


if __name__ == '__main__':
    main()
