import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


name_of_file=input("Enter the name of file(pendigits/satellite/yeast) : ")
if name_of_file=="pendigits":
    ll=[]
    for i in range(16):
        ll.append(str(f'Col{i}'))
    ll.append("Label")
    df=pd.read_fwf("UCI_datasets/pendigits_training.txt",names=ll)
    print(df)
elif name_of_file=="satellite":
    ll=[]
    for i in range(36):
        ll.append(str(f'Col{i}'))
    ll.append("Label")
    df=pd.read_csv("UCI_datasets/satellite_training.txt",sep=" ",names=ll,header=None)
    print(df)

elif name_of_file=="yeast":
    ll=[]
    for i in range(8):
        ll.append(str(f'Col{i}'))
    ll.append("Label")
    df=pd.read_csv("UCI_datasets/yeast_training.txt",delimiter="\s+",index_col=False,names=ll)
    print(df)



def initialize_centroids(data, k):
    np.random.seed(0)
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]
    return centroids

def assign_clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids

def k_means(data, k, max_iterations=20):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Calculate Error using Euclidean Distance
def calculate_error(data, clusters, centroids):
    distances = np.sqrt(((data - centroids[clusters])**2).sum(axis=1))
    return np.mean(distances) / np.sqrt(data.shape[1])  # Normalized error

# Initialize K-means for different K values and calculate Error
k_values = range(2, 11)
errors = []

for k in k_values:
    clusters, centroids = k_means(df.values, k)
    error = calculate_error(df.values, clusters, centroids)
    errors.append(error)
    print(f"For k = {k} After 20 iterations: Error = {error:.4f}")

# Plot Error vs k chart
plt.figure(figsize=(10, 6))
plt.plot(k_values, errors, marker='o')
plt.xlabel('K Values')
plt.ylabel('Error')
plt.title('Error vs K Values')
plt.xticks(k_values)
plt.grid(True)
plt.show()


