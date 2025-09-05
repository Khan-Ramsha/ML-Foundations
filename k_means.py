import numpy as np
class KMeans:
    def __init__(self, num_samples, num_features, num_clusters):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_clusters = num_clusters

    def create_dataset(self):
        arr = np.random.randint(1,20, size = (self.num_samples, self.num_features))
        #centroid initialization
        indices = np.random.choice(self.num_samples, size = self.num_clusters, replace= False) #pick a random indices
        centroids = arr[indices]

        converged = False
        # print(centroids.shape)
        while not converged:
            assignments = []
            old = centroids.copy()
            for data_point in arr:
                min_dis = 10000
                best_cluster = 0
                for i in range(len(centroids)):
                    #euclidean distance
                    distance = np.linalg.norm(data_point - centroids[i])
                    if distance < min_dis:
                        min_dis = distance
                        best_cluster = i #pick the index
                assignments.append(best_cluster)
            
            for k in range(self.num_clusters):
                # calculate the avg of all the points of that particular cluster and udpate the cluster
                points_in_k = []

                for i in range(len(arr)):            
                    if assignments[i] == k:
                        points_in_k.append(arr[i])
                        print(points_in_k)

                if points_in_k:
                    new_centroid = np.mean(points_in_k, axis = 0)
                    centroids[k] = new_centroid
            if np.allclose(old, centroids):
                converged = True
                break
            
obj = KMeans(num_samples=20, num_features=3, num_clusters=2)
obj.create_dataset()