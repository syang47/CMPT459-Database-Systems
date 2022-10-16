import numpy as np


class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])

        #print("entering while loop")
        while iteration < self.max_iter:
            # your code
            # get euclidean distance for X from the centroids
            iteration = iteration + 1
            dist = self.euclidean_distance(X, self.centroids[0, :])
            for class_ in range(1, self.n_clusters):
                dist = np.append(dist, self.euclidean_distance(X, self.centroids[class_, :]), axis=1)
            classes = np.argmin(dist, axis=1)

            # update centroids:
            self.update_centroids(classes, X)
           
        # predicting clustering:
        
        #print("exited while loop")
        dist = self.euclidean_distance(X, self.centroids[0])
        for class_ in range(1, self.n_clusters):
            dist = np.append(dist, self.euclidean_distance(X, self.centroids[class_, :]), axis=1)
            clustering = np.argmin(dist, axis=1)
        #print("finished prediction")

        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code
        
        for class_ in set(clustering):
            self.centroids[class_, :] = np.average(X[clustering==class_,:], axis = 0)
        

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
            centroids = X.copy()
            np.random.shuffle(centroids)
            self.centroids = centroids[:self.n_clusters] 
        elif self.init == 'kmeans++':
            # your code

            # initialize the centroids list and add a randomly selected data point to the list

            centroids_p = []
            k1=np.random.randint(0,X.shape[0])
            centroids_p.append(k1)
            while len(centroids_p) < self.n_clusters:
                distance = []
                centroids_p_list = X[centroids_p]
                for i in range(X.shape[0]):
                    distance.append(np.min(self.euclidean_distance(centroids_p_list, X[i])))
                dist = distance / np.sum(distance)
                p=np.random.rand()
                dist_sum = np.cumsum(dist).tolist()
                for m in range(X.shape[0]):
                    if dist_sum[m] >=p:
                        centroids_p.append(m)
                        break
            self.centroids = np.array(X[centroids_p])

        else:
            raise ValueError('Centroid initialization method should either be "random" or "kmeans++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        return np.linalg.norm(X1-X2, axis=1).reshape(-1,1)

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):    
        # a: distance of o to its centroid
        # b: distance of o to its second best centroid
        # s = (b-a)/max(a,b)
        dist_arr = []
        for i in range(len(self.centroids)):
            dist_arr.append(self.euclidean_distance(X, self.centroids[i,:]))
        dist_arr = np.array(dist_arr)
        dist_arr = np.squeeze(np.transpose(dist_arr))
        # distance of row object to its centroid
        a = dist_arr[range(dist_arr.shape[0]), clustering] 
        newarr = []
        for i in range(dist_arr.shape[0]):
            # distance of row object to its second best centroid
            newarr.append(np.delete(dist_arr[i], clustering[i]))
        b = np.min(newarr,axis=1) 
        upper = b-a
        silhouette_score=upper/np.maximum(a,b)
        # for i in range(len(self.centroids)):
        #     cluster_index = np.where(clustering==i)[0]
        #     sc_cluster=np.sum(silhouette_score[cluster_index])/len(cluster_index)
            # print(sc_cluster)
        # print(np.sum(silhouette_score)/len(silhouette_score))

        return np.sum(silhouette_score)/len(silhouette_score)
       
