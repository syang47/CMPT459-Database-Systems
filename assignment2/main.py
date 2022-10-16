from sklearn.cluster import cluster_optics_dbscan
import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='data.csv',
                        help='data path')
    parser.add_argument('--init', type=str, default='random', 
                        help='initialization method')
    a = parser.parse_args()
    return(a.n_clusters, a.data, a.init)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path, init_m = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)
    # Your code
    if n_classifiers > 0:
        kmeans = KMeans(n_clusters=n_classifiers, init=init_m)
        predicted_clusterings = kmeans.fit(X)    
        visualize_cluster(X[:,0], X[:,1], predicted_clusterings)
        plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1],s=50,c='k', marker='+')
        plt.title("Distribution of Points in k=%s Clusters: PCA0, PCA1 (%s)" % (n_classifiers, init_m))
        plt.xlabel("PCA0")
        plt.ylabel("PCA1")
        plt.savefig("%s_PCA0_PCA1.png" % init_m)
    elif n_classifiers == 0:
        ''' 
        Task 2: k from 2 to 9, random initialization:
        '''
        r_all_silhouette, r_all_clusterings, r_best_k_index = visualize_silhouette(np.arange(2,10), X, init_m)

        '''
        Task 3: k from 2 to 9, kmeans++ initialization:
        '''
        # p_all_silhouette, p_all_clusterings, p_best_k_index = visualize_silhouette(np.arange(2,10), X, "kmeans++")
        
        '''
        Task 4: plot scatter plot of best k from task 2:
        '''
        print("best k is: ", r_best_k_index)
        visualize_cluster(X[:,0], X[:,1], r_all_clusterings[r_best_k_index], 1)
        plt.title("Distribution of Points in k=%s Clusters: PCA0, PCA1 (random)" % r_best_k_index)
        plt.xlabel("PCA0")
        plt.ylabel("PCA1")
        plt.savefig("%s_PCA0_PCA1.png" % init_m)
        visualize_cluster(X[:,1], X[:,2], r_all_clusterings[r_best_k_index], 2)
        plt.title("Distribution of Points in k=%s Clusters: PCA1, PCA2 (random)" % r_best_k_index)
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.savefig("%s_PCA1_PCA2.png"  % init_m)
        # visualize_cluster(X[:,2], X[:,3], r_all_clusterings[r_best_k_index], 3)
        # plt.title("Distribution of Points in Clusters: PCA2, PCA3 (random)")
        # plt.xlabel("PCA2")
        # plt.ylabel("PCA3")
        # plt.savefig("R_PCA2_PCA3.png")
        
        # ## Kmeans++ Plots:
        # visualize_cluster(X[:,0], X[:,1], p_all_clusterings[p_best_k_index], 4)
        # plt.title("Distribution of Points in Clusters: PCA0, PCA1 (Kmeans++)")
        # plt.xlabel("PCA0")
        # plt.ylabel("PCA1")
        # plt.savefig("P_PCA0_PCA1.png")

        # visualize_cluster(X[:,1], X[:,2], p_all_clusterings[p_best_k_index], 5)
        # plt.title("Distribution of Points in Clusters: PCA1, PCA2 (Kmeans++)")
        # plt.xlabel("PCA1")
        # plt.ylabel("PCA2")
        # plt.savefig("P_PCA1_PCA2.png")
    else:
        print("n_clusters cannot be negative")

def visualize_cluster(x, y, clustering, fig_num):
    #Your code
    #random color coding
    vals=np.linspace(0,1,256)
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))
    plt.figure(fig_num)
    labels = np.unique(clustering)
    for i in labels:
        plt.scatter(x[clustering==i], y[clustering==i], s=20, facecolor=cmap(i))

def visualize_silhouette(n_classifiers, X, init):
    all_models,all_clusterings,all_silhouette = [],[],[]
    best_sc = -1

    for idx, n_c in enumerate(n_classifiers):
        all_models.append(KMeans(n_clusters=n_c, init=init))
        all_clusterings.append(all_models[idx].fit(X))
        all_silhouette.append(all_models[idx].silhouette(all_clusterings[idx], X))
    for idx, n_c in enumerate(n_classifiers):
        # print("At k = ", n_c, ", Silhouette Score is: ", all_silhouette[idx])
        if all_silhouette[idx] >= best_sc:
            best_sc = all_silhouette[idx]
            best_k_index = n_c
    # print("Best k is: ", best_k_index, "; SC = ", best_sc)
    
    # plt.figure()
    # plt.scatter(n_classifiers, all_silhouette)
    # plt.title("Silhouette Coefficient for k Clusters (init=%s)" % init)
    # plt.xlabel("k Clusters")
    # plt.ylabel("Silhouette Score")
    # plt.savefig("%s_SC_Plot.png" % init)
    return all_silhouette, all_clusterings, best_k_index

    

if __name__ == '__main__':
    main()
