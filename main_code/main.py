from create_heap_main_2 import document_clf
import warnings
warnings.filterwarnings('ignore')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from draw_heap import  draw_best_heap
from K_means_cluster import elbow_method, plot_k_means

"""
main function for K-means clustering
"""
def K_means_main(source_dir):
    # Initialize document_clf object
    cluster_object = document_clf(source_dir)

    # Create tf-idf matrix for source directory
    cluster_object.create_tf_idf_matrix()

    #plot elbow method to decide the number of clusters for K_means_clustering
    cluster_count = elbow_method(cluster_object.tf_idf_matrix)
    
    #After inertia decrease point from elbow method, set new num of clusters and get the wordcloud
    plot_k_means(cluster_object.tf_idf_matrix,cluster_object.term, source_dir, cluster_count)

"""
main function for heap-based clustering
"""
def heap_main(source_dir):
    # Initialize document_clf object
    cluster_object = document_clf(source_dir)

    # Create tf-idf matrix for source directory
    cluster_object.create_tf_idf_matrix()

    # Get a best single heap based on avg cosine similarity score
    best_single_heap = cluster_object.get_best_heap_cosine()

    # Draw a best heap
    draw_best_heap(best_single_heap)

    # Get a list of best heaps based on the number of nodes inside the heap structure
    best_heap_lst = cluster_object.get_heap_lst()

    # Cluster result labels
    # root document and its corresponding heap number
    labels, doc_to_heap = cluster_object.get_labels(best_heap_lst)
    cluster_object.k_means_cosine_sim(num_clusters=4)

    # Principal component analysis to reduce the dimension of the tf-idf to 2
    reduced_vectors = cluster_object.perform_pca(best_heap_lst)
    silhouete_score = cluster_object.compute_silhouette_score(labels, reduced_vectors)

    reduced_vectors = cluster_object.perform_pca(best_heap_lst)
    print(len(reduced_vectors))
    print(reduced_vectors)

    silhouete_score = cluster_object.compute_silhouette_score(labels, reduced_vectors)

    print(silhouete_score)

    

if __name__ == "__main__":
    source_dir = "/Users/junhoeum/Desktop/Summer_23/doc_clustering_algo/test_document_3_27_23/test_sample"
    K_means_main(source_dir)
    heap_main(source_dir)

