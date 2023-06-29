from create_heap_main_2 import document_clf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from dataclasses import dataclass
from nltk.corpus import stopwords
import os
from scipy.spatial.distance import cosine
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from draw_heap import draw_heap, draw_best_heap
from pdf_reader import pdf_to_text,read_pdf_tf_idf, get_combined_token_dict
from sklearn.feature_extraction.text import TfidfVectorizer
from K_means_cluster import elbow_method, K_means_cluster, plot_k_means
import nltk


"""
main function for K-means clustering
"""
def K_means_main():
    # Source directory to cluster documents from
    source_dir = "/Users/junhoeum/Desktop/Summer_23/doc_clustering_algo/test_document_3_27_23/test_sample"

    # Initialize document_clf object
    cluster_object = document_clf(source_dir)

    # Create tf-idf matrix for source directory
    cluster_object.create_tf_idf_matrix()

    #plot elbow method to decide the number of clusters for K_means_clustering
    cluster_count = elbow_method(cluster_object.tf_idf_matrix)
    
    #After inertia decrease point from elbow method, set new num of clusters and get the wordcloud
    plot_k_means(cluster_object.tf_idf_matrix,cluster_object.term, source_dir, cluster_count)

def heap_main():
    # Source directory to cluster documents from
    source_dir = "/Users/junhoeum/Desktop/Summer_23/doc_clustering_algo/test_document_3_27_23/test_sample"

    # Initialize document_clf object
    cluster_object = document_clf(source_dir)

    # Create tf-idf matrix for source directory
    cluster_object.create_tf_idf_matrix()

    # Since create_heap function plots all 200 document seperately to draw_every_heap() / get_heap()
    best_heap = cluster_object.get_heap()

    draw_best_heap(best_heap)

if __name__ == "__main__":
    source_dir = "Your_Source_path"
    K_means_main(source_dir)
    heap_main(source_dir)
