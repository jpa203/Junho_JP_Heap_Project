import matplotlib.pyplot as plt
import warnings
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')
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
nltk.download('words')
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

@dataclass
class document_clf:
    def __init__(self,source_dir):
        self.source_dir=source_dir # Working Directory to cluster documents 
        self.token_dict=get_combined_token_dict(source_dir) # Get dictionary of tokens retrieved from both docx and pdf
        self.tf_idf_vectors=None # Constructor to generate tf_idf_vector from combined_token_dict and store tf_idf_vectors
        self.tf_idf_matrix=None # Constructor to store tf_idf_matrix generated from tf_idf_vectors
        self.best_heap=None# store best heap once the loop is finished
        self.documents=None #
        self.term="" #Constructor to store file names after extracting topic and rename 

    def create_tf_idf_matrix(self):
        token_lst = list(self.token_dict.values())
        # Return the list of document text
        vectorizer = TfidfVectorizer()
        self.tf_idf_vectors = vectorizer.fit_transform(token_lst)
        self.tf_idf_matrix = self.tf_idf_vectors.toarray()
        self.term = vectorizer.get_feature_names_out()
        file_name_string = "_".join(self.term)

    def get_document_names(self):
        #Get list of document names to assign to heap structure visualization and labeling
        self.documents = list(self.token_dict.keys())

    def get_heap(self):
        self.get_document_names() # Get list of all document name and store it in a constructor
        k = 15 # number of sim documents to return for one heap
        threshold  = 0.2 # threshold for cosin similiarity
        # array to store document numbers stored by heap
        j = 0
        counter = 0
        best_heap = None
        best_heap_num_covered = 0
        while j < len(self.documents):
            # Initialize a heap data structure with the first document
            # heap will generate relevant document according to the root document
            file_name_lst = self.documents[j:len(self.documents)]
            heap = [(1, self.tf_idf_vectors[j], 1, file_name_lst[0])]

            # Loop through each subsequent document
            for i in range(j+1, self.tf_idf_vectors.shape[0]):
                # Calculate cosine similarity between the new document and each document currently in the heap
                cosine_similarities = [1 - cosine(heap_item[1].toarray()[0], self.tf_idf_vectors[i].toarray()[0]) for heap_item in heap]

                # Check if the cosine similarity between the new document and any document in the heap is greater than the threshold value
                if max(cosine_similarities) >= threshold:
                    # Add the new document to the heap
                    heap_item = (i + 1, self.tf_idf_vectors[i], max(cosine_similarities), file_name_lst[i-j])
                    heapq.heappush(heap, heap_item)

                    # Remove the document with the lowest cosine similarity if the heap is full
                    if len(heap) > k:
                        heapq.heappop(heap)

            # check if this heap has more covered documents than the current best heap
            if len(heap) > best_heap_num_covered:
                best_heap = heap
                best_heap_num_covered = len(heap)

            j += 1
            

        return best_heap
    
    def get_best_heap_cosine(self):
        # Get a single best heap structure after comparing avg cosine similarity of all heap strucutures generated from this algorithm
        # Get the list of document names
        self.get_document_names()

        file_name_lst = self.documents

        # Set the number of most similar documents to return
        k = 15
        # Set a threshold for cosine similarity
        threshold = 0.2
        
        # array to store document numbers stored by heap
        j = 0
        counter = 0
        best_heap = None
        best_heap_avg_sim = 0  # store the average similarity of the best heap
        while j < len(self.documents):
            # Initialize a heap data structure with the first document
            # heap will generate relevant document according to the root document
            file_name_lst2 = file_name_lst[j:len(self.documents)]
            heap = [(1, self.tf_idf_vectors[j], 1, file_name_lst2[0])]
            total_sim = 0  # to store the total similarity for this heap

            # Loop through each subsequent document
            for i in range(j+1, self.tf_idf_vectors.shape[0]):
                # Calculate cosine similarity between the new document and each document currently in the heap
                cosine_similarities = [1 - cosine(heap_item[1].toarray()[0], self.tf_idf_vectors[i].toarray()[0]) for heap_item in heap]
                max_sim = max(cosine_similarities)

                # Check if the cosine similarity between the new document and any document in the heap is greater than the threshold value
                if max_sim >= threshold:
                    # Add the new document to the heap
                    heap_item = (i + 1, self.tf_idf_vectors[i], max_sim, file_name_lst2[i-j])
                    heapq.heappush(heap, heap_item)
                    total_sim += max_sim  # add the similarity to the total

                    # Remove the document with the lowest cosine similarity if the heap is full
                    if len(heap) > k:
                        removed_item = heapq.heappop(heap)
                        total_sim -= removed_item[2]  # subtract the similarity of the removed item from the total

            # check if this heap has higher average cosine similarity than the current best heap
            heap_avg_sim = total_sim / len(heap) if len(heap) else 0
            if heap_avg_sim > best_heap_avg_sim:
                best_heap = heap
                best_heap_avg_sim = heap_avg_sim

            j += 1
        return best_heap
    
    def get_heap_lst(self):
        #The function returns the list of heaps, where each heap represents a cluster of documents. The length of this list is the number of clusters formed by this algorithm.
        #creating a list of heaps where each heap represents a cluster of documents 
        #based on their cosine similarities in the term-frequency inverse document frequency (TF-IDF) vector space
        self.create_tf_idf_matrix()
        self.get_document_names()
        
        k = 15
        threshold = 0.2
        j = 0
        best_heaps = []  

        while j < len(self.documents):
            file_name_lst = self.documents[j:len(self.documents)]
            heap = [(1, self.tf_idf_vectors[j], 1, file_name_lst[0])]
            for i in range(j+1, self.tf_idf_vectors.shape[0]):
                cosine_similarities = [1 - cosine(heap_item[1].toarray()[0], self.tf_idf_vectors[i].toarray()[0]) for heap_item in heap]
                if max(cosine_similarities) >= threshold:
                    heap_item = (i + 1, self.tf_idf_vectors[i], max(cosine_similarities), file_name_lst[i-j])
                    heapq.heappush(heap, heap_item)
                    if len(heap) > k:
                        heapq.heappop(heap)
            if len(heap) > 2:  # Ensure that the heap has more than one sample
                best_heaps.append(heap) 
            j += 1
        
        return best_heaps
    
    def get_labels(self, best_heap_lst):
    # Cluster result labels
    # root document and its corresponding heap number
        doc_to_heap = {}
        labels = []
        for i, heap in enumerate(best_heap_lst):
            for item in heap:
                doc_name = item[3]
                doc_to_heap[doc_name] = i
                labels.append(i)

        return labels, doc_to_heap
    
    # Use Principal component analysis to reduce dimension of tf-idf vector (high dimension)
    # Transform the high dimensional data into a 2D or 3D space where we can plot it.
    def perform_pca(self, best_heaps):
        # Concatenate all tf-idf vectors from all heaps
        all_vectors = []
        for heap in best_heaps:
            all_vectors.extend([heap_item[1].toarray() for heap_item in heap])

        all_vectors = np.concatenate(all_vectors, axis=0)

        # Perform PCA to reduce dimensions to 2
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(all_vectors)

        return reduced_vectors
    
    def compute_silhouette_score(self, labels, reduced_vectors):
        score = silhouette_score(reduced_vectors, labels)
        return score

    def k_means_cosine_sim(self, num_clusters):
        # function to compare k means clustering result with best_heap similarity score

        # First get the best heap
        best_heap = self.get_best_heap_cosine()
        # compute the average cosine similarity of the best heap
        avg_cosine_similarity_heap = sum(item[2] for item in best_heap) / len(best_heap) if best_heap else 0
        print('Average cosine similarity of the best heap:', avg_cosine_similarity_heap)

        # Then do the K-means clustering
        model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
        model.fit(self.tf_idf_vectors)

        labels = model.labels_
        clusters = {i: np.where(labels == i)[0] for i in range(model.n_clusters)}

        # print top terms per cluster and generate word clouds
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]

        avg_cosine_similarities_kmeans = []

        for cluster, indices in clusters.items():
            cluster_vectors = self.tf_idf_vectors[indices]
            pairwise_cosine_similarities = cosine_similarity(cluster_vectors)
            avg_cosine_similarity = np.mean(pairwise_cosine_similarities)
            avg_cosine_similarities_kmeans.append(avg_cosine_similarity)

        print('Average cosine similarities of K-means clusters:', avg_cosine_similarities_kmeans)
        print("Average Cosine Similarity for K means clusters:", sum(avg_cosine_similarities_kmeans)/len(avg_cosine_similarities_kmeans))

# Test heap creation and dimension reduction for silhouete score assignment later
if __name__=="__main__":

    # Directory containing small num of documents
    small_source_dir = "/Users/junhoeum/Desktop/Summer_23/doc_clustering_algo/test_document_3_27_23/test_sample"

    cluster_object = document_clf(small_source_dir)
    
    # Get a list of best heaps based on the number of nodes inside the heap structure
    best_heap_lst = cluster_object.get_heap_lst()

    # Draw heaps in best heap lst
    for heap in best_heap_lst:
        draw_best_heap(heap)

    # Get a best single heap based on avg cosine similarity score
    best_single_heap = cluster_object.get_best_heap_cosine()

    # Draw a best heap
    draw_best_heap(best_single_heap)

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
