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

@dataclass
class document_clf:
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.token_dict = get_combined_token_dict(source_dir)
        self.tf_idf_vectors = None
        self.tf_idf_matrix = None
        self.best_heap = None
        self.documents = None
        self.term = ""

    def create_tf_idf_matrix(self):
        token_lst = list(self.token_dict.values())
        # Return the list of document text
        vectorizer = TfidfVectorizer()
        self.tf_idf_vectors = vectorizer.fit_transform(token_lst)
        self.tf_idf_matrix = self.tf_idf_vectors.toarray()
        self.term = vectorizer.get_feature_names_out()
        file_name_string = "_".join(self.term)

    def get_document_names(self):
        self.documents = list(self.token_dict.keys())

    # Used as a constructor to get the best heap
    def draw_every_heap(self):
        # Get the list of document names
        self.get_document_names()

        file_name_lst = self.documents
    
        # Set the number of most similar documents to return
        k = 15
        # Set a threshold for cosine similarity
        threshold = 0.5
        
        # array to store document numbers stored by heap
        j = 0
        counter = 0
        best_heap = None
        best_heap_num_covered = 0
        while j < len(self.documents):
            # Initialize a heap data structure with the first document
            # heap will generate relevant document according to the root document
            file_name_lst2 = file_name_lst[j:len(self.documents)]
            heap = [(1, self.tf_idf_vectors[j], 1, file_name_lst2[0])]

            # Loop through each subsequent document
            for i in range(j+1, self.tf_idf_vectors.shape[0]):
                # Calculate cosine similarity between the new document and each document currently in the heap
                cosine_similarities = [1 - cosine(heap_item[1].toarray()[0], self.tf_idf_vectors[i].toarray()[0]) for heap_item in heap]

                # Check if the cosine similarity between the new document and any document in the heap is greater than the threshold value
                if max(cosine_similarities) >= threshold:
                    # Add the new document to the heap
                    heap_item = (i + 1, self.tf_idf_vectors[i], max(cosine_similarities), file_name_lst2[i-j])
                    heapq.heappush(heap, heap_item)

                    # Remove the document with the lowest cosine similarity if the heap is full
                    if len(heap) > k:
                        heapq.heappop(heap)

            # check if this heap has more covered documents than the current best heap
            if len(heap) > best_heap_num_covered:
                best_heap = heap
                best_heap_num_covered = len(heap)

            j += 1
            draw_heap(heap)

        return best_heap
    
        # Used as a constructor to get the best heap
    def get_heap(self):
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
        best_heap_num_covered = 0
        while j < len(self.documents):
            # Initialize a heap data structure with the first document
            # heap will generate relevant document according to the root document
            file_name_lst2 = file_name_lst[j:len(self.documents)]
            heap = [(1, self.tf_idf_vectors[j], 1, file_name_lst2[0])]

            # Loop through each subsequent document
            for i in range(j+1, self.tf_idf_vectors.shape[0]):
                # Calculate cosine similarity between the new document and each document currently in the heap
                cosine_similarities = [1 - cosine(heap_item[1].toarray()[0], self.tf_idf_vectors[i].toarray()[0]) for heap_item in heap]

                # Check if the cosine similarity between the new document and any document in the heap is greater than the threshold value
                if max(cosine_similarities) >= threshold:
                    # Add the new document to the heap
                    heap_item = (i + 1, self.tf_idf_vectors[i], max(cosine_similarities), file_name_lst2[i-j])
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
    
if __name__ == "__main__":
    source_dir = "/Users/junhoeum/Desktop/Summer_23/doc_clustering_algo/test_document_3_27_23/Baseball_stat"

    # Directory containing small num of documents
    small_source_dir = "/Users/junhoeum/Desktop/Summer_23/doc_clustering_algo/test_document_3_27_23/test_sample"
    cluster_object = document_clf(small_source_dir)
    
    cluster_object.create_tf_idf_matrix()  # Call create_tf_idf_matrix() to initialize tf_idf_vectors
    print(len(cluster_object.tf_idf_matrix))
    elbow_method(3,cluster_object.tf_idf_matrix)
    plot_k_means(cluster_object.tf_idf_matrix,cluster_object.term, 3)

    # Since create_heap function plots all 200 document seperately to draw_every_heap() / get_heap()
    best_heap = cluster_object.get_heap()

    draw_best_heap(best_heap)