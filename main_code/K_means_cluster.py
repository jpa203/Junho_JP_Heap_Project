import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def elbow_method(num_of_clusters,tf_idf_matrix):
        # Initialize lists and mappings for distortions and inertias
        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        
        # Define range of k-values
        K = range(1, num_of_clusters + 1)

        # Loop through each value of k and fit the model
        for k in K:
            kmeanModel = KMeans(n_clusters=k).fit(tf_idf_matrix)
            kmeanModel.fit(tf_idf_matrix)

            # Calculate distortions and inertias and store in lists
            distortions.append(sum(np.min(cdist(tf_idf_matrix, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / len(tf_idf_matrix[0]))
            inertias.append(kmeanModel.inertia_)

            # Store distortions and inertias in mappings
            mapping1[k] = sum(np.min(cdist(tf_idf_matrix, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / len(tf_idf_matrix[0])
            mapping2[k] = kmeanModel.inertia_

        # Print out the distortions for each value of k
        for key, val in mapping1.items():
            print(f'{key} : {val}')

        # Plot the elbow curve
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.show()

def K_means_cluster(tf_idf_matrix,doc_lst,num_clusters=4):
    # after finding the best number of clusters, use K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=23)
    kmeans.fit(tf_idf_matrix)
    cluster_labels = kmeans.predict(tf_idf_matrix)

    doc_label_dict = dict()
    for i in range(len(doc_lst)):
        # Assign labels to each document based on cluster labels
        doc_label_dict[doc_lst[i]] = cluster_labels[i]
    inertia = kmeans.inertia_
    print(f"Inertia: {inertia}")
    
    return doc_label_dict


def plot_k_means(tf_idf_vectors,terms,doc_source,num_clusters):
    model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
    model.fit(tf_idf_vectors)

    # print top terms per cluster and generate word clouds
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]


    # Create directory if it doesn't exist
    output_dir = os.path.join(doc_source, "K-means_wordcloud")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_clusters):
        print(f"Cluster {i}:")
        words = [terms[ind] for ind in order_centroids[i, :10]]
        print(" ".join(words))
        
        wordcloud = WordCloud(background_color='white').generate(" ".join(words))
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Cluster {i + 1}")
        plt.show()

        # Save the figure to a file in the output directory
        plt.savefig(os.path.join(output_dir, f"wordcloud_cluster_{i + 1}.png"))
