
# Document Classifier implemented with heap data structure

This project introduces an approach to document clustering, employing Term Frequency-Inverse Document Frequency (TF-IDF) and cosine similarity, implemented with heap data structures. This algorithm is designed to aid users in organizing documents within a directory in a computationally optimized way. By leveraging the heap data structure's average time complexity of O(log n) for insertions and deletions, we aim to improve performance when dealing with large datasets.

The algorithm calculates the similarity between each document using cosine similarity and TF-IDF, powerful techniques in the text analysis field. By doing so, it can effectively group similar documents based on their content. This project serves as a guide for individuals and organizations who wish to optimize their document clustering process, thereby saving time and computational resources.


## Dependencies

- Python 3
- matplotlib
- networkx
- scipy
- nltk
- sklearn
- [draw_heap (local module)](#Draw_Heap)
- [pdf_reader (local module)](#PDF_Reader)
- [K_means_cluster (local module)](#K_Means)
- [read_docx_tf_idf (local module)](#K_Means)
- [lsa (local module)](#lsa)

## Instructions
1. Install all the required dependencies. You can use pip to install these packages. For example:
```
pip install matplotlib networkx scipy nltk sklearn 
```
2. Clone or download this repository to your local machine.

3. You will need the modules draw_heap, pdf_reader, document_clf, lsa, main, and K_means_cluster to be located in your Python path or the same directory.

4. Run main.py file with the provided test_data within the repository. If it is run without any error, change the ‘source_dir’ to your source directory to be clustered and run the main.py again.
```
python3 main.py
```

## Output
- The output is comprised of multiple elements. Here is an overview:

    - Topic Words: These are the key terms or keywords that define a particular topic. They are identified based on the frequency and relevance in the documents being analyzed. For instance, "august, season, average, slugging, home, last, surge" and "award, position, gold, player, may, defensive, model" are the topic words for two documents.
      
    - Average cosine similarity of the best heap: This is a measure of the average similarity between the documents in the best heap. A heap in this context is a collection of documents. Cosine similarity is a metric used to determine how similar two documents are irrespective of their size.
      
    - Best Heap: This contains tuples of document indices, the document's vector representation (in sparse matrix format), the cosine similarity of the document with the centroid of its cluster, and the name of the document. Each tuple corresponds to a document in the dataset.
        ```
        (document_index, document_vector_representation, cosine_similarity, document_name)
        ```
        - For example: (1, <1x4251 sparse matrix of type '<class 'numpy.float64'> with 1740 stored elements in Compressed Sparse Row format>, 1, 'attendance_strike_effects_variable_price_year_season.pdf')

    - Matrix of Data Points: This two-dimensional array corresponds to the data points of the documents in a two-dimensional space. Each array corresponds to a single document, with the two elements in each array being the x and y coordinates of the document in the two-dimensional space.

    - Silhouette Score: This is a measure of how similar an object is to its own cluster compared to other clusters. The silhouette scores range from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

## Expected Output
An example of expected output would look like this:
```
Topic Words: 'august', 'season', 'average', 'slugging', 'home', 'last', 'surge'

Average Cosine Similarity of the Best Heap: 0.89

Best Heap:


[
(10, <1x4251 sparse matrix of type '<class 'numpy.float64'> with 1740 stored elements in Compressed Sparse Row format>, 0.79, 'attendance_strike_effects_variable_price_year_season.pdf'),
(33, <1x4251 sparse matrix of type '<class 'numpy.float64'> with 1563 stored elements in Compressed Sparse Row format>, 0.85, 'second_base_positions_effects_average.pdf'),
...
]

Matrix of Data Points:


[[0.5, 0.7], 
 [0.1, 0.4], 
 [0.3, 0.8], 
 ...
]

Silhouette Score: 0.65
```

## Additional Information
- The document_clf class handles the main functionality. It takes in a source directory of documents and constructs a TF-IDF matrix. This matrix is then used to calculate cosine similarities between documents and form heaps based on these similarities. It can also generate and plot an elbow graph for K-means clustering.

- The create_tf_idf_matrix method constructs the TF-IDF matrix using the sklearn's TfidfVectorizer.

- The get_document_names method extracts all the document names from the source directory.

- The draw_every_heap method constructs heaps for every document, visualizes each heap, and returns the "best" heap (the one that covers the most documents).

- The get_heap method is similar to draw_every_heap but does not produce visualizations for each heap.

- The draw_best_heap function from draw_heap module is used in the main function to visualize the "best" heap.

Note: Please adjust the threshold and k values (the number of most similar documents to return) according to your specific use case in the get_heap and draw_every_heap methods. The threshold is currently set to 0.2 and k to 15.

This script assumes that the documents to be analyzed are in PDF and docx format. Please ensure that the pdf_reader module functions pdf_to_text, read_pdf_tf_idf, and get_combined_token_dict are able to read your documents.

### Draw_Heap

- The scripts require a heap data structure as input. The heap should be a binary tree of tuples where each tuple contains a similarity score and a pair of documents. Replace the placeholders in the draw_heap() and draw_best_heap() functions with your data.

Additional Information
- The draw_heap() function generates a pyramid-like visualization of the heap, with each node labelled with the corresponding document name. The size of the node represents its level in the heap. It saves the visualization as a .png image and displays it.

- The draw_best_heap() function does the same as draw_heap() but is specifically designed to highlight the "best" heap configuration.

Note: The script does not build or manage the heap data structure. It is assumed that you have already generated the heap and are providing it as input to the draw_heap() and draw_best_heap() functions.

### pdf_reader

- This Python script provides functions to extract key topics from .docx and .pdf files and renames the files based on the extracted topics. The script is implemented in Python and utilizes several libraries such as PyPDF2, NLTK, scikit-learn, and a custom module named read_docx_tf_idf.

- Instructions
    -  Install all the required dependencies. You can use pip to install these packages. For example:
        ```
        pip install PyPDF2 nltk scikit-learn
        ```

- This script requires a directory of .pdf and .docx files to be processed. Replace file_path in the main function with the path to your directory.
- This script performs several operations on .docx and .pdf files. It starts by extracting text from the files, tokenizing the text, removing stopwords, and then identifies key topics using Latent Semantic Analysis (LSA).
- The identified key topics are used to rename the files. The original files will be renamed, so please ensure that you have a backup of your files if necessary.
- The read_docx_tf_idf and read_pdf_tf_idf are custom modules used to read .docx and .pdf files respectively. Ensure you have these modules in your working directory or installed in your environment.
- The get_combined_token_dict() function creates a combined dictionary of the extracted text from the .docx and .pdf files and preprocesses the text before further operations.
- Non-English words and words not in the NLTK words corpus are filtered out during preprocessing.

### K_means_cluster
- This Python script implements K-Means clustering on a collection of documents represented in a TF-IDF matrix, identifying optimal cluster count using the Elbow method and visualizing results using word clouds.
- Instructions
    - Install all the required dependencies. You can use pip to install these packages. For example:
        ```
        pip install numpy matplotlib sklearn wordcloud
        ```
    - The scripts require a TF-IDF matrix and a list of corresponding document names as inputs. Replace the placeholders in the main function with your data.
- Additional Information
    - The elbow_method() function calculates and plots the sum of squared distances (inertia) for K values in a specified range. This is used to identify the optimal K (number of clusters) for K-Means clustering.
    - The K_means_cluster() function performs K-Means clustering on the input TF-IDF matrix and returns a dictionary mapping document names to their respective cluster labels.
    - The plot_k_means() function performs K-Means clustering and visualizes the top terms in each cluster with word clouds.

### read_docx_tf_idf

- This script performs text extraction, tokenization, and cleaning of .docx files. It removes stopwords and non-alphabetic words from the text.
- The function read_docx_tf_idf2() reads the .docx files, extracts and cleans the text content, and finally stores the cleaned text in a dictionary.
- The dictionary contains the filename as the key and its cleaned text as the value.
- This script requires a directory of .docx files to be processed. Replace source_dir in the main function with the path to your directory.

### LSA
- Extracting Topics from Documents
    - Our script includes two functions, extract_topic_docx(source_dir, num_topics=3) and extract_topic_pdf(pdf_token, num_topics=3), that leverage Latent Semantic Analysis (LSA) to extract topics from the preprocessed DOCX and PDF files respectively.

- Instructions

    - extract_topic_docx(source_dir, num_topics=3): This function reads the preprocessed DOCX files from the source directory using the read_docx_tf_idf function. The resulting list of preprocessed text data is then transformed into a TF-IDF matrix using TfidfVectorizer. LSA is applied to this matrix using the TruncatedSVD function from sklearn, specifying the number of topics to extract (default is 3). The function then prints out the top 5 words for each topic, giving us an idea of the main themes in the DOCX files.

    - extract_topic_pdf(pdf_token, num_topics=3): This function works similarly to extract_topic_docx but operates on preprocessed text data from PDF files. It transforms the provided text data into a TF-IDF matrix, applies LSA, and then extracts the specified number of topics (default is 3). The function then prints out the top 7 words for each topic. Additionally, it returns the top words from each topic combined with underscores, which could be used for renaming the processed PDF files based on their topics.

    - These two functions enable us to perform topic modeling on our document corpus, providing insights into the main themes present in our data. The number of topics and the number of top words to display for each topic can be adjusted based on your specific needs.

### Hyperparameter Tuning
- The grid_search function has been incorporated into the algorithm to optimize the performance of the document classifier. This function takes in a range of potential k values (the number of documents to return for a single heap structure), and finds the one that provides the highest average cosine similarity in the heap.

- The optimal k value signifies the number of most similar documents that the algorithm should return. The selection of the range of k values to test in the grid search is a critical step, and it can significantly impact the performance of the classifier.

- Here's a sample usage of the grid_search function:

        ```
        # Initialize Document Classifier
        doc_classifier = DocumentClassifier()
        
        # Define range of potential k values
        k_values = range(1, 101)
        
        # Perform grid search to find optimal k
        best_k, best_avg_sim = doc_classifier.grid_search(k_values)
        
        print("The optimal k value is:", best_k)
        print("The average similarity for the optimal k is:", best_avg_sim)
        ```

- The returned optimal k is then used in the get_best_heap_cosine function to determine the heap structure that returns the documents with the highest average similarity. This ensures that the algorithm is tuned for optimal performance for your specific dataset. The average cosine similarity of the best heap signifies the overall relevance of the documents returned by the classifier.

- Please note that the range of k_values passed to the grid_search function may need to be adjusted based on the size and nature of your dataset.

### User Input for K-means Clustering 
- User input part of the script, which is designed to determine the optimal number of clusters to be used in K-means clustering. This is done using the Elbow method, which involves plotting the explained variation as a function of the number of clusters, and picking the elbow of the curve as the number of clusters to use.
  
    - How many clusters do you want to use?: The script first prompts you with this question. Your response should be a positive integer. This number represents the number of clusters ("k") you want to compute inertia for. For example, if you want to compute inertia for two clusters, you'd enter 2.

    - Inertia Computation and Plotting: After you provide your input, the script computes the inertia for each value of "k" from 1 up to the number you entered. Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are. The script will also generate a plot of the explained variation as a function of the number of clusters. The inertia for each value of "k" is then displayed as Intertia for k = x: y, where x is the number of clusters and y is the calculated inertia.

    - Repeat Process: You are then asked again for the number of clusters to use. This allows you to compute inertia for different numbers of clusters and see the plot for each scenario. For instance, if you initially input 2 and then input 3, the script will compute and display the inertia for 1, 2, and then 3 clusters, updating the plot each time.

    - Exit and Selection: If you want to stop the process and proceed with the clustering, you enter 0. The script will use the last number you entered before 0 as the number of clusters for K-means clustering. For instance, if you input 2, then 3, and finally 0, the script will proceed with K-means clustering using 3 clusters.

    - Keyword Word Cloud Generation: After determining the optimal number of clusters, the plot_k_means() function will generate a word cloud. This word cloud comprises the top keywords of each cluster, providing a quick and intuitive visualization of the main topics in each cluster.
    - # Example of Generated Word Clouds

    - Below are examples of word clouds generated by the script for different numbers of clusters. These word clouds show the top keywords for each cluster.

    - ## Word Cloud for 3 Clusters
    - ![1 Clusters Word Cloud](./main_code/word_cloud_figures/cluster_1_wordcloud.png)
    - ![1 Clusters Word Cloud](./main_code/word_cloud_figures/cluster2_wordcloud.png)
    - ![1 Clusters Word Cloud](./main_code/word_cloud_figures/cluster3_wordcloud.png)

- This interactive approach allows you to explore different cluster quantities and their respective inertia, helping you select the number of clusters that minimizes inertia and optimizes the clustering process. Note: Elbow method is a heuristic method of interpretation and visualization of the explained variation as a function of the number of clusters. You'll need to interpret the plot to choose the "elbow point," the point of inflection on the curve.




