
# Document Classifier and Heap Visualization

This Python script creates a document classifier based on the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm and visualizes heap data structures generated from document similarity using the networkx and matplotlib libraries.


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

3. You will need the modules draw_heap, pdf_reader, and K_means_cluster to be located in your Python path or the same directory.

4. Update the source_dir and small_source_dir in the main function with the directory of your documents.

5. To run the script, navigate to the directory containing the script and use the following command in the terminal:
```
python3 document_classification.py
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
Instructions

Ensure you have Python 3 installed. You can download it from here.
- Install all the required dependencies. You can use pip to install these packages. For example:

```
pip install matplotlib networkx scipy nltk
```
- Clone or download this repository to your local machine.

- You will need the module read_docx_tf_idf to be located in your Python path or the same directory. The read_docx_tf_idf module should provide a function that reads a .docx file and transforms it into a TF-IDF matrix.

- The scripts require a heap data structure as input. The heap should be a binary tree of tuples where each tuple contains a similarity score and a pair of documents. Replace the placeholders in the draw_heap() and draw_best_heap() functions with your data.

- To run the script, navigate to the directory containing the script and use the following command in the terminal:

Additional Information
- The draw_heap() function generates a pyramid-like visualization of the heap, with each node labelled with the corresponding document name. The size of the node represents its level in the heap. It saves the visualization as a .png image and displays it.

- The draw_best_heap() function does the same as draw_heap() but is specifically designed to highlight the "best" heap configuration, according to your specified criteria.

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
