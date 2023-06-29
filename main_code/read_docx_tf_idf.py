import warnings
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize
import docx
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def read_docx_tf_idf(source_dir):
    """Read docx files from a given directory and extract their text content then changes them into tf_idf_matrix."""
    # Get a list of all docx files in the source directory
    docx_files = sorted(os.listdir(source_dir))

    # Initialize empty lists to store the document text and tokenized text
    doc_dict = dict()
    token_lst = []
    stop_words = set(stopwords.words('english'))
    # Loop through each file in the directory and extract its text content
    print("Top 5 keywords for each document")
    for file_name in docx_files:
        if file_name.endswith(".docx"):
            # Get the full path of the file
            file_path = source_dir + '/' + file_name

            # Open the docx file and extract its text content
            doc = docx.Document(file_path)
            fullText = []
            for para in doc.paragraphs:
                fullText.append(para.text)
            mystring =  '\n'.join(fullText)

            # Tokenize the text content and remove stop words and non-alphabetic words
            tokens = word_tokenize(mystring)
            tokens = [w for w in tokens if not w in stop_words]
            tokens = [word.lower() for word in tokens if word.isalpha()]

            # Add the tokenized text to the list of tokenized texts
            token_lst.append(tokens)

            # Join the tokens back into a string and add it to the list of document text
            doc_string = ' '.join(tokens)
            doc_dict[file_name] = doc_string

    return doc_dict
