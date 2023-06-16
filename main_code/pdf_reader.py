# Import necessary libraries and suppress warnings
import os
import re
import shutil
import warnings
from nltk.corpus import words, stopwords
from nltk.tokenize import RegexpTokenizer
from PyPDF2 import PdfReader
from read_docx_tf_idf import read_docx_tf_idf
from lsa import extract_topic_pdf
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')

# Get a set of English words
english_words = set(words.words())

def remove_tuple_brackets(string):
    """Remove brackets in tuples."""
    pattern = r'\(([^()]+)\)'  # Regular expression pattern to match tuples with characters inside brackets
    result = re.sub(pattern, r'\1', string)  # Remove tuple brackets, keeping the characters inside
    return result

def preprocess_text(text):
    """Preprocess text by tokenizing, removing non-English words and joining back."""
    # Tokenize the input text
    words_in_text = text.split()

    # Filter out non-English words
    words = [word for word in words_in_text if word in english_words]

    # Handle splitting of long words - this is a simple heuristic and may not work perfectly
    i = 0
    while i < len(words) - 1:
        combined_word = words[i] + words[i+1]
        if combined_word in english_words:
            words[i] = combined_word
            del words[i+1]
        else:
            i += 1

    # Join words back into a string
    text = ' '.join(words)

    return text

def pdf_to_text(file_path):
    """Extract text from pdf file."""
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')  # This will effectively remove punctuation marks
    text_arr = []

    # creating a pdf reader object
    reader = PdfReader(file_path)
    max_page = len(reader.pages)

    for i in range(max_page):
        page = reader.pages[i]
        # extracting text from page
        text = page.extract_text()
        words = tokenizer.tokenize(text.lower())  # Tokenize and convert to lower case

        words = [word for word in words if not word in stop_words]  # Remove stopwords
        text_arr.extend(words)

    return text_arr
    
def read_pdf_tf_idf(source_dir):
    """Read PDF files from a given directory, extract text, convert to TF-IDF matrix and rename the file."""
    pdf_files = sorted(os.listdir(source_dir))
    pdf_lst = []
    token_lst = []
    stop_words = set(stopwords.words('english'))
    pdf_token_dict = {}

    for file_name in pdf_files:
        if file_name.endswith(".pdf"):
            file_path = os.path.join(source_dir, file_name)
            try:
                pdf_token = pdf_to_text(file_path)
                pdf_string = ' '.join(pdf_token)  # Join the preprocessed words back into a string
                pdf_string = preprocess_text(pdf_string)
                pdf_string = re.sub(r'\([^()]+\)', '', pdf_string)  # Remove brackets and their contents
                token_lst = [pdf_string]
                file_name_string = extract_topic_pdf(token_lst)

                # Separate the extension from the filename
                basename, extension = os.path.splitext(file_name)
                new_filename = file_name_string + extension

                # Create full paths for old and new filename
                old_file = os.path.join(source_dir, file_name)
                new_file = os.path.join(source_dir, new_filename)

                # Rename the file (be careful with this, it will overwrite existing files)
                shutil.move(old_file, new_file)

                
                pdf_token_dict[file_name] = pdf_string

            except:
                print(f"Error reading {file_name}:")
                continue

    return pdf_token_dict

def get_combined_token_dict(file_path):
    """Combine the token dictionaries from pdf and docx files."""
    pdf_dict = read_pdf_tf_idf(file_path)
    docx_dict = read_docx_tf_idf(file_path)
    combined_dict = {}
    combined_dict.update(pdf_dict)
    combined_dict.update(docx_dict)
    combined_dict = {k: v for k, v in sorted(combined_dict.items(), key=lambda item: item[0])}

    token_dict = {}
    tokens_lst = list(combined_dict.values())
    for key,value in combined_dict.items():
        token_dict[key] = preprocess_text(value)

    return token_dict


if __name__ == "__main__":
    file_path = "/Users/junhoeum/Desktop/Summer_23/doc_clustering_algo/test_document_3_27_23/test_sample"
    combined_dict = get_combined_token_dict(file_path)
    # feature_names = rename_pdf_files(file_path)
    # print(len(feature_names))


    
