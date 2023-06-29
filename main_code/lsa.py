import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from read_docx_tf_idf import read_docx_tf_idf


def extract_topic_docx(source_dir,num_topics=3):
    
    doc_lst = read_docx_tf_idf(source_dir)[0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(doc_lst)

    lsa = TruncatedSVD(n_components=num_topics)  # Specify the number of topics to extract
    topic_matrix = lsa.fit_transform(X)

    terms = vectorizer.get_feature_names_out()
    num_top_words = 5 # Number of top words to display for each topic

    print("Top 5 keywords for each document"+"\n"+"__________________________")
    for i, component in enumerate(lsa.components_):
        top_words = [terms[index] for index in component.argsort()[:-num_top_words - 1:-1]]
        print(f'Topic {i+1}: {",".join(top_words)}')

def extract_topic_pdf(pdf_token,num_topics=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(pdf_token)
    lsa = TruncatedSVD(n_components=num_topics)  # Specify the number of topics to extract
    topic_matrix = lsa.fit_transform(X)

    terms = vectorizer.get_feature_names_out()
    num_top_words = 7 # Number of top words to display for each topic

    topics = []
    
    for i, component in enumerate(lsa.components_):
        top_words = [terms[index] for index in component.argsort()[:-num_top_words - 1:-1]]
        print(f'Topic words: {",".join(top_words)}')
        topic_word = "_".join(top_words)
    return topic_word
    
