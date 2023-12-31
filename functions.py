import pickle
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import os

from docx import Document
from odf import text, teletype
from odf.opendocument import load

# For the classification
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.corpus import wordnet as wn

# For the summary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import spacy
import re
from gensim.models import Word2Vec
from scipy import spatial
from scipy import sparse
import networkx as nx
from flask import current_app

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )

classes_mapping = {
    'Literature': 0,
    'News': 1,
    'Blog': 2,
    'Political speech': 3,
    'Jurisdiction': 4
}

# Source: https://automatetheboringstuff.com/chapter13/#:~:text=Reading%20Word%20Documents&text=docx%20file%20in%20Python%2C%20call,a%20list%20of%20Paragraph%20objects.
def read_docx_file(file_path):
    doc = Document(file_path)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    file_content = '\n'.join(paragraphs)
    return file_content

# Source: https://stackoverflow.com/questions/51054770/how-to-read-odt-using-python
def read_odt_file(file):
    doc = load(file)
    content = []
    for elem in doc.getElementsByType(text.P):
        text_content = teletype.extractText(elem)
        content.append(text_content)
    file_content = "\n".join(content)
    return file_content

# Function for Tokenization, Remove stopwords, Lowercasing, Lemmatization, Remove punctuation
def preprocess_text(text):
    # Source: https://stackoverflow.com/questions/18214612/how-to-access-app-config-in-a-blueprint
    # Tokenization
    nlp = current_app.config['nlp']
    doc = nlp(text)
    # Source: https://stackoverflow.com/questions/64185831/am-i-missing-the-preprocessing-function-in-spacys-lemmatization
    # Remove stopwords, Lowercasing, Lemmatization, Remove punctuation
    preprocessed_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

    # Return the preprocessed tokens as text
    preprocessed_text = " ".join(preprocessed_tokens)
    return preprocessed_text

def textrank_summarizer(text, compression_rate):
    compression_rate = compression_rate/100
    # Split text sentence
    sentences = sent_tokenize(text) # https://www.guru99.com/tokenize-words-sentences-nltk.html
    sort_dict = {}
    for i in range(len(sentences)):
        sort_dict[i] = sentences[i]

    preprocessed_sentences = []
    for sentence in sentences:
        preprocessed_sentences.append(preprocess_text(sentence))
    sentence_tokens = []
    for sentence in preprocessed_sentences:
        sentence_tokens.append(word_tokenize(sentence))

    w2v=Word2Vec(sentence_tokens,vector_size=1,min_count=1, epochs=1000)
    sentence_embeddings = [[w2v.wv[word][0] for word in words] for words in sentence_tokens]
    max_len=max([len(tokens) for tokens in sentence_tokens])
    sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]
    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i,row_embedding in enumerate(sentence_embeddings):
        for j,column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    try:
        scores = nx.pagerank(nx_graph, max_iter=2000, tol=1e-6)
    except:
        return 'Error'
    sentence_scores = list(scores.values())
    # Print the sentence scores
    ranking = sorted(range(len(sentence_scores)), key=lambda x: sentence_scores[x], reverse=True)

    ranking_texts = [sort_dict[index] for index in ranking]

    max_words = len(word_tokenize(text)) * (1-compression_rate)
    words=0
    chosen_texts = []
    for text in ranking_texts:
        if words<=max_words:
            sen_length = len(word_tokenize(text)) # https://www.guru99.com/tokenize-words-sentences-nltk.html
            words += sen_length
            chosen_texts.append(text)
    
    chosen_dict = {}
    for i in range (len(sort_dict)):
        for text in chosen_texts:
            if sort_dict[i] == text:
                chosen_dict[i] = text

    summary_sentences = chosen_dict.values()
    summary = ' '.join(summary_sentences)

    return summary

# Source: 
# - https://iq.opengenus.org/latent-semantic-analysis-for-text-summarization/
# - https://towardsdatascience.com/document-summarization-using-latent-semantic-indexing-b747ef2d2af6
# - https://github.com/luisfredgs/LSA-Text-Summarization

def lsa_summarizer(text, compression_rate):
    compression_rate = compression_rate/100
    # Split text sentence
    sentences = sent_tokenize(text) # https://www.guru99.com/tokenize-words-sentences-nltk.html
    sort_dict = {}
    for i in range(len(sentences)):
        sort_dict[i] = sentences[i]

    preprocessed_sentences = []
    for sentence in sentences:
        preprocessed_sentences.append(preprocess_text(sentence))
    # Text Vectorization
    vectorizer = CountVectorizer()
    term_document_matrix = vectorizer.fit_transform(preprocessed_sentences)
    # LSA-Model
    num_components = max(int(len(preprocessed_sentences) * compression_rate), 1)
    lsa_model = TruncatedSVD(n_components=num_components)
    lsa_matrix = lsa_model.fit_transform(term_document_matrix)

    # Ranking sentences
    sentence_scores = lsa_matrix.sum(axis=1)
    
    # Print the sentence scores
    ranking = sorted(range(len(sentence_scores)), key=lambda x: sentence_scores[x], reverse=True)

    ranking_texts = [sort_dict[index] for index in ranking]

    max_words = len(word_tokenize(text)) * (1-compression_rate)
    words=0
    chosen_texts = []
    for text in ranking_texts:
        if words<=max_words:
            sen_length = len(word_tokenize(text)) # https://www.guru99.com/tokenize-words-sentences-nltk.html
            words += sen_length
            chosen_texts.append(text)
    
    chosen_dict = {}
    for i in range (len(sort_dict)):
        for text in chosen_texts:
            if sort_dict[i] == text:
                chosen_dict[i] = text

    summary_sentences = chosen_dict.values()
    summary = ' '.join(summary_sentences)

    return summary

def text_classification(user_text):
    # Source: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

    # Change to lowercase
    modified_text = user_text.lower()
    # Tokenization
    modified_text = word_tokenize(modified_text)

    preprocessed_text = []

    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(modified_text):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    preprocessed_text.append(str(Final_words))

    # Load the TF-IDF vectorizer
    filename = './vectorizers/tfidf-vectorizer.sav'
    Tfidf_vect = pickle.load(open(filename, 'rb'))

    transformed_text = Tfidf_vect.transform(preprocessed_text)

    # load the created model
    filename = './models/finalized_model.sav'
    print(filename)
    loaded_model = pickle.load(open(filename, 'rb'))
    print(loaded_model)

    # Get prediction in the form of probabilities
    prediction = loaded_model.predict_proba(transformed_text) # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    print(prediction)

    # Get the predicted class
    predicted_class = [k for k, v in classes_mapping.items() if v == np.argmax(prediction[0])]
    # Get the probabilitie for the predicted class
    predicted_value_max = max(prediction[0])

    return predicted_class[0], predicted_value_max