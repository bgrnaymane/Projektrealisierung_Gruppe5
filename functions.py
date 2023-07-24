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
    allparas = doc.getElementsByType(text.P)
    file_content = teletype.extractText(allparas[0])
    return file_content

# Source: 
# - https://iq.opengenus.org/latent-semantic-analysis-for-text-summarization/
# - https://towardsdatascience.com/document-summarization-using-latent-semantic-indexing-b747ef2d2af6
# - https://github.com/luisfredgs/LSA-Text-Summarization

def text_summary(text, compression_rate):
    compression_rate = int(compression_rate)/100
    # Split text sentence
    sentences = sent_tokenize(text) # https://www.guru99.com/tokenize-words-sentences-nltk.html
    sort_dict = {}
    for i in range(len(sentences)):
        sort_dict[i] = sentences[i]

    # Text Vectorization
    vectorizer = CountVectorizer()
    term_document_matrix = vectorizer.fit_transform(sentences)

    # LSA-Model
    num_components = max(int(len(sentences) * (1-compression_rate)), 1)
    lsa_model = TruncatedSVD(n_components=num_components)
    lsa_matrix = lsa_model.fit_transform(term_document_matrix)

    # Ranking sentences
    sentence_scores = lsa_matrix.sum(axis=1)
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
    summary = '. '.join(summary_sentences)

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
    loaded_model = pickle.load(open(filename, 'rb'))

    # Get prediction in the form of probabilities
    prediction = loaded_model.predict_proba(transformed_text) # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    print(prediction)

    # Get the predicted class
    predicted_class = [k for k, v in classes_mapping.items() if v == np.argmax(prediction[0])]
    # Get the probabilitie for the predicted class
    predicted_value_max = max(prediction[0])

    return predicted_class[0], predicted_value_max







# def preprocessing(input_text, tokenizer):
#   return tokenizer.encode_plus(
#                         input_text,
#                         add_special_tokens = True,
#                         max_length = 32,
#                         pad_to_max_length = True,
#                         return_attention_mask = True,
#                         return_tensors = 'pt'
#                    )


# def text_classification(user_text):

#     # When a gpu is available it gets used. Otherwise the cpu gets used
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # load the created model
#     filename = './models/finalized_model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))

#     # Preprocessing of the text to be used by the Bert-Model
#     test_ids = []
#     test_attention_mask = []
#     encoding = preprocessing(user_text, tokenizer)
#     test_ids.append(encoding['input_ids'])
#     test_attention_mask.append(encoding['attention_mask'])
#     test_ids = torch.cat(test_ids, dim = 0)
#     test_attention_mask = torch.cat(test_attention_mask, dim = 0)

#     # Get the prediction from the model
#     with torch.no_grad():
#         output = loaded_model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))
#     keys = [k for k, v in classes_mapping.items() if v == np.argmax(output.logits.cpu().numpy()).flatten().item()]
#     value_prediction = torch.max(output.logits).item()

#     # Print the output-tensor
#     print(output)

#     # Get the predicted class
#     predicted_class = keys[0]

#     # Return the predicted class and the value for this prediction, so that the treshold value can be used
#     return predicted_class, value_prediction