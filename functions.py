import pickle
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import os

from docx import Document
from odf import text, teletype
from odf.opendocument import load

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

def read_docx_file(file_path):
    doc = Document(file_path)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    file_content = '\n'.join(paragraphs)
    return file_content

# Source: https://stackoverflow.com/questions/51054770/how-to-read-odt-using-python
def read_odf_file(file):
    doc = load(file)
    allparas = doc.getElementsByType(text.P)
    file_content = teletype.extractText(allparas[0])
    return file_content

def text_summary(user_text, compression_rate):
    summary = 'Das ist eine Testzusammenfassung'
    return summary

def preprocessing(input_text, tokenizer):
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 32,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )


def text_classification(user_text):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model from disk
    filename = './models/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # We need Token IDs and Attention Mask for inference on the new sentence
    test_ids = []
    test_attention_mask = []

    # Apply the tokenizer
    encoding = preprocessing(user_text, tokenizer)

    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    # Forward pass, calculate logit predictions
    with torch.no_grad():
        output = loaded_model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))
    keys = [k for k, v in classes_mapping.items() if v == np.argmax(output.logits.cpu().numpy()).flatten().item()]
    value_prediction = torch.max(output.logits).item()
    print(output)
    print(value_prediction)
    predicted_class = keys[0]

    return predicted_class, value_prediction