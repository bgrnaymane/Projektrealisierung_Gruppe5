from flask import Flask, render_template, request
from functions import text_classification, lsa_summarizer, textrank_summarizer, read_docx_file, read_odt_file
import spacy
app = Flask(__name__)

@app.route('/')
def index():
    # https://machinelearningprojects.net/deploy-a-flask-app-online/
    app.config['nlp'] = spacy.load("en_core_web_sm")
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_text = ''

    # Read the file by the user input
    file = request.files['txt_file']

    # If a file was uploaded
    if file:
        if file.filename.endswith('.docx'):
            user_text = read_docx_file(file)
        elif file.filename.endswith('.odt'):
            user_text = read_odt_file(file)
        elif file.filename.endswith('.txt'):
            user_text = file.read().decode('utf-8') 
        else:
            return 'Invalid file format. Only .txt, .docx and .odt files are allowed.'
    else:
        user_text = request.form['user_text']
    
    # Check the functionalities the user wants to access
    classificationCheckbox = request.form.get('classificationCheckbox')
    summaryCheckbox = request.form.get('summaryCheckbox')

    # Return what the user wants
    if classificationCheckbox and not summaryCheckbox:
        predicted_class, value_prediction = text_classification(user_text)
        if value_prediction >= 0.65:
            pred_class = predicted_class
        else: 
            pred_class = f'Text classification failed. The class with the highest probability is: {predicted_class}'
        return render_template('result.html', user_text=user_text, predicted_class=pred_class)
    
    elif summaryCheckbox and not classificationCheckbox:
        compression_rate = int(request.form['compression_rate'])
        user_summary = lsa_summarizer(user_text, compression_rate)
        actual_compression_rate = str(100 - (round(len(user_summary.split()) / len(user_text.split()) * 100)))
        return render_template('result.html', user_text=user_text, compression_rate=compression_rate, user_summary=user_summary, actual_compression_rate=actual_compression_rate)
    
    elif classificationCheckbox and summaryCheckbox:
        compression_rate = int(request.form['compression_rate'])
        predicted_class, value_prediction = text_classification(user_text)
        user_summary = lsa_summarizer(user_text, compression_rate)
        actual_compression_rate = str(100 - (round(len(user_summary.split()) / len(user_text.split()) * 100)))
        if value_prediction >= 0.65:
            pred_class = predicted_class
        else: 
            pred_class = f'Text classification failed. The class with the highest probability is: {predicted_class}'
        return render_template('result.html', user_text=user_text, compression_rate=compression_rate, user_summary=user_summary, predicted_class=pred_class, actual_compression_rate=actual_compression_rate)