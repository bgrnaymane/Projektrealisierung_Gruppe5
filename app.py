from flask import Flask, render_template, request
from functions import text_classification, text_summary, read_docx_file, read_odf_file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_text = ''
    file = request.files['txt_file']
    if file:
        if file.filename.endswith('.docx'):
            user_text = read_docx_file(file)
        elif file.filename.endswith('.odt'):
            user_text = read_odf_file(file)
        elif file.filename.endswith('.txt'):
            user_text = file.read().decode('utf-8') 
        else:
            return 'Invalid file format. Only .txt, .docx and .odt files are allowed.'
    else:
        user_text = request.form['user_text']
    compression_rate = request.form['compression_rate']
    user_summary = text_summary(user_text, compression_rate)
    predicted_class, value_prediction = text_classification(user_text)

    if value_prediction >= 1:
        pred_class = predicted_class
    else: 
        pred_class = 'Der Text konnte nicht klassifiziert werden.'
    return render_template('result.html', user_text=user_text, compression_rate=compression_rate, user_summary=user_summary, predicted_class=pred_class)
