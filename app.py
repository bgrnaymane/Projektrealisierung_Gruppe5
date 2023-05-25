from flask import Flask, render_template, request
from functions import text_classification, text_summary

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_text = request.form['user_text']
    compression_rate = request.form['compression_rate']
    user_summary = text_summary(user_text, compression_rate)
    text_class = text_classification(user_text)
    return render_template('result.html', user_text=user_text, compression_rate=compression_rate, user_summary=user_summary, text_class=text_class)
