import pandas as pd
import requests

def summarize_text(text, api_token):
    # Hugging Face API URL
    url = "https://api-inference.huggingface.co/models/t5-base"

    # Set request headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # Prepare request payload
    payload = {
        "inputs": text,
        "options": {
            "task": "summarization",
            "model": "t5-base",
            "num_return_sequences": 1,
            "num_beams": 4,
            "do_sample": True
        }
    }

    # Send POST request to Hugging Face API
    response = requests.post(url, json=payload, headers=headers)

    # Get the summarized text from the response
    summarized_text = response.json()[0]['generated_text']
    return summarized_text

def summarize_csv(input_file, output_file, text_column='text', summary_column='summary', api_token=''):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Create a new column for the summary
    df[summary_column] = df[text_column].apply(lambda x: summarize_text(x, api_token))

    # Save the DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Beispielaufruf
input_file = 'test.csv'
output_file = 'output.csv'
text_column = 'article'
summary_column = 'summary'
api_token = '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'

summarize_csv(input_file, output_file, text_column, summary_column, api_token)
