import pandas as pd
from gensim.summarization import summarize

def summarize_text(text):
    return summarize(text)

def summarize_csv(input_file, output_file, text_column='text', summary_column='summary'):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Create a new column for the summary
    df[summary_column] = df[text_column].apply(summarize_text)

    # Save the DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Beispielaufruf
input_file = 'test.csv'
output_file = 'output.csv'
text_column = 'article'
summary_column = 'summary'

summarize_csv(input_file, output_file, text_column, summary_column)