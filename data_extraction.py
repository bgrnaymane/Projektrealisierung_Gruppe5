import pandas as pd
import os

text_folder_path = './data/BBC News Summary/News Articles'
summary_folder_path = './data/BBC News Summary/Summaries'

data = []
# Durchlaufe die Textordner und lese die Texte ein
for category_folder in os.listdir(text_folder_path):
    category_path = os.path.join(text_folder_path, category_folder)
    if os.path.isdir(category_path):
        # Durchlaufe die Textdateien in jedem Kategorieordner
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
                category = category_folder
                
                # Finde den entsprechenden Zusammenfassungsordner
                summary_category_path = os.path.join(summary_folder_path, category_folder)
                if os.path.isdir(summary_category_path):
                    summary_file_name = file_name
                    summary_file_path = os.path.join(summary_category_path, summary_file_name)
                    with open(summary_file_path, 'r', encoding='latin-1') as summary_file:
                        summary = summary_file.read()
                        data.append({'Text': text, 'Summary': summary, 'Class': category})

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Save the dataset as csv-file
save_path = os.path.join('./data/', 'dataset.csv')
df.to_csv(save_path, index=False)