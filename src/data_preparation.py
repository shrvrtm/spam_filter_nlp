import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path,encoding='latin-1')
    #оставляем только нужные столбцы
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    return data

def preprocess_labels(data):
    data['label'] = data['label'].map({'ham':0, 'spam':1})
    return data