from src.data_preparation import load_data, preprocess_labels
from src.text_features import vectorize_text
from src.train_model import train_model, save_model, file_exists, load_model
from src.predict import predict_spam

input_path = 'input_output/input.txt'
output_path = 'input_output/output.txt'
path_to_model = 'models/spam_filter_model.pkl'
path_to_vectorizer = 'models/tfidf_vectorizer.pkl'

if (not file_exists(path_to_model)) & (not file_exists(path_to_model)):
    data = load_data('data/spam.csv')
    data = preprocess_labels(data)
    X, vectorizer = vectorize_text(data['message'])
    y = data['label']
    model = train_model(X, y)
    save_model(model, vectorizer)
else:
    model = load_model(path_to_model)
    vectorizer = load_model(path_to_vectorizer)

print(predict_spam(model, vectorizer, input_path, output_path))