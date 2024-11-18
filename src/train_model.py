from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import load, dump
import os
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

def save_model(model, vectorizer):
    dump(model, 'models/spam_filter_model.pkl')
    dump(vectorizer, 'models/tfidf_vectorizer.pkl')

def load_model(path_to_model):
    model = load(path_to_model)
    return model
def file_exists(file_path):
    return os.path.isfile(file_path)