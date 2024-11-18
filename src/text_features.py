from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(messages):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(messages)
    return X, vectorizer