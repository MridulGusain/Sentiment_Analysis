# training.py
import os
import re
import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Ensure NLTK dependencies are downloaded
import nltk
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(content, stemmer, stop_words):
    content = re.sub('[^a-zA-Z]', ' ', content)  # Keep only alphabetic characters
    content = content.lower().split()  # Convert to lowercase and split into words
    content = [stemmer.stem(word) for word in content if word not in stop_words]  # Remove stopwords and stem words
    return ' '.join(content)

# Paths for saving/loading models
BASE_PATH = os.path.abspath('.')
MODEL_PATH = os.path.join(BASE_PATH, 'trained_model.sav')
VECTORIZER_PATH = os.path.join(BASE_PATH, 'vectorizer.sav')

# Load data
def load_data():
    try:
        file_path = 'C:/Users/acer/Videos/Captures/training.1600000.processed.noemoticon.csv'
        column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
        data = pd.read_csv(file_path, names=column_names, encoding='ISO-8859-1')
        return data
    except FileNotFoundError:
        print("Dataset not found. Please ensure the dataset is in the script directory.")
        return None

# Main function for training

def train_model():
    data = load_data()
    if data is None:
        return

    # Preprocessing
    print("Initial target distribution:")
    print(data['target'].value_counts())

    data['target'] = data['target'].replace({4: 1})  # Convert 4 to 1 for binary classification

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    data['stemmed_content'] = data['text'].apply(lambda x: preprocess_text(x, stemmer, stop_words))

    # Balance the dataset
    majority = data[data.target == 1]
    minority = data[data.target == 0]

    minority_upsampled = resample(
        minority,
        replace=True,  # Sample with replacement
        n_samples=len(majority),  # Match majority count
        random_state=42  # Reproducibility
    )

    data_balanced = pd.concat([majority, minority_upsampled])
    print("Balanced target distribution:")
    print(data_balanced['target'].value_counts())

    # Split data
    X = data_balanced['stemmed_content'].values
    Y = data_balanced['target'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    # Vectorize
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Save vectorizer
    pickle.dump(vectorizer, open(VECTORIZER_PATH, 'wb'))

    # Train model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, Y_train)

    # Save model
    pickle.dump(model, open(MODEL_PATH, 'wb'))

    # Evaluate model
    train_accuracy = accuracy_score(Y_train, model.predict(X_train))
    test_accuracy = accuracy_score(Y_test, model.predict(X_test))

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    train_model()

