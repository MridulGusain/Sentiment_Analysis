import pickle
import re
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tkinter import Tk, Label, Entry, Button, Text, END

# Ensure NLTK dependencies are downloaded
import nltk
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(content, stemmer, stop_words):
    content = re.sub('[^a-zA-Z]', ' ', content)  # Keep only alphabetic characters
    content = content.lower().split()  # Convert to lowercase and split into words
    content = [stemmer.stem(word) for word in content if word not in stop_words]  # Remove stopwords and stem words
    return ' '.join(content)

# Function to fetch tweet content from a Twitter URL
def fetch_tweet_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return "Failed to fetch the tweet. Please check the URL."

        soup = BeautifulSoup(response.text, 'html.parser')
        tweet_content = soup.find('meta', attrs={'property': 'og:description'})
        if tweet_content:
            return tweet_content['content']
        else:
            return "Could not extract tweet content."
    except Exception as e:
        return f"An error occurred while fetching the tweet: {e}"

# Paths for loading models
MODEL_PATH = 'trained_model.sav'
VECTORIZER_PATH = 'vectorizer.sav'

def predict_sentiment():
    tweet_url = tweet_entry.get()

    # Fetch tweet content
    tweet_text = fetch_tweet_content(tweet_url)
    if "error" in tweet_text.lower() or "failed" in tweet_text.lower():
        result_text.delete(1.0, END)
        result_text.insert(END, tweet_text)
        return

    try:
        # Load model and vectorizer
        model = pickle.load(open(MODEL_PATH, 'rb'))
        vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    except FileNotFoundError:
        result_text.delete(1.0, END)
        result_text.insert(END, "Model or vectorizer files not found. Please train the model first.")
        return

    # Preprocess input
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    preprocessed_text = preprocess_text(tweet_text, stemmer, stop_words)
    vectorized_text = vectorizer.transform([preprocessed_text])

    # Predict
    prediction = model.predict(vectorized_text)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    result_text.delete(1.0, END)
    result_text.insert(END, f"Tweet Content: {tweet_text}\n")
    result_text.insert(END, f"Tweet Sentiment: {sentiment}")

if __name__ == "__main__":
    root = Tk()
    root.title("Sentiment Analysis")

    Label(root, text="Enter URL:").pack(pady=5)
    tweet_entry = Entry(root, width=50)
    tweet_entry.pack(pady=5)

    Button(root, text="Analyze Sentiment", command=predict_sentiment).pack(pady=10)

    result_text = Text(root, height=10, width=60)
    result_text.pack(pady=5)

    root.mainloop()
