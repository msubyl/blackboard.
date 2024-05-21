import time
import streamlit as st
import pickle
import re
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import threading

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load the trained model and the vectorizer
model_file_path = 'logistic_model.pkl'
vectorizer_file_path = 'vectorizer.pkl'

with open(model_file_path, 'rb') as model_file:
    LG_model = pickle.load(model_file)
with open(vectorizer_file_path, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Custom stop words
custom_stop_words = {
    'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'that', 'for', 'on', 'with', 'as', 'this', 'was', 'are', 'be', 'at', 'but', 'by', 
    'not', 'or', 'from', 'an', 'my', 'have', 'has', 'you', 'i', 'me', 'we', 'do', 'so', 'can', 'if', 'its', 'about', 'all', 'he', 'she', 
    'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'there', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 
    'your', 'his', 'her', 'our', 'us', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'here', 'there', 'when', 
    'where', 'why', 'how', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
    'than', 'too', 'very', 's', 't', 'don', 'just', 'now'
}

stop_words = set(stopwords.words('english')).union(custom_stop_words)

# Text preprocessing function
def remove_stop_words(text):
    words = TextBlob(text).words
    return " ".join([w for w in words if w not in stop_words and len(w) >= 2])

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('\d', '', text)
    text = remove_stop_words(text)
    return text

# Sentiment classification function
def classify_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = LG_model.predict(vectorized_text)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# FileSystemEventHandler subclass to watch for file changes
class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == model_file_path or event.src_path == vectorizer_file_path:
            st.experimental_rerun()

# Function to start the file watcher
def start_watcher():
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Start the file watcher in a separate thread
watcher_thread = threading.Thread(target=start_watcher, daemon=True)
watcher_thread.start()

def main():
    st.title("Sentiment Analysis")
    html_temp = """
    <div style="background-color:teal; padding:10px">
    <h2 style="color:white; text-align:center;">Sentiment Analysis</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    review = st.text_input('Enter your review:', '')

    if st.button('Classify'):
        if review:
            sentiment = classify_sentiment(review)
            st.success(f'The sentiment of the review is {sentiment}.')
        else:
            st.error('Please enter a review.')

if __name__ == '__main__':
    main()
