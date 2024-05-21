import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_files
import nltk
from nltk.corpus import stopwords
import string

# Download the stopwords from NLTK
nltk.download('stopwords')

# Load your dataset (replace this with your own dataset)
reviews = load_files('C:\Users\Asus\Desktop\MLP\blackboard dataset.csv')  # Assuming the dataset is in the 'txt_sentoken' directory
X, y = reviews.data, reviews.target

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply text preprocessing to the dataset
X = [preprocess_text(review.decode('utf-8')) for review in X]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)