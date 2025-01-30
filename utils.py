import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError
import json
from nltk.corpus import stopwords
import random
import re
import nltk


# Load JSON data from a file
def load_json_data(file_path):
    try:
        return pd.read_json(file_path)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None


# Save trained model
def save_model(model, model_path):
    try:
        joblib.dump(model, model_path)
        print(f"Model saved at {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


# Load pre-trained model if it exists
def load_trained_model(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except (FileNotFoundError, NotFittedError) as e:
        print(f"Model not found or not trained yet: {e}")
        return None


# Save the classification report to a file
def save_report(report, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Classification report saved at {file_path}")
    except Exception as e:
        print(f"Error saving report: {e}")


# # Download NLTK stopwords if not already downloaded
# nltk.download('stopwords')


# Function to preprocess the text
def preprocess_text(text):
    # Check if stopwords are already downloaded, if not, download them
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    # Step 1: Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(cleaned_words)
    # Step 2: Replace \n with space
    text = text.replace('\n', ' ')
    # Step 3: Replace '*' with space
    text = text.replace('*', ' ')
    # Step 4: Replace **** with 4 random numeric digits
    text = re.sub(r'\*{4}', lambda x: str(random.randint(1000, 9999)), text)
    # Step 5: Remove extra spaces (replace multiple spaces with one)
    text = re.sub(r'\s+', ' ', text)
    # Step 6: Remove leading and trailing spaces
    text = text.strip()
    return text