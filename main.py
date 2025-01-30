import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from utils import save_model, load_json_data, load_trained_model, save_report, preprocess_text

cwd = os.getcwd()
print(cwd)


class DocumentClassifier:
    def __init__(self, model=None):
        self.model = model
        self.label_encoder = LabelEncoder()
        self.embed_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

        # (len(X), 1, 384)  -> (len(X), 384)

    def get_embeddings(self, text):
        preprocessed_text = preprocess_text(text)
        embeddings = self.embed_model.encode([preprocessed_text])
        return embeddings

    def fit(self, X, y):
        # Convert the document texts to embeddings
        y_encoded = self.label_encoder.fit_transform(y)
        X_embeddings = np.array([self.get_embeddings(text) for text in X]).squeeze(axis=1)
        self.model.fit(X_embeddings, y_encoded)
        return self

    def predict(self, X):
        X_embeddings = np.array([self.get_embeddings(text) for text in X]).squeeze(axis=1)
        y_predicted_encoded = self.model.predict(X_embeddings)
        return self.label_encoder.inverse_transform(y_predicted_encoded)

    def predict_proba(self, X):
        X_embeddings = np.array([self.get_embeddings(text) for text in X]).squeeze(axis=1)
        return self.model.predict_proba(X_embeddings)


# Train and save model
def train_and_save_model(model_choice, data, model_file_path, report_file_path):
    # Preprocess the data
    X = data['text']
    y = data['type']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # Initialize BERT classifier with the chosen model
    if model_choice == 'LogisticRegression'.lower():
        clf = DocumentClassifier(model=LogisticRegression())
    elif model_choice == 'DT'.lower():
        clf = DocumentClassifier(model=DecisionTreeClassifier())
    elif model_choice == 'RF'.lower():
        clf = DocumentClassifier(model=RandomForestClassifier())
    elif model_choice == 'XGB'.lower():
        clf = DocumentClassifier(model=xgb.XGBClassifier())
    else:
        print(f"Unknown model: {model_choice}")
        return None

    # Train the model
    clf.fit(X_train, y_train)

    save_model(clf, model_file_path)

    # Evaluate the model
    y_predict = clf.predict(X_test)
    report = classification_report(y_test, y_predict, output_dict=True)
    print("***************Classification Report***************")
    print(classification_report(y_test, y_predict))
    print("*" * 50)
    save_report(report, report_file_path)
    return clf


# Predict function for new data
def classify_document(model, text):
    try:
        prediction = model.predict([text])
        prob = model.predict_proba([text])
        confidence = np.max(prob)  # The highest probability as confidence score
        return prediction[0], confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0


# Main function
def main():
    while True:
        model_choice = input("Choose model (LogisticRegression, DT, RF, XGB): ").strip().lower()
        model_file_path = os.path.join(cwd, 'models', f'{model_choice}_document_classifier_model.pkl')
        report_file_path = os.path.join(cwd, 'reports', f'{model_choice}_classification_report.json')
        model = load_trained_model(model_file_path)  # Load trained model if it exists
        if model is None:
            json_file_path = os.path.join(cwd, 'data', 'documents.json')  # Path to the JSON file with document data
            data = load_json_data(json_file_path)
            if data is None:
                return
            model = train_and_save_model(model_choice, data, model_file_path, report_file_path)

        # Test classification with a sample document
        test_text = input("Please provide the text information to classify.\n")
        # test_text = "2023 W-2 Wage and Tax Statement\nEmployee's social security number 123-45-XXXX\nEmployer identification number 12-3456789\nEmployee name: John Smith\nWages, tips, other compensation: $75,000.00\nSocial security wages: $75,000.00\nMedicare wages and tips: $75,000.00\nFederal income tax withheld: $15,750.00"
        classification, confidence = classify_document(model, test_text)
        print(f"Text Classified as : {classification}, Confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()