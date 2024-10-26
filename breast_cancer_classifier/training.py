# training.py

import pickle
from xgboost import XGBClassifier
from preprocessing import load_data, preprocess_data

def train_model(X_train_scaled, y_train):
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(X_train_scaled, y_train)

    # Save the model using pickle
    with open('breast_cancer_detector.pickle', 'wb') as model_file:
        pickle.dump(xgb_classifier, model_file)

if __name__ == "__main__":
    cancer_df = load_data()
    X_train_scaled, _, y_train, _ = preprocess_data(cancer_df)
    train_model(X_train_scaled, y_train)