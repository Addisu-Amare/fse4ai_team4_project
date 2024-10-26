# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    # Load breast cancer dataset from CSV
    cancer_df = pd.read_csv('data/breast_cancer_data.csv')
    return cancer_df

def preprocess_data(cancer_df):
    # Split DataFrame into features and target variable
    X = cancer_df.drop(['target'], axis=1)
    y = cancer_df['target']
    
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Ensure the data directory exists
    import os
    os.makedirs('data', exist_ok=True)
    
    # Load and preprocess data
    cancer_df = load_data()
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(cancer_df)