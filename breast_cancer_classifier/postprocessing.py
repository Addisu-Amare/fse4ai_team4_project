import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from preprocessing import load_data, preprocess_data

# Create an output directory to store results
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Load data and preprocess it
cancer_df = load_data()
X_train, X_test_scaled, y_train, y_test = preprocess_data(cancer_df)

# Define parameters for Grid Search
params = {
    'colsample_bytree': [0.3, 0.4],
    'max_depth': [3, 15],
    'learning_rate': [0.1, 0.3],
    'n_estimators': [100]
}

# Initialize the XGBoost classifier
xgb_classifier = XGBClassifier(
    base_score=0.5,
    booster='gbtree',
    gamma=0.2,
    min_child_weight=1,
    objective='binary:logistic',
    random_state=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    subsample=1,
    verbosity=1
)

# Perform Grid Search to find the best parameters
grid_search = GridSearchCV(xgb_classifier, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

# Get the best estimator from grid search
best_xgb_classifier = grid_search.best_estimator_

# Fit the model with the best parameters
best_xgb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_xgb_classifier.predict(X_test_scaled)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the best model: {accuracy:.2f}')

# Save accuracy to a text file
with open(os.path.join(output_dir, 'accuracy.txt'), 'w') as f:
    f.write(f'Accuracy of the best model: {accuracy:.2f}\n')

# Print classification report and save it to a text file
report = classification_report(y_test, y_pred)
print(report)
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion Matrix Visualization and save it as an image
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Heatmap of Confusion Matrix', fontsize=15)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))  # Save confusion matrix plot
plt.close()  # Close the plot to avoid display in notebooks

# Save the trained model to a file for later use
with open('breast_cancer_detector.pickle', 'wb') as model_file:
    pickle.dump(best_xgb_classifier, model_file)

def evaluate_model(X_test_scaled, y_test):
    # Load the trained model
    with open('breast_cancer_detector.pickle', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Make predictions and evaluate the model
    y_pred = loaded_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Accuracy of the loaded model: {accuracy:.2f}')
    
    # Save evaluation results to output directory
    with open(os.path.join(output_dir, 'loaded_model_accuracy.txt'), 'w') as f:
        f.write(f'Accuracy of the loaded model: {accuracy:.2f}\n')
    
    return accuracy  # Return accuracy for testing

if __name__ == "__main__":
    # Evaluate the model using preprocessed data and save results
    evaluate_model(X_test_scaled, y_test)