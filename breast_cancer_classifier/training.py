# training.py

import pickle, os, sys, time
from xgboost import XGBClassifier
from preprocessing import main as preprocessing_main
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import warnings


def train_model(X_train, X_test_scaled, y_train, y_test):
    print('Start train XGBClassifier()')

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
        verbosity=0,
        silent=True,
        use_label_encoder=False
    )
    warnings.filterwarnings(action='ignore', category=UserWarning)

    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(xgb_classifier, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)
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

    # Save the model using pickle
    print('Save best model to breast_cancer_detector.pickle')
    with open('breast_cancer_detector.pickle', 'wb') as model_file:
        pickle.dump(best_xgb_classifier, model_file)

def main():
    print('Start training...')

    if not os.path.exists('./data/preprocessed_data.pickle'): 
        print('Preprocessed data cannot be found. Lets do it!') 
        preprocessing_main()
    

    with open('./data/preprocessed_data.pickle', 'rb') as f:
        X_train, X_test_scaled, y_train, y_test = pickle.load(f)

    print('Read preprocessed data from ./data/preprocessed_data.pickle')

    train_model(X_train, X_test_scaled, y_train, y_test)

    print('End training!')


if __name__ == "__main__":
    main()
