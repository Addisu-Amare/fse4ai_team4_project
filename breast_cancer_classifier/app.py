import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Handle the case where the model fails to load

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) 
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Model not loaded. Please try again later.')

    try:
        # Extract input features from the form
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]
        
        # Define feature names corresponding to the model
        features_name = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
        
        # Create DataFrame for prediction
        df = pd.DataFrame(features_value, columns=features_name)
        
        # Make prediction
        output = model.predict(df)
        
        # Interpret output
        if output[0] == 0:
            res_val = "** breast cancer **"
        else:
            res_val = "no breast cancer"

        return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text='Error occurred during prediction.')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)