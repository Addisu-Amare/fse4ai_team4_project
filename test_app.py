import unittest
from flask import Flask
from app import app  

class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        # Set up the Flask test client
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        # Test the home page response
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Breast Cancer Prediction Using XGBoost Classifier', response.data)
        self.assertIn(b'Enter Fields Below', response.data)  

    def test_predict(self):
        # Test the prediction endpoint with valid input
        valid_input = {
            'mean_radius': 14.0,
            'mean_texture': 20.0,
            'mean_perimeter': 90.0,
            'mean_area': 600.0,
            'mean_smoothness': 0.1,
            'mean_compactness': 0.2,
            'mean_concavity': 0.3,
            'mean_concave_points': 0.4,
            'mean_symmetry': 0.5,
            'mean_fractal_dimension': 0.06,
            'radius_error': 1.0,
            'texture_error': 1.5,
            'perimeter_error': 2.0,
            'area_error': 30.0,
            'smoothness_error': 0.002,
            'compactness_error': 0.03,
            'concavity_error': 0.04,
            'concave_points_error': 0.05,
            'symmetry_error': 0.01,
            'fractal_dimension_error': 0.0025,
            'worst_radius': 15.5,
            'worst_texture': 25.5,
            'worst_perimeter': 100.5,
            'worst_area': 700.5,
            'worst_smoothness': 0.12,
            'worst_compactness': 0.25,
            'worst_concavity': 0.35,
            'worst_concave_points': 0.45,
            'worst_symmetry': 0.55,
            'worst_fractal_dimension': 0.07
        }

        response = self.app.post('/predict', data=valid_input)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Patient has', response.data)

    def test_predict_invalid_input(self):
        # Test the prediction endpoint with invalid input (non-float values)
        invalid_input = {
            'mean_radius': "invalid",
            # Add other fields as needed
        }
        response = self.app.post('/predict', data=invalid_input)
        self.assertEqual(response.status_code, 500) 

if __name__ == '__main__':
    unittest.main()