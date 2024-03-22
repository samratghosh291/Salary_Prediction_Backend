from flask import request, jsonify, render_template
from app import app
import pickle

# Load the trained model
with open('app/model.pkl', 'rb') as file:
    clf = pickle.load(file)

# Default route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request form
        experience = float(request.form['experience'])

        # Ensure the input is a 2D array for prediction
        features = [[experience]]

        # Make prediction
        prediction = clf.predict(features)[0]

        # Return the prediction to the UI
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        # Log the error
        app.logger.error(f"An error occurred during prediction: {e}")
        # Return error message to the UI
        return render_template('error.html', error_message="An error occurred during prediction. Please try again later."), 500

# Define a route for handling errors
@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', error_message="Page not found."), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('error.html', error_message="Internal server error."), 500
