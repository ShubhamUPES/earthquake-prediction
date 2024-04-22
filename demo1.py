from flask import Flask, render_template, request

# Load the trained model and other necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset and preprocess it
data = pd.read_csv('dataset.csv')
magnitude_threshold = 5.0
data['earthquake_label'] = data['Magnitude'].apply(lambda x: 1 if x >= magnitude_threshold else 0)
X = data[['latitude', 'longitude']]
y = data['earthquake_label']

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create Flask app-
app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get latitude and longitude inputs from the form
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    
    # Make prediction using the trained model
    prediction = model.predict([[latitude, longitude]])
    
    # Display prediction result
    if prediction[0] == 1:
        result = "Attention: This area is at risk of earthquakes."
    else:
        result = "This area is earthquake-free."

    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True,port=5003,use_reloader=False)
