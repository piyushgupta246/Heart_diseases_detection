from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("heart_disease.csv")
x = df[['age', 'cp', 'thalach']]
y = df['target']

# Split data and train the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve inputs
            age = int(request.form['age'])
            cp = int(request.form['cp'])
            thalach = int(request.form['thalach'])
            
            # Validate inputs
            if not (1 <= age <= 120):
                return render_template('result.html', prediction="Invalid input: Age must be between 1 and 120.")
            if not (0 <= cp <= 3):
                return render_template('result.html', prediction="Invalid input: Chest pain type (cp) must be between 0 and 3.")
            if not (30 <= thalach <= 220):
                return render_template('result.html', prediction="Invalid input: Heart rate (thalach) must be between 30 and 220.")

            # Create a DataFrame for prediction
            user_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])
            prediction = model.predict(user_data)[0]
            
            # Return result
            result = "Yes, the person is suffering from heart disease." if prediction == 1 else "No, the person is not suffering from heart disease."
            return render_template('result.html', prediction=result)
        except ValueError:
            return render_template('result.html', prediction="Invalid input: Please enter valid numerical values.")
    else:
        return render_template('result.html', prediction="Invalid request method.")

if __name__ == '__main__':
    app.run(debug=True)
