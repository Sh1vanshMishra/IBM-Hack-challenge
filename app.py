# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('Placement_Data.csv')
data = data.dropna()
data['gender'] = data['gender'].map({'M': 0, 'F': 1})

# EDA: Create visualizations (e.g., histograms, scatter plots, correlation matrix)
sns.pairplot(data, hue='status')
plt.show()
data['total_marks'] = data['ssc_p'] + data['hsc_p']

# Perform one-hot encoding for categorical features
data_encoded = pd.get_dummies(data, columns=['hsc_s', 'degree_p', 'workex', 'specialisation'])

# Encode the target variable 'status'
# Use one-hot encoding for the target variable
data_encoded = pd.get_dummies(data_encoded, columns=['status'])

# Split data into features (X) and target (y)
X = data_encoded.drop(['status_Placed', 'status_Not Placed'], axis=1)
y = data_encoded[['status_Placed', 'status_Not Placed']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model (Random Forest as an example)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve user input from the form
            ssc_p = float(request.form['ssc_p'])
            hsc_p = float(request.form['hsc_p'])
            degree_p = float(request.form['degree_p'])
            work_experience = 1 if request.form['work_experience'] == 'Yes' else 0
            specialization = request.form['specialization']

            # Create a feature vector from user input (similar to what you did for training data)
            feature_vector = pd.DataFrame({
                'ssc_p': [ssc_p],
                'hsc_p': [hsc_p],
                'degree_p': [degree_p],
                'workex_Yes': [work_experience],
                'specialisation': [specialization]
            })

            # Use the trained model to make predictions
            predictions = model.predict(feature_vector)

            # The 'predictions' variable now contains the predicted status, you can use it to update the HTML template
            # Assuming you have two classes: 'Placed' and 'Not Placed'
            predicted_status = 'Placed' if predictions[0][0] == 1 else 'Not Placed'

            return render_template('index.html', prediction=f"Predicted Status: {predicted_status}")

        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
