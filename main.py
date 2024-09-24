from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('C:/Users/anu99/PycharmProjects/Final_AIML_Project/final.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Print the incoming form data for debugging
        print("Received form data:", request.form)

        # Get form data
        input_data = {
            'State/UT Name': request.form['State/UT Name'],
            'Mine Name': request.form['Mine Name'],
            'Coal Mine Owner Name': request.form['Coal Mine Owner Name'],
            'Coal/Lignite': int(request.form['Coal/Lignite']),
            'Govt Owned/Private': int(request.form['Govt Owned/Private']),
            'TypeofMine ': int(request.form['TypeofMine ']),
            'Latitude ': float(request.form['Latitude ']),
            'Longitude ': float(request.form['Longitude '])
        }

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)

        return render_template('index.html', prediction=prediction[0])
    except Exception as e:
        return render_template('index.html', error=str(e))



app.run(port=5001)
