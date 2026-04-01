from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['gender']),
            float(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            int(request.form['ever_married']),
            int(request.form['work_type']),
            int(request.form['residence_type']),
            float(request.form['avg_glucose']),
            float(request.form['bmi']),
            int(request.form['smoking_status'])
        ]
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1] * 100

        if prediction == 1:
            result = f"⚠️ High Stroke Risk — {probability:.1f}% probability"
            color = "red"
        else:
            result = f"✅ Low Stroke Risk — {probability:.1f}% probability"
            color = "green"

        return render_template('index.html', result=result, color=color)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}", color="orange")

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)