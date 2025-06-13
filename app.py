from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('house_price_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features_array = np.array([features])
        prediction = model.predict(features_array)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Predicted House Price: ${output}')
    except Exception as e:
        return render_template('index.html', prediction_text='Error in input data. Please check your inputs.')
    

if __name__ == "__main__":
    app.run(debug=True)