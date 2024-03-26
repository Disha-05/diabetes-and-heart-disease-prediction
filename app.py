from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# @app.route("/")
# def home():
#     return render_template("home.html")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting input values from the form
        glucose = float(request.form['glucose'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = float(request.form['age'])

        # Making predictions using the loaded model
        prediction = model.predict([[glucose, insulin, bmi, age]])

        # Displaying the prediction result
        if prediction[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Not Diabetic'

        return render_template('home.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)