# app.py
import pickle
import numpy as np
from flask import Flask, request, render_template

# Load trained model
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")   # web form

@app.route("/predict", methods=["POST"])
def predict():
    # Get input from form
    sl = float(request.form["sepal_length"])
    sw = float(request.form["sepal_width"])
    pl = float(request.form["petal_length"])
    pw = float(request.form["petal_width"])

    features = np.array([[sl, sw, pl, pw]])
    prediction = model.predict(features)[0]

    iris_classes = ["Setosa", "Versicolor", "Virginica"]
    result = iris_classes[prediction]

    return render_template("index.html", prediction_text=f"Predicted Flower: {result}")

if __name__ == "__main__":
    app.run(debug=True)
