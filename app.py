'''
# train_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Save model
with open("knn_model.pkl", "wb") as f:
    pickle.dump(model, f)

'''

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)   # ✅ make sure the variable name is "app"

# Example route
@app.route('/')
def home():
    return "ML API is working!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Load model (adjust path)
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict([data['features']])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
