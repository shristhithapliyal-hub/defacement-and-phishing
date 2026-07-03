
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load models
defacement_model = joblib.load("defacement_model.pkl")
phishing_model = joblib.load("phishing_model.pkl")
tfidf_vectorizer = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder_defacement.pkl")
selector = joblib.load("selector_defacement.pkl")

@app.route("/")
def home():
    return "Flask app running!"

@app.route("/predict_defacement", methods=["POST"])
def predict_defacement():
    data = request.json
    text = data.get("content", "")
    X = tfidf_vectorizer.transform([text])
    X_selected = selector.transform(X)
    prediction = defacement_model.predict(X_selected)
    label = label_encoder.inverse_transform(prediction)[0]
    return jsonify({"prediction": label})

@app.route("/predict_phishing", methods=["POST"])
def predict_phishing():
    data = request.json
    url = data.get("url", "")
    X = tfidf_vectorizer.transform([url])
    prediction = phishing_model.predict(X)
    return jsonify({"prediction": prediction[0]})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

