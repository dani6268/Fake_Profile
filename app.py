from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
MODEL_PATH = "fake_profile_model.pkl"
model = joblib.load(MODEL_PATH)

# MUST match the feature order printed by training script
FEATURE_NAMES = [
    "profile_pic",
    "username_ratio",
    "fullname_words",
    "fullname_ratio",
    "name_equals_username",
    "desc_len",
    "has_url",
    "private",
    "posts",
    "followers",
    "following"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        values = [float(data.get(f, 0)) for f in FEATURE_NAMES]
        arr = np.array([values])
        prob = model.predict_proba(arr)[0][1] * 100
        pred = int(model.predict(arr)[0])
        result = "Fake Profile" if pred == 1 else "Real Profile"
        return jsonify({"result": result, "probability": round(prob, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
