# backend/app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS  # allows frontend access

app = Flask(__name__)
CORS(app)  # allow requests from frontend

# Load your trained model
model = load_model("model/wound_detector.h5")

# Preprocessing function
def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(currentImage)
    arr = preprocess(img)
    preds = model.predict(arr)
    label = "infected" if preds[0][0] > 0.5 else "not infected"
    confidence = float(preds[0][0])
    return jsonify({"prediction": label, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
