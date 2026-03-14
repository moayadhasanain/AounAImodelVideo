from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model with compatibility settings
print("Loading model...")
try:
    # Try loading with default settings first
    model = load_model("video_summary_model.h5")
except TypeError as e:
    print(f"Encountered TypeError: {e}")
    print("Attempting to load with custom objects...")
    
    # If that fails, load with custom object scope to handle legacy layers
    model = load_model("video_summary_model.h5", compile=False)
    
    # Recompile the model if needed
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
print("Model loaded successfully!")

@app.route("/")
def home():
    return "Video summarization model is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"])
        
        if len(features) != 512:
            return jsonify({"error": "Input must contain exactly 512 features"}), 400
            
        features = features.reshape(1, 512)
        prediction = model.predict(features)
        score = float(prediction[0][0])
        
        return jsonify({"importance_score": score})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host="0.0.0.0", port=port)