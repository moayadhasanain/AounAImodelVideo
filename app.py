from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import os
import requests
import cv2
import tempfile

app = Flask(__name__)

# -------------------------------
# Load your trained model
# -------------------------------
def load_model_compatibly(model_path):
    try:
        print("Attempting standard load...")
        return tf.keras.models.load_model(model_path)
    except Exception as e1:
        print(f"Standard load failed: {e1}")
        try:
            print("Attempting load with compile=False...")
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e2:
            print(f"Load with compile=False failed: {e2}")
            try:
                print("Rebuilding model architecture from weights...")
                inputs = Input(shape=(512,), name='input_layer')
                x = Dense(256, activation='relu', name='dense_256')(inputs)
                x = Dense(128, activation='relu', name='dense_128')(x)
                x = Dense(64, activation='relu', name='dense_64')(x)
                outputs = Dense(1, activation='sigmoid', name='output_layer')(x)
                new_model = Model(inputs=inputs, outputs=outputs)
                new_model.load_weights(model_path)
                new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                print("Model successfully rebuilt from weights!")
                return new_model
            except Exception as e3:
                print(f"All loading attempts failed: {e3}")
                raise

print("Loading model...")
model = load_model_compatibly("video_summary_model.h5")
print("Model loaded successfully!")

# -------------------------------
# Helper function: extract 512-dim features from video
# -------------------------------
def extract_video_features(video_path):
    """
    Example placeholder: extract features from video frames.
    Replace with your real feature extraction logic.
    """
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Simple placeholder: resize and flatten frame
        frame = cv2.resize(frame, (16, 16))  # small for demo
        frame = frame.mean(axis=2)           # convert to grayscale
        features.append(frame.flatten())
        frame_count += 1
        if frame_count >= 32:  # limit frames for speed
            break
    cap.release()
    
    # Flatten all frames and pad/truncate to 512
    feature_vector = np.concatenate(features)
    if len(feature_vector) < 512:
        feature_vector = np.pad(feature_vector, (0, 512 - len(feature_vector)))
    else:
        feature_vector = feature_vector[:512]
    
    # Normalize
    feature_vector = feature_vector.astype('float32') / np.max(feature_vector)
    return feature_vector

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return "Video summarization model is running on Render!"

@app.route("/predict_url", methods=["POST"])
def predict_url():
    try:
        data = request.get_json()
        if not data or 'video_url' not in data:
            return jsonify({"error": "No video URL provided"}), 400
        
        video_url = data["video_url"]
        
        # Download video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            r = requests.get(video_url, stream=True)
            if r.status_code != 200:
                return jsonify({"error": "Failed to download video"}), 400
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            tmp_video_path = tmp_file.name
        
        # Extract features
        features = extract_video_features(tmp_video_path)
        
        # Predict
        features = features.reshape(1, 512)
        prediction = model.predict(features, verbose=0)
        score = float(prediction[0][0])
        
        # Delete temporary video
        os.remove(tmp_video_path)
        
        return jsonify({
            "importance_score": score,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host="0.0.0.0", port=port)