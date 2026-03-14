from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import os

app = Flask(__name__)

def load_model_compatibly(model_path):
    """Load model with compatibility for older Keras versions"""
    try:
        # Try 1: Standard load
        print("Attempting standard load...")
        return tf.keras.models.load_model(model_path)
    except Exception as e1:
        print(f"Standard load failed: {e1}")
        
        try:
            # Try 2: Load with compile=False
            print("Attempting load with compile=False...")
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e2:
            print(f"Load with compile=False failed: {e2}")
            
            try:
                # Try 3: Load weights into a newly built model
                print("Attempting to rebuild model architecture from weights...")
                
                # Recreate your model architecture (based on your train_model.py)
                inputs = Input(shape=(512,), name='input_layer')
                x = Dense(256, activation='relu', name='dense_256')(inputs)
                x = Dense(128, activation='relu', name='dense_128')(x)
                x = Dense(64, activation='relu', name='dense_64')(x)
                outputs = Dense(1, activation='sigmoid', name='output_layer')(x)
                
                new_model = Model(inputs=inputs, outputs=outputs)
                
                # Load weights
                new_model.load_weights(model_path)
                
                # Compile the model
                new_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                print("Model successfully rebuilt from weights!")
                return new_model
                
            except Exception as e3:
                print(f"All loading attempts failed: {e3}")
                raise

print("Loading model...")
model = load_model_compatibly("video_summary_model.h5")
print("Model loaded successfully!")

@app.route("/")
def home():
    return "Video summarization model is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "No features provided"}), 400
            
        features = np.array(data["features"])
        
        if len(features) != 512:
            return jsonify({"error": f"Input must contain exactly 512 features, got {len(features)}"}), 400
            
        features = features.reshape(1, 512)
        prediction = model.predict(features, verbose=0)
        score = float(prediction[0][0])
        
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