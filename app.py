from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("video_summary_model.h5")


@app.route("/")
def home():
    return "Video summarization model is running."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting 512 features
        features = np.array(data["features"])

        if len(features) != 512:
            return jsonify({
                "error": "Input must contain exactly 512 features"
            }), 400

        features = features.reshape(1, 512)

        prediction = model.predict(features)

        score = float(prediction[0][0])

        return jsonify({
            "importance_score": score
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))  # Render provides PORT env variable
    app.run(host="0.0.0.0", port=port)