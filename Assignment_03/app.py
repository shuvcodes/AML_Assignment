import os
import joblib
from flask import Flask, request, jsonify
from score import score


def create_app():
    app = Flask(__name__)

    # Load model once when app is created
    MODEL_PATH = os.path.join("SAVED_MODELS", "best_spam_classifier_model.pkl")
    model = joblib.load(MODEL_PATH)

    @app.route("/score", methods=["POST"])
    def score_endpoint():

        # Check content type
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 415

        data = request.get_json()

        # Validate text field
        if "text" not in data:
            return jsonify({"error": "Missing text field"}), 400

        if not isinstance(data["text"], str):
            return jsonify({"error": "Text must be string"}), 422

        threshold = data.get("threshold", 0.5)

        if not isinstance(threshold, (int, float)):
            return jsonify({"error": "Threshold must be number"}), 422

        ### Call score function
        prediction, propensity = score(data["text"], model, threshold)

        return jsonify({
            "prediction": prediction,
            "propensity": propensity
        })

    return app


# Create app instance for testing
app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)