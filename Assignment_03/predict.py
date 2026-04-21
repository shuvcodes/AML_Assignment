# predict.py

import os
import joblib
import numpy as np

# Load model and vectorizer

MODEL_PATH = os.path.join("saved_models", "best_spam_classifier_model.pkl")
VECTORIZER_PATH = os.path.join("saved_models", "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_message(text, threshold=0.5):
    features = vectorizer.transform([text])

    if hasattr(model, "predict_proba"):
        propensity = model.predict_proba(features)[0][1]
    else:
        decision_score = model.decision_function(features)[0]
        propensity = 1 / (1 + np.exp(-decision_score))

    prediction = propensity >= threshold

    return prediction, propensity


# 🔹 User Input
user_input = input("Enter your message: ")

prediction, score = predict_message(user_input)

if prediction:
    print(f"\nPrediction: SPAM")
else:
    print(f"\nPrediction: HAM")

print(f"Spam Probability: {score:.4f}")