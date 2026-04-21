import pytest
import joblib
from score import score
from app import app


#### Test Cases

SCORE_CASES = [

    #### Spam
{"id": "spam_9", "text": "Act fast! Your account is eligible for a $1000 bonus reward.", "expected": True},
{"id": "spam_10", "text": "Get rich quick with this amazing investment opportunity.", "expected": True},
{"id": "spam_11", "text": "You have been pre-approved for a zero interest loan. Apply today!", "expected": True},
{"id": "spam_12", "text": "Congratulations! Claim your free vacation package now.", "expected": True},
{"id": "spam_13", "text": "Limited offer! Buy one get one free. Shop immediately.", "expected": True},
{"id": "spam_14", "text": "Winner! You have won a brand new smartphone. Click to confirm.", "expected": True},
{"id": "spam_15", "text": "Earn money from home easily. Sign up now.", "expected": True},
{"id": "spam_16", "text": "Final reminder: Your prize will expire today. Respond urgently.", "expected": True},

    #### Ham
   {"id": "ham_9", "text": "Let's schedule a meeting for next week.", "expected": False},
{"id": "ham_10", "text": "I have sent you the notes via email.", "expected": False},
{"id": "ham_11", "text": "Can we reschedule our appointment?", "expected": False},
{"id": "ham_12", "text": "Dinner is ready. Please come home soon.", "expected": False},
{"id": "ham_13", "text": "Don't forget to bring the documents tomorrow.", "expected": False},
{"id": "ham_14", "text": "The train will arrive at 6 pm.", "expected": False},
{"id": "ham_15", "text": "Thank you for your help yesterday.", "expected": False},
{"id": "ham_16", "text": "Please review the attached file and let me know.", "expected": False},
]


# Unit Test Setup (score.py)

@pytest.fixture(scope="module")
def model():
    """
    Load trained model once for all unit tests.
    """
    return joblib.load("saved_models/best_spam_classifier_model.pkl")


#### Unit Tests for score()

@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_score_prediction(model, case):

    pred, prop = score(case["text"], model, 0.5)

    assert isinstance(pred, bool), f"{case['id']} prediction is not bool"
    assert isinstance(prop, float), f"{case['id']} propensity is not float"
    assert 0 <= prop <= 1, f"{case['id']} propensity out of range"
    assert pred == case["expected"], \
        f"{case['id']} expected {case['expected']}, got {pred}"


@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_threshold_edges(model, case):
    """
    threshold=0 should predict True
    threshold=1 should predict False
    """
    pred_zero, _ = score(case["text"], model, 0.0)
    assert pred_zero is True

    pred_one, _ = score(case["text"], model, 1.0)
    assert pred_one is False


####Integration Test Setup
@pytest.fixture
def client():
    """
    Flask test client (NO subprocess).
    This fixes coverage issue.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


#### Integration Tests (Flask endpoint)

@pytest.mark.integration
@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_flask_score(client, model, case):

    payload = {
        "text": case["text"],
        "threshold": 0.5
    }

    response = client.post("/score", json=payload)

    assert response.status_code == 200, \
        f"{case['id']} returned status {response.status_code}"

    result = response.get_json()

    assert "prediction" in result
    assert "propensity" in result

    expected_pred, expected_prop = score(case["text"], model, 0.5)

    assert result["prediction"] == expected_pred
    assert abs(result["propensity"] - expected_prop) < 1e-6


@pytest.mark.integration
def test_flask_invalid_input(client):
    """
    Test missing text field.
    """
    response = client.post("/score", json={"threshold": 0.5})
    assert response.status_code in [400, 422]