import joblib 
import os 
import numpy as np 
from score import score

### Laod the model
MODEL_PATH=os.path.join("SAVED_MODELS","best_spam_classifier_model.pkl")

model=joblib.load(MODEL_PATH)


def test_score():
    pred,prop=score("Hello Arnab",model,0.5)

    assert pred is not None, "Prediction should not be None"
    assert prop is not None, "Probability should not be None"


def test_score_format():
    pred,prop=score("This is a spam message",model,0.5 )
    assert isinstance(pred, bool)
    assert isinstance(prop, float)

def test_prediction_value():
    ### Prediction should be true or false
    pred,_=score("Congratulations! You've won a free ticket. Click here to claim.",model,0.5)
    assert pred in [True, False]

def test_propensity_range():
    ### must be b/w 0 and 1
    _,prop=score("This is a normal message",model,0.5)
    assert 0.0 <= prop <= 1.0

def test_threshold_effect():
    ### if threshold is 0, all should be predicted as spam
    pred,_=score("This is a normal message",model,1.0)
    assert pred is False 

def test_obvious_spam():
    """Typical spam message"""
    spam_text = "Congratulations!!! You won $1000. Click here now!"
    pred, _ = score(spam_text, model, 0.5)
    assert pred is True


def test_obvious_ham():
    """Typical non-spam message"""
    ham_text = "Hi Arnab, are we meeting tomorrow for project discussion?"
    pred, _ = score(ham_text, model, 0.5)
    assert pred is False
test_score()
test_score_format()
test_prediction_value()
test_propensity_range()
test_threshold_effect()
test_obvious_spam()
test_obvious_ham()

print("All tests passed successfully!")