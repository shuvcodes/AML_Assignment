import joblib 
import os 
import numpy as np

### load the models and vectorizer
MODEL_PATH=os.path.join("SAVED_MODELS","best_spam_classifier_model.pkl")
VECTORIZER_PATH=os.path.join("SAVED_MODELS","tfidf_vectorizer.pkl")

model=joblib.load(MODEL_PATH)
vectorizer=joblib.load(VECTORIZER_PATH)


def score(text:str,model,threshold:float):
    """
    Scores a trained modek on input text
    
    parameters:  
        1)text(str): Input sms text
        2)model: trained model
        3)threshold(float): Classification threshold 
    Returns:
        1)prediction (in boolean): True(Spam) or False(Ham)
        2)probability b/w 0 and 1
    """
    #### transforms the input text into tf-idf features 
    features=vectorizer.transform([text])   

    ### get the prob. score
    if hasattr(model, "predict_proba"):
        prob=model.predict_proba(features)[0][1]
    else:
        prob=model.decision_function(features)[0]
        prob=1/(1+np.exp(-prob))  ### convert to probability using sigmoid function
    ### Apply threshold to get the final prediction
    prediction=prob>=threshold

    return bool(prediction), float(prob)

