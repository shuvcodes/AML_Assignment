# 📩 SMS Spam Detection API  
### Applied Machine Learning Assignment  
**Author:Shuvodeep Dutta** 

---

## 🔎 Overview

This repository contains an end-to-end implementation of a **Spam Detection system for SMS messages**, designed following production-style practices.

The project includes:

- A trained Machine Learning classification model  
- Probability-based prediction with adjustable threshold  
- Unit testing implemented using `pytest`  
- A RESTful API built with Flask for serving predictions  
- Integration tests to validate API functionality  

The system classifies incoming SMS text into one of two categories:

- **Spam**
- **Ham (Non-Spam)**  

In addition to the predicted label, the API returns a confidence score (propensity) ranging from `0` to `1`.

---

## 📁 Repository Layout

```
Assignment-03/
│
├── app.py
├── score.py
├── predict.py
├── integration_test.py
├── test.py
│
├── coverage.txt
├── full_test.log
├── unit_test.log
│
└── README.md
```

---

## 🧠 Machine Learning Workflow

### Text Feature Extraction
- TF-IDF Vectorization

### Algorithms Explored
- Logistic Regression  
- Multinomial Naive Bayes  
- Linear Support Vector Machine  

### Evaluation Metric
- F1-Score  

The model achieving the best performance is saved using `joblib` and stored inside the `saved_models/` directory for later use.

---

## ⚙️ Prediction Logic

The scoring functionality is defined in `score.py`:

```python
def score(text: str, model, threshold: float) -> (bool, float):
```

### Output Description

- `prediction` → Boolean value (`True` for Spam, `False` for Ham)  
- `propensity` → Confidence probability between `0` and `1`  

The `threshold` parameter determines the classification cutoff and allows control over prediction sensitivity.

---

## 🌐 API Interface

### Endpoint

```
POST /score
```

### Example Request

```json
{
  "text": "Congratulations! You won 1000 dollars!",
  "threshold": 0.5
}
```

### Example Response

```json
{
  "prediction": true,
  "propensity": 0.92
}
```

---

## 🚀 How to Run the Application

### Install Required Dependencies

```bash
pip install flask requests pytest scikit-learn joblib numpy
```

### Launch the Flask Server

```bash
python app.py
```

The API will be available at:

```
http://127.0.0.1:5000
```

---

## 🧪 Testing Strategy

### Unit Tests

The unit test suite validates:

- Basic functionality (smoke test)  
- Output format consistency  
- Correct prediction data type  
- Valid probability range  
- Threshold boundary behavior  
- Clear spam and non-spam cases  

Execute unit tests using:

```bash
pytest test.py
```

---

### Integration Tests

The integration tests:

- Automatically start the Flask server  
- Send HTTP requests to the API endpoint  
- Verify the JSON response structure  
- Ensure graceful shutdown  

Run using:

```bash
pytest test.py
```

---

## 🛠 Technology Stack

- Python  
- Scikit-learn  
- Flask  
- Pytest  
- Joblib  
- RESTful API architecture  

---

## 📌 Key Learning Outcomes

This project demonstrates:

- Complete ML pipeline implementation  
- Model serialization and deployment  
- Probability-driven classification with adjustable threshold  
- REST API development for ML services  
- Unit and integration testing in ML systems  
- Clean, production-style code organization  

---

 

---

