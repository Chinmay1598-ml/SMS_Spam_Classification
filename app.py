"""
app.py
FastAPI backend for SMS Spam Classification

Run with:
    uvicorn app:app --reload
"""

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import os

# -----------------------
# Define Request Schema
# -----------------------
class MessageRequest(BaseModel):
    message: str

# -----------------------
# Initialize FastAPI
# -----------------------
app = FastAPI(
    title="📩 SMS Spam Classifier",
    description="Classify SMS messages as Spam or Ham using ML pipelines",
    version="1.0"
)

# -----------------------
# Load Models
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

imbalanced_model = joblib.load(os.path.join(BASE_DIR, "models", "spam_classifier_imbalanced.pkl"))
smote_model = joblib.load(os.path.join(BASE_DIR, "models", "spam_classifier_smote.pkl"))
label_encoder_imbalanced = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder_imbalanced.pkl"))
label_encoder_smote = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder_smote.pkl"))

# -----------------------
# Utility function
# -----------------------
def make_prediction(model, label_encoder, text: str):
    """Return prediction + confidence from pipeline."""
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
    elif hasattr(model, "decision_function"):
        # Fallback for SVM without predict_proba
        decision = model.decision_function([text])
        proba = [1 - decision[0], decision[0]]  # crude approx

    pred = model.predict([text])[0]
    label = label_encoder.inverse_transform([pred])[0]

    confidence = float(max(proba)) if proba is not None else None

    return {"prediction": label, "confidence": confidence}

# -----------------------
# Endpoints
# -----------------------
@app.post("/predict-imbalanced")
def predict_imbalanced(req: MessageRequest):
    return make_prediction(imbalanced_model, label_encoder_imbalanced, req.message)

@app.post("/predict-smote")
def predict_smote(req: MessageRequest):
    return make_prediction(smote_model, label_encoder_smote, req.message)

# -----------------------
# Health Check
# -----------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "SMS Spam Classifier API is running 🚀"}
