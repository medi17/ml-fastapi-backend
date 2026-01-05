"""
Main FastAPI application for ML model predictions.
This module provides endpoints for Logistic Regression and Decision Tree models.
"""

from fastapi import FastAPI
import joblib

app = FastAPI(title="ML FastAPI Backend")

logistic_model = joblib.load("models/logistic_model.joblib")
tree_model = joblib.load("models/decision_tree_model.joblib")

@app.get("/")
def root():
    return {"message": "FastAPI ML Backend Running"}

@app.post("/predict/logistic")
def predict_logistic(features: list):
    prediction = logistic_model.predict([features])
    return {"model": "Logistic Regression", "prediction": int(prediction[0])}

@app.post("/predict/tree")
def predict_tree(features: list):
    prediction = tree_model.predict([features])
    return {"model": "Decision Tree", "prediction": int(prediction[0])}
