from fastapi import FastAPI
from pydantic import BaseModel
import logging
import numpy as np
import mlflow.sklearn

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# App
# -------------------------------
app = FastAPI()

# -------------------------------
# Input Schema
# -------------------------------
class PredictionInput(BaseModel):
    age: int
    bmi: float
    children: int

# -------------------------------
# Load Model from MLflow
# -------------------------------
MODEL_URI = "models:/insurance-model/1"   # version 1

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    logger.info("Model loaded from MLflow")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None


# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def home():
    return {"message": "Insurance API running 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        if model is None:
            return {"error": "Model not loaded"}

        features = np.array([[data.age, data.bmi, data.children]])
        prediction = model.predict(features)

        return {"prediction": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}