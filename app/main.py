from fastapi import FastAPI
from pydantic import BaseModel
import logging
import numpy as np
import pandas as pd
from app.db import SessionLocal
from app.models import Prediction

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
    sex: str
    smoker: str
    region: str

# -------------------------------
# Load Model from MLflow
# -------------------------------
import mlflow

mlflow.set_tracking_uri("http://13.222.10.43:5000")

try:
    model = mlflow.sklearn.load_model("models:/insurance-model/Production")
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
    db = SessionLocal()
    try:
        if model is None:
            return {"error": "Model not loaded"}

        input_df = pd.DataFrame([{
            "age": data.age,
            "bmi": data.bmi,
            "children": data.children,
            "sex": data.sex,
            "smoker": data.smoker,
            "region": data.region
        }])

        prediction = model.predict(input_df)

        record = Prediction(
            age=data.age,
            bmi=data.bmi,
            children=data.children,
            sex=data.sex,
            smoker=data.smoker,
            region=data.region,
            prediction=float(prediction[0])
        )

        db.add(record)
        db.commit()

        return {"prediction": float(prediction[0])}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": "Prediction failed"}

    finally:
        db.close()