from fastapi import FastAPI
import logging
import pickle
import numpy as np
import os

# -------------------------------
# Logging Setup
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Initialize App
# -------------------------------
app = FastAPI()

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = os.path.join("artifacts", "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None


# -------------------------------
# Home Endpoint
# -------------------------------
@app.get("/")
def home():
    return {"message": "Insurance Premium Prediction API is running 🚀"}


# -------------------------------
# Health Check Endpoint
# -------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(data: dict):
    """
    Expected input example:
    {
        "features": [25, 28.5, 2]
    }
    """

    try:
        logger.info(f"Received input: {data}")

        if model is None:
            return {"error": "Model not loaded"}

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        logger.info(f"Prediction: {prediction[0]}")

        return {
            "prediction": float(prediction[0])
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}