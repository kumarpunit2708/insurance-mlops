from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import PredictPipeline

app = FastAPI()


class InsuranceInput(BaseModel):
    age: int
    bmi: float
    children: int
    sex: str
    smoker: str
    region: str


@app.get("/")
def home():
    return {"message": "Insurance API Running 🚀"}


@app.post("/predict")
def predict(data: InsuranceInput):
    input_data = data.dict()

    pipeline = PredictPipeline()
    result = pipeline.predict(input_data)

    return {"prediction": result}