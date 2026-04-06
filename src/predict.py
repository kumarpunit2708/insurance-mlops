import pickle
import pandas as pd


class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"

    def predict(self, data: dict):
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        df = pd.DataFrame([data])
        prediction = model.predict(df)

        return prediction[0]