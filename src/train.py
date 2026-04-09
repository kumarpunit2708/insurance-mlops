import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("data/Insurance_Premium_Regression.csv")

X = df.drop(columns=["charges"])
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]


preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])


model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,root_mean_squared_error

mlflow.set_experiment("insurance-premium")

with mlflow.start_run():

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Prediction
    y_pred = model_pipeline.predict(X_test)

    # Metric
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)

    # Log metrics
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)

    # Log model
    mlflow.sklearn.log_model(
    sk_model=model_pipeline,
    artifact_path="model",
    registered_model_name="insurance-model"
)
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)

os.makedirs("artifacts", exist_ok=True)

with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)
