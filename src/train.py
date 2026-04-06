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


model_pipeline.fit(X_train, y_train)


y_pred = model_pipeline.predict(X_test)

score = r2_score(y_test, y_pred)
print("R2 Score:", score)


os.makedirs("artifacts", exist_ok=True)

with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)
