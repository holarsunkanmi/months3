# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib

# app = FastAPI()

# # Load your trained model
# model = joblib.load("models/model1.pkl")


# class InputData(BaseModel):
#     features: list[float]   # Accepts list of numbers


# @app.get("/")
# def read_root():
#     return {"message": "API is working"}


# @app.post("/predict")
# def predict(data: InputData):
#     try:
#         features = [data.features]
#         prediction = model.predict(features)
#         return {"prediction": prediction.tolist()}
#     except Exception as e:
#         return {"error": str(e)}




from fastapi import FastAPI
from pydantic import BaseModel
from models_db import Prediction
from sqlalchemy.orm import sessionmaker
from database import engine
import pandas as pd
import joblib
import numpy as np
import os


app = FastAPI()

os.makedirs("models", exist_ok=True)

model = joblib.load("models/model1.pkl")

columns = joblib.load("models/features.pkl")


class InputData(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

SessionLocal = sessionmaker(bind=engine)


@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.features], columns=columns)
        prediction = model.predict(df)

        db = SessionLocal()

        new_pred = Prediction(
            prediction=str(prediction[0])
        )

        db.add(new_pred)
        db.commit()
        db.close()

        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}