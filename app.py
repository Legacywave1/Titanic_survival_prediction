# app.py  ← FINAL VERSION (Windows + Docker + mlrun folder)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys
import mlflow


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MLRUN_PATH = os.path.join(BASE_DIR, "mlrun")

mlflow.set_tracking_uri(f"file:///{MLRUN_PATH}")


from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = FastAPI(title="Titanic Survival Prediction API")

# --- This will now SUCCEED ---
print("Loading model from mlrun folder...")
pipeline = PredictPipeline()          # ← No try/except → you SEE the real error if any
print("Model loaded successfully!")

class PassengerRequest(BaseModel):
    pclass: int
    sex: str
    age: Optional[float] = None
    sibsp: int
    parch: int
    fare: float
    embarked: str

class PredictionResponse(BaseModel):
    survived: int
    message: str

@app.get("/")
def root():
    return {"message": "Titanic API is LIVE"}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PassengerRequest):
    data = CustomData(
        Pclass=payload.pclass,
        Name="Unknown",
        Sex=payload.sex,
        Age=payload.age,
        SibSp=payload.sibsp,
        Parch=payload.parch,
        Fare=payload.fare,
        Embarked=payload.embarked
    )
    result = int(pipeline.predict(data)[0])
    return PredictionResponse(
        survived=result,
        message="Survived" if result == 1 else "Did not survive"
    )
