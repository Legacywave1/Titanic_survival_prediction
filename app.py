from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = FastAPI(title="Titanic Survival Prediction API")


try:
    pipeline = PredictPipeline()
except Exception as e:
    pipeline = None
    print(f"[WARN] No model in Production/Staging: {e}")
    print("    → Run train_pipeline.py and promote a model first.")

class PassengerRequest(BaseModel):
    pclass: int
    sex: str
    age: float | None = None
    sibsp: int
    parch: int
    fare: float
    embarked: str

class PredictionResponse(BaseModel):
    survived: int
    message: str

@app.get("/")
def root():
    return {"message": "Titanic API is up. Use /predict to infer."}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PassengerRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not available. Train and promote first.")
    try:
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
        result = pipeline.predict(data)[0]
        return PredictionResponse(
            survived=int(result),
            message="Survived" if result == 1 else "Did not survive"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
