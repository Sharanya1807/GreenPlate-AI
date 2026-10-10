
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GreenPlate AI Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("food_waste_model.pkl")


class PredictionInput(BaseModel):
    meals_served: float = Field(ge=0)
    kitchen_staff: float = Field(ge=0)
    temperature_C: float
    humidity_percent: float = Field(ge=0, le=100)
    day_of_week: float
    special_event: int = Field(ge=0, le=1)
    past_waste_kg: float = Field(ge=0)
    staff_experience: str
    waste_category: str


@app.get("/")
def home():
    return {"message": "GreenPlate AI API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        row = pd.DataFrame([{
            "meals_served": data.meals_served,
            "kitchen_staff": data.kitchen_staff,
            "temperature_C": data.temperature_C,
            "humidity_percent": data.humidity_percent,
            "day_of_week": data.day_of_week,
            "special_event": data.special_event,
            "past_waste_kg": data.past_waste_kg,
            "staff_experience": data.staff_experience,
            "waste_category": data.waste_category,
        }])

        prediction = float(model.predict(row)[0])

        return {
            "predicted_food_waste_kg": round(prediction, 2),
            "model": "RandomForestRegressor"
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )