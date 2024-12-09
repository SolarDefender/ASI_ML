import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from autogluon.tabular import TabularPredictor

# Input schema for the weather parameters
class WeatherInput(BaseModel):
    datetime: str
    temperature: float
    humidity: float
    wind_speed: float
    general_diffuse_flows: float
    diffuse_flows: float
    target_zone: Optional[int] = None  # Zone to predict; 1, 2, or 3


def api_run(best_model):
    if best_model is None or not isinstance(best_model, TabularPredictor):
        raise ValueError("Invalid or missing best model for the API.")

    app = FastAPI()

    @app.get("/", tags=["intro"])
    async def index():
        return {"message": "Welcome to the Weather Prediction API"}

    @app.post("/predict", tags=["prediction"], status_code=200)
    async def get_predictions(input_data: WeatherInput):
        # Prepare data for prediction
        X_new = pd.DataFrame([{
            "Datetime": input_data.datetime,
            "Temperature": input_data.temperature,
            "Humidity": input_data.humidity,
            "WindSpeed": input_data.wind_speed,
            "GeneralDiffuseFlows": input_data.general_diffuse_flows,
            "DiffuseFlows": input_data.diffuse_flows
        }])

        # Preprocess Datetime if necessary
        X_new["Datetime"] = pd.to_datetime(X_new["Datetime"])
        X_new["Hour"] = X_new["Datetime"].dt.hour
        X_new["DayOfWeek"] = X_new["Datetime"].dt.dayofweek
        X_new.drop(columns=["Datetime"], inplace=True)

        # Predict for the specified zone or all zones
        if input_data.target_zone:
            prediction = best_model.predict(X_new)
            return {"zone": input_data.target_zone, "prediction": prediction.tolist()}
        else:
            predictions = {
                f"Zone_{zone}": best_model.predict(X_new).tolist()  # Reuse the model for each zone
                for zone in [1, 2, 3]
            }
            return {"predictions": predictions}

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
