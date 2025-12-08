from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parents[1]

model = joblib.load(BASE / "models" / "model.pkl")
encoders = joblib.load(BASE / "models" / "encoders.pkl")

class FlightInput(BaseModel):
    AIRLINE: str
    ORIGIN: str
    ORIGIN_CITY: str
    DEST: str
    DEST_CITY: str
    DEP_DELAY: float
    CANCELLED: int
    DIVERTED: int
    DISTANCE: int

@app.get("/")
def home():
    return {"message": "Flight Delay Predictor API running 🚀"}

@app.post("/predict")
def predict(input: FlightInput):
    try:
        df = pd.DataFrame([input.dict()])

        for col in encoders:
            val = df[col].values[0]
            if val not in encoders[col].classes_:
                df[col] = 0
            else:
                df[col] = encoders[col].transform(df[col].values)

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {
            "is_delayed": int(pred),
            "probability": float(prob)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/metadata")
def metadata():
    raw = pd.read_csv(BASE / "data" / "processed" / "train_clean_city.csv")

    
    route_map = {
        f"{o},{d}": dist
        for (o, d), dist in raw.groupby(["ORIGIN", "DEST"])["DISTANCE"].first().items()
    }

    return {
        "airlines": list(raw["AIRLINE"].unique()),
        "origins": list(raw["ORIGIN"].unique()),
        "destinations": list(raw["DEST"].unique()),
        "origin_cities": list(raw["ORIGIN_CITY"].unique()),
        "dest_cities": list(raw["DEST_CITY"].unique()),
        "route_distance": route_map,  
    }

