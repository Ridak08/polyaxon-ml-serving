from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

DDOS_CLASS_MAPPING = {0: "Tráfico benigno", 1: "Ataque DNS", 2: "Ataque Syn", 3: "Ataque UDP", 4: "Ataque NetBIOS", 5: "Ataque NTP", 6: "Ataque SNMP", 7: "Ataque SSDP"}

def load_model(model_path: str):
    model = open(model_path, "rb")
    return joblib.load(model)


app = FastAPI()
classifier = load_model("./model.joblib")


class DataFeatures(BaseModel):
    L4_SRC_PORT: int
    L4_DST_PORT: int
    PROTOCOL: int
    IN_PKTS: int
    IN_BYTES: float
    OUT_PKTS: int
    OUT_BYTES: float
    
def get_features(data: DataFeatures) -> np.ndarray:
    return np.array(
        [data.L4_SRC_PORT, data.L4_DST_PORT, data.PROTOCOL, data.IN_PKTS, data.IN_BYTES, data.OUT_PKTS, data.OUT_BYTES],
        ndmin=2,
    )


def predict(features: np.ndarray, proba: bool = False) -> Dict:
    if proba:
        probabilities = {
            k: float(v)
            for k, v in zip(
                DDOS_CLASS_MAPPING.values(), classifier.predict_proba(features)[0]
            )
        }
        return {"probabilities": probabilities}

    prediction = int(classifier.predict(features)[0])
    return {
        "prediction": {"value": prediction, "class": DDOS_CLASS_MAPPING[prediction]}
    }


@app.post("/api/v1/predict")
async def get_prediction(data: DataFeatures):
    features = get_features(data)
    return predict(features)


@app.post("/api/v1/proba")
async def get_probabilities(data: DataFeatures):
    features = get_features(data)
    return predict(features, proba=True)


@app.get("/", response_class=HTMLResponse)
def index():
    return (
        "<p>Hello, This is a REST API used for Polyaxon ML Serving examples!</p>"
        "<p>Click the fullscreen button the get the URL of your serving API!<p/>"
    )
