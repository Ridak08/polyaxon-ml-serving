from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

DDOS_CLASS_MAPPING = {0: "Benigno", 1: "DNS", 2: "Syn", 3: "UDP", 4: "NetBIOS", 5: "NTP", 6: "SNMP", 7: "SSDP"}


def load_model(model_path: str):
    model = open(model_path, "rb")
    return joblib.load(model)


app = FastAPI()
classifier = load_model("./model.joblib")


class DataFeatures(BaseModel):
    Protocol: int
    Flow Duration: int
    Total Fwd Packets: int
    Total Backward Packets: int
    Total Length of Fwd Packets: float
    Total Length of Bwd Packets: float
    Fwd Packet Length Max: float
    Fwd Packet Length Min: float
    Fwd Packet Length Mean: float
    Fwd Packet Length Std: float
    Bwd Packet Length Max: float
    Bwd Packet Length Min: float
    Bwd Packet Length Mean: float
    Bwd Packet Length Std: float
    Flow Bytes/s: float
    Flow Packets/s: float
    Flow IAT Mean: float
    Flow IAT Std: float
    Flow IAT Max: float
    Flow IAT Min: float
    Fwd IAT Total: float
    Fwd IAT Mean: float
    Fwd IAT Std: float
    Fwd IAT Max: float
    Fwd IAT Min: float
    Bwd IAT Total: float
    Bwd IAT Mean: float
    Bwd IAT Std: float
    Bwd IAT Max: float
    Bwd IAT Min: float
    Fwd PSH Flags: float
    Fwd Header Length: float
    Bwd Header Length: float
    Fwd Packets/s: float
    Bwd Packets/s: float
    Min Packet Length: float
    Max Packet Length: float
    Packet Length Mean: float
    Packet Length Std: float
    Packet Length Variance: float
    SYN Flag Count: float
    ACK Flag Count: float
    URG Flag Count: float
    CWE Flag Count: float
    Down/Up Ratio: float
    Average Packet Size: float
    Avg Fwd Segment Size: float
    Avg Bwd Segment Size: float
    Init_Win_bytes_forward: float
    Init_Win_bytes_backward: float
    act_data_pkt_fwd: float
    min_seg_size_forward: float
    Active Mean: float
    Active Std: float
    Active Max: float
    Active Min: float
    Idle Mean: float
    Idle Std: float
    Idle Max: float
    Idle Min: float
    Inbound: float
    
def get_features(data: DataFeatures) -> np.ndarray:
    return np.array(
        [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width],
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
