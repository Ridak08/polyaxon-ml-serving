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
    Protocol
    Flow Duration
    Total Fwd Packets
    Total Backward Packets
    Total Length of Fwd Packets
    Total Length of Bwd Packets
    Fwd Packet Length Max
    Fwd Packet Length Min
    Fwd Packet Length Mean
    Fwd Packet Length Std
    Bwd Packet Length Max
    Bwd Packet Length Min
    Bwd Packet Length Mean
    Bwd Packet Length Std
    Flow Bytes/s
    Flow Packets/s
    Flow IAT Mean
    Flow IAT Std
    Flow IAT Max
    Flow IAT Min
    Fwd IAT Total
    Fwd IAT Mean
    Fwd IAT Std
    Fwd IAT Max
    Fwd IAT Min
    Bwd IAT Total
    Bwd IAT Mean
    Bwd IAT Std
    Bwd IAT Max
    Bwd IAT Min
    Fwd PSH Flags
    Fwd Header Length
    Bwd Header Length
    Fwd Packets/s
    Bwd Packets/s
    Min Packet Length
    Max Packet Length
    Packet Length Mean
    Packet Length Std
    Packet Length Variance
    SYN Flag Count
    ACK Flag Count
    URG Flag Count
    CWE Flag Count
    Down/Up Ratio
    Average Packet Size
    Avg Fwd Segment Size
    Avg Bwd Segment Size
    Init_Win_bytes_forward
    Init_Win_bytes_backward
    act_data_pkt_fwd
    min_seg_size_forward
    Active Mean
    Active Std
    Active Max
    Active Min
    Idle Mean
    Idle Std
    Idle Max
    Idle Min
    Inbound
    
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


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
