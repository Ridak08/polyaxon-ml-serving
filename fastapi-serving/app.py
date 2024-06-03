from typing import Dict
#
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
    Flow_Duration: int
    Total_Fwd_Packets: int
    Total_Backward_Packets: int
    Total_Length_of_Fwd_Packets: float
    Total_Length_of_Bwd_Packets: float
    Fwd_Packet_Length_Max: float
    Fwd_Packet_Length_Min: float
    Fwd_Packet_Length_Mean: float
    Fwd_Packet_Length_Std: float
    Bwd_Packet_Length_Max: float
    Bwd_Packet_Length_Min: float
    Bwd_Packet_Length_Mean: float
    Bwd_Packet_Length_Std: float
    Flow_Bytes_s: float
    Flow_Packets_s: float
    Flow_IAT_Mean: float
    Flow_IAT_Std: float
    Flow_IAT_Max: float
    Flow_IAT_Min: float
    Fwd_IAT_Total: float
    Fwd_IAT_Mean: float
    Fwd_IAT_Std: float
    Fwd_IAT_Max: float
    Fwd_IAT_Min: float
    Bwd_IAT_Total: float
    Bwd_IAT_Mean: float
    Bwd_IAT_Std: float
    Bwd_IAT_Max: float
    Bwd_IAT_Min: float
    Fwd_PSH_Flags: float
    Fwd_Header_Length: float
    Bwd_Header_Length: float
    Fwd_Packets_s: float
    Bwd_Packets_s: float
    Min_Packet_Length: float
    Max_Packet_Length: float
    Packet_Length_Mean: float
    Packet_Length_Std: float
    Packet_Length_Variance: float
    SYN_Flag_Count: float
    ACK_Flag_Count: float
    URG_Flag_Count: float
    CWE_Flag_Count: float
    Down_Up_Ratio: float
    Average_Packet_Size: float
    Avg_Fwd_Segment_Size: float
    Avg_Bwd_Segment_Size: float
    Init_Win_Bytes_Forward: float
    Init_Win_Bytes_Backward: float
    Act_Data_Pkt_Fwd: float
    Min_Seg_Size_Forward: float
    Active_Mean: float
    Active_Std: float
    Active_Max: float
    Active_Min: float
    Idle_Mean: float
    Idle_Std: float
    Idle_Max: float
    Idle_Min: float
    Inbound: float
    
def get_features(data: DataFeatures) -> np.ndarray:
    return np.array(
        [data.Protocol, data.Flow_Duration, data.Total_Fwd_Packets, data.Total_Backward_Packets, data.Total_Length_of_Fwd_Packets, data.Total_Length_of_Bwd_Packets, data.Fwd_Packet_Length_Max,data.Fwd_Packet_Length_Min, data.Fwd_Packet_Length_Mean, data.Fwd_Packet_Length_Std,data.Bwd_Packet_Length_Max, data.Bwd_Packet_Length_Min, data.Bwd_Packet_Length_Mean,data.Bwd_Packet_Length_Std, data.Flow_Bytes_s, data.Flow_Packets_s,data.Flow_IAT_Mean, data.Flow_IAT_Std, data.Flow_IAT_Max,data.Flow_IAT_Min, data.Fwd_IAT_Total, data.Fwd_IAT_Mean,data.Fwd_IAT_Std, data.Fwd_IAT_Max, data.petal_length,data.Fwd_IAT_Min, data.Bwd_IAT_Total, data.Bwd_IAT_Mean,data.Bwd_IAT_Std, data.Bwd_IAT_Max, data.Bwd_IAT_Min,data.Fwd_PSH_Flags, data.Fwd_Header_Length, data.Bwd_Header_Length,data.Fwd_Packets_s, data.Bwd_Packets_s, data.Min_Packet_Length,data.Max_Packet_Length, data.Packet_Length_Mean, data.Packet_Length_Std,data.Packet_Length_Variance, data.SYN_Flag_Count, data.ACK_Flag_Count,data.URG_Flag_Count, data.CWE_Flag_Count, data.Down_Up_Ratio, data.Average_Packet_Size, data.Avg_Fwd_Segment_Size, data.Avg_Bwd_Segment_Size,data.Init_Win_Bytes_Forward, data.Init_Win_Bytes_Backward, data.Act_Data_Pkt_Fwd,data.Min_Seg_Size_Forward, data.Active_Mean, data.Active_Std,data.Active_Max, data.Active_Min, data.Idle_Mean,data.Idle_Std, data.Idle_Max, data.Idle_Min, data.Inbound],
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
