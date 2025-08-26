
from fastapi import FastAPI, Request
import joblib
import numpy as np
import uvicorn

app = FastAPI()

# Load model sekali saat cold start (efisien)
model = joblib.load("model.joblib")

@app.post("/api/predict")
async def predict(req: Request):
    body = await req.json()
    # ekspektasi body: {"age": 30, "income": 4000, "loan_amt": 1500}
    features = np.array([[body["age"], body["income"], body["loan_amt"]]])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features).tolist()[0] if hasattr(model, "predict_proba") else None
    return {"prediction": int(pred), "probability": proba}
