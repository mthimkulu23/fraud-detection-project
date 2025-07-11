from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import os
import sys
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Adjust sys.path for direct execution during development
if __name__ == "__main__" and "backend" not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.ml_service import MLService
    from backend.config import MODEL_PATH
else:
    from .ml_service import MLService
    from .config import MODEL_PATH


# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time fraud detection using a pre-trained ML model.",
    version="1.0.0"
)


origins = [
    "http://localhost",
    "http://localhost:8000",
    "null",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRONTEND_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "frontend"

static_files_dir = FRONTEND_DIR / "static"
print(f"Attempting to serve static files from: {static_files_dir.resolve()}")
app.mount("/static", StaticFiles(directory=static_files_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the main frontend HTML page.
    """
    html_file_path = FRONTEND_DIR / "index.html"
    if not html_file_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    with open(html_file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

try:
    ml_service = MLService(model_path=MODEL_PATH)
    print(f"ML Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading ML model: {e}")
    ml_service = None


class Transaction(BaseModel):
    amount: float
    transaction_frequency_24h: int
    location_risk_score: float
    time_of_day_hour: int
    is_international: int
    ip_country_mismatch: int
    failed_auth_attempts: int = 0
    card_validation_status: str # New: Added card_validation_status
    ip_address: str = None # New: Added ip_address with a default of None


@app.get("/health")
async def health_check():
    """
    Checks the health of the API and ML model loading status.
    """
    if ml_service and ml_service.model:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False, "message": "ML model not loaded"}


@app.post("/predict_fraud")
async def predict_fraud(transaction: Transaction):
    """
    Receives transaction data and returns a fraud prediction.
    """
    if not ml_service or not ml_service.model:
        raise HTTPException(status_code=503, detail="ML service not ready. Model not loaded.")

    try:
        transaction_data = transaction.dict()
        
        # NOTE: The current ML model (fraud_detection_model.joblib) is assumed to NOT
        # use 'failed_auth_attempts', 'card_validation_status', or 'ip_address' as features.
        # If you wish for the model to incorporate these, the ML model would need to be retrained
        # with these new features. For now, we'll remove them from the data sent to predict
        # to avoid errors if the model expects a fixed number of features.
        data_for_prediction = {
            k: v for k, v in transaction_data.items()
            if k not in ['failed_auth_attempts', 'card_validation_status', 'ip_address']
        }

        prediction, probability = ml_service.predict(data_for_prediction)
        
        return {
            "is_fraud": bool(prediction),
            "fraud_probability": float(probability),
            "message": "Transaction flagged as Fraud" if prediction else "Transaction is Not Fraud"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}. Please run ml_service.py to generate it.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
