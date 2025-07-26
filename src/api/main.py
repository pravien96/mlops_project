from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
import sqlite3
import json
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Prometheus metrics
prediction_requests = Counter('ml_prediction_requests_total', 'Total prediction requests')
prediction_duration = Histogram('ml_prediction_duration_seconds', 'Prediction request duration')

app = FastAPI(
    title="Housing Price Predictor",
    description="ML API for predicting California housing prices",
    version="1.0.0"
)

# Load model and scaler at startup
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    scaler = None

class HousingFeatures(BaseModel):
    MedInc: float = Field(..., ge=0, description="Median income in block group")
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age in block group")
    AveRooms: float = Field(..., ge=0, description="Average number of rooms per household")
    AveBedrms: float = Field(..., ge=0, description="Average number of bedrooms per household")
    Population: float = Field(..., ge=0, description="Block group population")
    AveOccup: float = Field(..., ge=0, description="Average number of household members")
    Latitude: float = Field(..., ge=-90, le=90, description="Block group latitude")
    Longitude: float = Field(..., ge=-180, le=180, description="Block group longitude")

class PredictionLogger:
    def __init__(self, db_path="logs/predictions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                request_data TEXT,
                prediction REAL,
                response_time_ms REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_prediction(self, request_data, prediction, response_time):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO predictions (request_data, prediction, response_time_ms)
            VALUES (?, ?, ?)
        ''', (json.dumps(request_data), prediction, response_time))
        conn.commit()
        conn.close()

prediction_logger = PredictionLogger()

@app.get("/")
async def root():
    return {"message": "Housing Price Predictor API", "status": "running"}

@app.get("/health")
async def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "scaler_status": scaler_status
    }

@app.post("/predict")
async def predict_price(features: HousingFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    prediction_requests.inc()
    
    with prediction_duration.time():
        try:
            # Convert to numpy array
            input_data = np.array([[
                features.MedInc,
                features.HouseAge,
                features.AveRooms,
                features.AveBedrms,
                features.Population,
                features.AveOccup,
                features.Latitude,
                features.Longitude
            ]])
            
            # Scale the input
            scaled_input = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(scaled_input)[0]
            
            # Log the prediction
            import time
            start_time = time.time()
            response_time = (time.time() - start_time) * 1000
            
            prediction_logger.log_prediction(
                features.dict(), 
                prediction, 
                response_time
            )
            
            logger.info(f"Prediction made: {prediction}")
            
            return {
                "predicted_price": round(prediction, 2),
                "price_in_hundreds_of_thousands": f"${round(prediction * 100, 2)}k",
                "features_used": features.dict()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/prediction-history")
async def get_prediction_history(limit: int = 10):
    """Get recent predictions"""
    try:
        conn = sqlite3.connect(prediction_logger.db_path)
        cursor = conn.execute('''
            SELECT timestamp, request_data, prediction, response_time_ms
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "timestamp": row[0],
                "request_data": json.loads(row[1]),
                "prediction": row[2],
                "response_time_ms": row[3]
            })
        
        conn.close()
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
