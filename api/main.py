import sys
import os
import pickle
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Any

# Add project root to sys.path to allow importing from src
# current file is in api/, so project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from api.schemas import MachineData
# Import FeatureEngineer to ensure class definition is available for unpickling
try:
    from src.features.feature_engineering import FeatureEngineer
except ImportError:
    # If src is not found, we might need to be explicit or check sys.path
    logging.warning("Could not import FeatureEngineer from src.features.feature_engineering")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global storage for models
models: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models on startup and clean up on shutdown.
    """
    try:
        model_dir = os.path.join(project_root, "models")
        
        # Load Feature Engineer
        fe_path = os.path.join(model_dir, "feature_engineer.pkl")
        if os.path.exists(fe_path):
            with open(fe_path, "rb") as f:
                models["feature_engineer"] = pickle.load(f)
            logger.info("Feature Engineer loaded successfully.")
        else:
            logger.error(f"Feature Engineer not found at {fe_path}")

        # Load XGBoost Model
        model_path = os.path.join(model_dir, "xgboost_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                models["model"] = pickle.load(f)
            logger.info("XGBoost Model loaded successfully.")
        else:
            logger.error(f"XGBoost Model not found at {model_path}")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # We continue even if models fail to load, but endpoints will error out.
    
    yield
    
    models.clear()

app = FastAPI(title="Predictive Maintenance API", lifespan=lifespan)

@app.post("/predict")
async def predict(data: MachineData):
    if "feature_engineer" not in models or "model" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # 1. Convert input to DataFrame
        # model_dump() returns a dict, we wrap it in a list to create a one-row DataFrame
        df = pd.DataFrame([data.model_dump()])
        
        # 2. Transform features
        # The FeatureEngineer expects a DataFrame and will handle column ordering
        df_transformed = models["feature_engineer"].transform(df)
        
        # 3. Verify column order matches training (defensive check)
        if hasattr(models["feature_engineer"], 'feature_names_'):
            expected_cols = models["feature_engineer"].feature_names_
            actual_cols = df_transformed.columns.tolist()
            if actual_cols != expected_cols:
                logger.error(f"❌ Column mismatch! Got {actual_cols}, expected {expected_cols}")
                raise HTTPException(status_code=500, detail="Feature mismatch between training and inference")
            logger.debug(f"✅ Features match training: {actual_cols}")
        else:
            logger.warning("⚠️ FeatureEngineer missing feature_names_ attribute")

        # 4. Predict
        # predict() returns an array, we take the first element
        prediction = models["model"].predict(df_transformed)[0]
        
        # predict_proba() returns array of probabilities for each class
        # We assume binary classification (0: Safe, 1: Failure)
        # We return the probability of Failure (index 1)
        probs = models["model"].predict_proba(df_transformed)
        failure_prob = probs[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(failure_prob)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
