import pandas as pd
import xgboost as xgb
import yaml
import pickle
import logging
import os
import mlflow
import mlflow.xgboost
from src.features.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="src/config/config.yaml"):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        raise

def run_training():
    config = load_config()
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("predictive_maintenance_prod")
    
    with mlflow.start_run(run_name="pipeline_training"):
        # 1. Load Data
        train_path = config['data']['processed_train_path']
        logging.info(f"Loading processed training data from {train_path}...")
        df_train = pd.read_csv(train_path)
        
        # 2. Separate Target
        target = "machine_failure"
        drop_cols = [target, 'twf', 'hdf', 'pwf', 'osf', 'rnf']
        
        X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
        y_train = df_train[target]
        
        # 3. Feature Engineering
        logging.info("Fitting Feature Engineer...")
        engineer = FeatureEngineer()
        X_train_eng = engineer.fit_transform(X_train)
        
        # Log feature names for verification
        logging.info(f"Engineered features: {engineer.feature_names_}")
        logging.info(f"Feature shape: {X_train_eng.shape}") 
        
        # 4. Train Model
        logging.info("Training XGBoost Model...")
        params = config['model']['params']
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_eng, y_train)
        
        # 5. Save Artifacts to Root 'models/' folder
        output_dir = "models"  
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, "xgboost_model.pkl")
        engineer_path = os.path.join(output_dir, "feature_engineer.pkl")
        
        logging.info(f"Saving artifacts to {output_dir}...")
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        with open(engineer_path, "wb") as f:
            pickle.dump(engineer, f)
            
        logging.info(f"✅ Training finished. Model and Engineer saved to {output_dir}/")
        
        # 6. Log to MLflow
        mlflow.log_params(params)
        mlflow.xgboost.log_model(model, "model")

if __name__ == "__main__":
    run_training()