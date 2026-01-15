import pandas as pd
import yaml
import pickle
import logging
import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="src/config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_artifacts():
    """Load the saved model and engineer from the ROOT models folder."""
    # <--- CHANGED THESE PATHS
    model_path = "models/xgboost_model.pkl"
    engineer_path = "models/feature_engineer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(engineer_path):
        raise FileNotFoundError("❌ Artifacts not found in models/ folder. Run training_pipeline.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(engineer_path, "rb") as f:
        engineer = pickle.load(f)
        
    return model, engineer

def run_evaluation():
    config = load_config()
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("predictive_maintenance_prod")
    
    with mlflow.start_run(run_name="pipeline_evaluation"):
        # 1. Load Test Data
        test_path = config['data']['processed_test_path']
        logging.info(f"Loading processed test data from {test_path}...")
        df_test = pd.read_csv(test_path)
        
        target = "machine_failure"
        drop_cols = [target, 'twf', 'hdf', 'pwf', 'osf', 'rnf']
        
        X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
        y_test = df_test[target]
        
        # 2. Load Artifacts from Root models/
        logging.info("Loading pre-trained model and engineer...")
        try:
            model, engineer = load_artifacts()
        except FileNotFoundError as e:
            logging.error(e)
            return

        # 3. Apply Engineering (Transform ONLY)
        X_test_eng = engineer.transform(X_test)
        
        # Verify feature order matches training
        if hasattr(engineer, 'feature_names_') and engineer.feature_names_:
            actual_cols = X_test_eng.columns.tolist()
            if actual_cols != engineer.feature_names_:
                logging.warning(f"⚠️ Column mismatch! Expected: {engineer.feature_names_}, Got: {actual_cols}")
            else:
                logging.info("✅ Feature columns match training order")
        else:
            logging.warning("⚠️ Engineer doesn't have feature_names_ attribute")
        
        # 4. Predict
        logging.info("Running predictions on Test set...")
        y_pred = model.predict(X_test_eng)
        
        # 5. Metrics
        metrics = {
            "test_f1": f1_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred)
        }
        
        logging.info(f"📊 Evaluation Results: {metrics}")
        print("\n" + classification_report(y_test, y_pred))
        
        # 6. Log Metrics
        mlflow.log_metrics(metrics)
        
        # 7. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Evaluation Pipeline Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        plot_path = "evaluation_confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        
        logging.info("✅ Evaluation Complete.")

if __name__ == "__main__":
    run_evaluation()