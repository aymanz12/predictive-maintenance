import pandas as pd
import yaml
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="src/config/config.yaml"):
    """Load configuration from the YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise

def ingest_data(config_path="src/config/config.yaml"):
    """Download AI4I dataset and save to raw folder."""
    config = load_config(config_path)
    raw_path = config['data']['raw_path']
    url = config['data']['dataset_url']

    # Create directory
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    # 1. Load or Download
    if os.path.exists(raw_path):
        logging.info(f"File found at {raw_path}. Loading locally...")
        df = pd.read_csv(raw_path)
    else:
        logging.info(f"Downloading data from {url}...")
        try:
            df = pd.read_csv(url)
            df.to_csv(raw_path, index=False)
            logging.info(f"Data saved to {raw_path}")
        except Exception as e:
            logging.error(f"Failed to ingest data: {e}")
            raise

    # 2. Validation & Sanity Checks (Run this ALWAYS)
    logging.info(f"Dataset Shape: {df.shape}")

    expected_cols = ['UDI', 'Product ID', 'Type', 'Air temperature [K]', 
                     'Process temperature [K]', 'Rotational speed [rpm]', 
                     'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 
                     'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if not missing_cols:
        logging.info("✔ Column validation passed.")
    else:
        logging.warning(f"⚠ Missing columns: {missing_cols}")

    # Log specific stats relevant to this dataset
    logging.info(f"Null values:\n{df.isnull().sum()[df.isnull().sum() > 0]}") # Only show cols with nulls
    
    return df

if __name__ == "__main__":
    ingest_data()