import pandas as pd
import yaml
import os
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="src/config/config.yaml"):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise

def clean_column_names(df):
    """
    Standardize column names to snake_case and remove units.
    Matches the logic used in the EDA notebook.
    """
    df.columns = [
        col.replace('[K]', '')
           .replace('[rpm]', '')
           .replace('[Nm]', '')
           .replace('[min]', '')
           .strip()
           .replace(' ', '_')
           .lower() 
        for col in df.columns
    ]
    return df

def preprocess_and_split():
    """
    Load raw data, clean columns, and perform stratified split.
    """
    config = load_config()
    
    raw_path = config['data']['raw_path']
    train_path = config['data']['processed_train_path']
    test_path = config['data']['processed_test_path']
    
    test_size = config['training']['test_size']
    random_state = config['training']['random_state']
    
    # 1. Load Data
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not found. Please run src/data/ingestion.py first.")
    
    logging.info("Loading raw data...")
    df = pd.read_csv(raw_path)
    
    # 2. Clean Columns
    df = clean_column_names(df)
    
    # 3. Drop ID columns (irrelevant for training)
    # We keep failure types (twf, hdf, etc) for now in train/test 
    # so we can analyze them later if needed, but we will drop them during training.
    cols_to_drop = ['udi', 'product_id']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 4. Stratified Split
    logging.info("Splitting data...")
    # Stratify by 'machine_failure' to handle class imbalance
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['machine_failure']
    )
    
    # 5. Save processed split
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logging.info(f"✅ Split completed.")
    logging.info(f"Train set: {train_path} | Shape: {train_df.shape}")
    logging.info(f"Test set:  {test_path} | Shape: {test_df.shape}")

if __name__ == "__main__":
    preprocess_and_split()