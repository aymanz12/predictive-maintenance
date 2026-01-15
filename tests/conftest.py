import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_raw_data():
    """Sample raw data matching AI4I dataset structure."""
    return pd.DataFrame({
        'UDI': [1, 2, 3, 4, 5],
        'Product ID': ['L47181', 'L47182', 'M14773', 'M14774', 'H20583'],
        'Type': ['L', 'L', 'M', 'M', 'H'],
        'Air temperature [K]': [298.1, 298.2, 298.3, 298.4, 298.5],
        'Process temperature [K]': [308.6, 308.7, 308.8, 308.9, 309.0],
        'Rotational speed [rpm]': [1551, 1552, 1553, 1554, 1555],
        'Torque [Nm]': [42.8, 43.1, 43.4, 43.7, 44.0],
        'Tool wear [min]': [0, 10, 20, 30, 40],
        'Machine failure': [0, 0, 0, 1, 0],
        'TWF': [0, 0, 0, 0, 0],
        'HDF': [0, 0, 0, 0, 0],
        'PWF': [0, 0, 0, 1, 0],
        'OSF': [0, 0, 0, 0, 0],
        'RNF': [0, 0, 0, 0, 0]
    })

@pytest.fixture
def sample_processed_data():
    """Sample processed data with cleaned column names."""
    return pd.DataFrame({
        'type': ['L', 'M', 'H', 'L', 'M'],
        'air_temperature': [298.1, 298.2, 298.3, 298.4, 298.5],
        'process_temperature': [308.6, 308.7, 308.8, 308.9, 309.0],
        'rotational_speed': [1551, 1552, 1553, 1554, 1555],
        'torque': [42.8, 43.1, 43.4, 43.7, 44.0],
        'tool_wear': [0, 10, 20, 30, 40],
        'machine_failure': [0, 0, 0, 1, 0]
    })

@pytest.fixture
def sample_feature_input():
    """Sample input data for feature engineering (without target)."""
    return pd.DataFrame({
        'type': ['L', 'M', 'H'],
        'air_temperature': [300.0, 298.0, 295.0],
        'process_temperature': [310.0, 308.0, 305.0],
        'rotational_speed': [1500, 1600, 1700],
        'torque': [40.0, 45.0, 50.0],
        'tool_wear': [10, 50, 100]
    })

@pytest.fixture
def mock_config():
    """Mock configuration dictionary."""
    return {
        'data': {
            'raw_path': 'data/raw/ai4i2020.csv',
            'processed_train_path': 'data/processed/train.csv',
            'processed_test_path': 'data/processed/test.csv',
            'dataset_url': 'https://raw.githubusercontent.com/example/dataset.csv'
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42
        },
        'model': {
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
    }

@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)
