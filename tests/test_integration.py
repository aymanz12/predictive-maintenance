import pytest
import pandas as pd
import pickle
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.features.feature_engineering import FeatureEngineer


class TestEndToEndPrediction:
    """Integration tests for the complete prediction pipeline."""
    
    @pytest.fixture
    def real_models(self):
        """Load real models if they exist, otherwise skip."""
        model_path = Path("models/xgboost_model.pkl")
        engineer_path = Path("models/feature_engineer.pkl")
        
        if not model_path.exists() or not engineer_path.exists():
            pytest.skip("Models not found. Run training pipeline first.")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(engineer_path, "rb") as f:
            engineer = pickle.load(f)
        
        return {"model": model, "engineer": engineer}
    
    def test_feature_engineering_plus_prediction(self, real_models):
        """Test feature engineering followed by model prediction."""
        # Sample input data
        df = pd.DataFrame({
            'type': ['L', 'M', 'H'],
            'air_temperature': [300.0, 298.0, 295.0],
            'process_temperature': [310.0, 308.0, 305.0],
            'rotational_speed': [1500, 1600, 1700],
            'torque': [40.0, 45.0, 50.0],
            'tool_wear': [10, 50, 100]
        })
        
        # Transform features
        engineer = real_models['engineer']
        df_transformed = engineer.transform(df)
        
        # Make predictions
        model = real_models['model']
        predictions = model.predict(df_transformed)
        probabilities = model.predict_proba(df_transformed)
        
        # Verify outputs
        assert len(predictions) == 3
        assert predictions.shape == (3,)
        assert probabilities.shape == (3, 2)
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= prob <= 1 for prob in probabilities.flatten())
    
    def test_single_sample_prediction(self, real_models):
        """Test prediction on a single sample."""
        df = pd.DataFrame({
            'type': ['M'],
            'air_temperature': [300.0],
            'process_temperature': [310.0],
            'rotational_speed': [1500],
            'torque': [40.0],
            'tool_wear': [10]
        })
        
        engineer = real_models['engineer']
        model = real_models['model']
        
        df_transformed = engineer.transform(df)
        prediction = model.predict(df_transformed)[0]
        probability = model.predict_proba(df_transformed)[0]
        
        assert prediction in [0, 1]
        assert len(probability) == 2
        assert sum(probability) == pytest.approx(1.0, abs=1e-5)
    
    def test_high_risk_scenario(self, real_models):
        """Test prediction on high-risk scenario (likely failure)."""
        # High temperature difference, high torque, high wear
        df = pd.DataFrame({
            'type': ['H'],
            'air_temperature': [300.0],
            'process_temperature': [320.0],  # High process temp
            'rotational_speed': [1200],  # Low speed
            'torque': [70.0],  # High torque
            'tool_wear': [280]  # High wear
        })
        
        engineer = real_models['engineer']
        model = real_models['model']
        
        df_transformed = engineer.transform(df)
        prediction = model.predict(df_transformed)[0]
        failure_prob = model.predict_proba(df_transformed)[0][1]
        
        # Higher probability of failure expected
        assert isinstance(prediction, (int, np.integer))
        assert 0 <= failure_prob <= 1
    
    def test_low_risk_scenario(self, real_models):
        """Test prediction on low-risk scenario (unlikely failure)."""
        # Normal operating conditions
        df = pd.DataFrame({
            'type': ['L'],
            'air_temperature': [298.0],
            'process_temperature': [308.0],  # Normal temp diff
            'rotational_speed': [1550],
            'torque': [40.0],
            'tool_wear': [5]  # Low wear
        })
        
        engineer = real_models['engineer']
        model = real_models['model']
        
        df_transformed = engineer.transform(df)
        prediction = model.predict(df_transformed)[0]
        failure_prob = model.predict_proba(df_transformed)[0][1]
        
        # Lower probability of failure expected
        assert isinstance(prediction, (int, np.integer))
        assert 0 <= failure_prob <= 1


class TestAPIIntegration:
    """Integration tests for API with real models."""
    
    def test_api_with_mocked_models(self, sample_feature_input):
        """Test API endpoint with mocked models."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        mock_engineer = FeatureEngineer()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.95, 0.05]]
        
        models_dict = {
            'feature_engineer': mock_engineer,
            'model': mock_model
        }
        
        with patch('api.main.models', models_dict):
            client = TestClient(app)
            
            payload = {
                "air_temperature": 300.0,
                "process_temperature": 310.0,
                "rotational_speed": 1500,
                "torque": 40.0,
                "tool_wear": 10,
                "type": "M"
            }
            
            response = client.post("/predict", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == 0
            assert data["probability"] == 0.05


class TestDataPipeline:
    """Integration tests for data processing pipeline."""
    
    def test_clean_columns_to_feature_engineering(self, sample_raw_data):
        """Test pipeline from raw data to engineered features."""
        from src.data.preprocessing import clean_column_names
        
        # Clean columns
        df = clean_column_names(sample_raw_data)
        
        # Verify columns are cleaned
        assert 'air_temperature' in df.columns
        assert 'Air temperature [K]' not in df.columns
        
        # Drop non-feature columns
        feature_cols = ['type', 'air_temperature', 'process_temperature', 
                       'rotational_speed', 'torque', 'tool_wear']
        df_features = df[[col for col in feature_cols if col in df.columns]]
        
        # Engineer features
        engineer = FeatureEngineer()
        df_engineered = engineer.transform(df_features)
        
        # Verify engineered features exist
        assert 'temp_difference' in df_engineered.columns
        assert 'power_factor' in df_engineered.columns
        assert 'strain_wear_product' in df_engineered.columns


# Import numpy for type checking
import numpy as np
