import pytest
import pandas as pd
import pickle
import os
from fastapi.testclient import TestClient
from api.main import app


class TestPredictionConsistency:
    """Test that predictions are consistent between evaluation pipeline and API."""
    
    @pytest.fixture
    def load_artifacts(self):
        """Load the trained model and feature engineer."""
        model_path = "models/xgboost_model.pkl"
        engineer_path = "models/feature_engineer.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(engineer_path):
            pytest.skip("Models not found. Run training_pipeline.py first.")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(engineer_path, "rb") as f:
            engineer = pickle.load(f)
        
        return model, engineer
    
    @pytest.fixture
    def sample_data(self):
        """Sample test data for prediction consistency tests."""
        return [
            {
                "air_temperature": 300.0,
                "process_temperature": 310.0,
                "rotational_speed": 1500,
                "torque": 40.0,
                "tool_wear": 10,
                "type": "M"
            },
            {
                "air_temperature": 298.0,
                "process_temperature": 308.0,
                "rotational_speed": 1700,
                "torque": 30.0,
                "tool_wear": 5,
                "type": "L"
            },
            {
                "air_temperature": 302.0,
                "process_temperature": 315.0,
                "rotational_speed": 1300,
                "torque": 60.0,
                "tool_wear": 200,
                "type": "H"
            }
        ]
    
    def test_feature_engineer_has_feature_names(self, load_artifacts):
        """Test that the feature engineer has feature_names_ attribute set."""
        model, engineer = load_artifacts
        
        assert hasattr(engineer, 'feature_names_'), "FeatureEngineer missing feature_names_ attribute"
        assert engineer.feature_names_ is not None, "feature_names_ should not be None"
        assert len(engineer.feature_names_) > 0, "feature_names_ should not be empty"
        
        # Expected features
        expected_features = [
            'type', 'air_temperature', 'process_temperature', 
            'rotational_speed', 'torque', 'tool_wear',
            'temp_difference', 'power_factor', 'strain_wear_product'
        ]
        
        assert engineer.feature_names_ == expected_features, \
            f"Feature names mismatch. Expected {expected_features}, got {engineer.feature_names_}"
    
    def test_direct_prediction_vs_api(self, load_artifacts, sample_data):
        """Test that direct model predictions match API predictions."""
        model, engineer = load_artifacts
        with TestClient(app) as client:
            for sample in sample_data:
                # 1. Direct prediction (like evaluation pipeline)
                df = pd.DataFrame([sample])
                df_transformed = engineer.transform(df)
                direct_prediction = int(model.predict(df_transformed)[0])
                direct_probability = float(model.predict_proba(df_transformed)[0][1])
                
                # 2. API prediction
                response = client.post("/predict", json=sample)
                assert response.status_code == 200, f"API failed for sample {sample}"
                
                api_result = response.json()
                api_prediction = api_result["prediction"]
                api_probability = api_result["probability"]
                
                # 3. Compare results
                assert direct_prediction == api_prediction, \
                    f"Prediction mismatch for {sample}. Direct: {direct_prediction}, API: {api_prediction}"
                
                assert abs(direct_probability - api_probability) < 1e-6, \
                    f"Probability mismatch for {sample}. Direct: {direct_probability}, API: {api_probability}"
    
    def test_column_ordering_consistency(self, load_artifacts, sample_data):
        """Test that feature engineering produces consistent column ordering."""
        model, engineer = load_artifacts
        
        for sample in sample_data:
            df = pd.DataFrame([sample])
            df_transformed = engineer.transform(df)
            
            # Check that columns match stored feature names
            actual_cols = df_transformed.columns.tolist()
            expected_cols = engineer.feature_names_
            
            assert actual_cols == expected_cols, \
                f"Column order mismatch. Expected {expected_cols}, got {actual_cols}"
    
    def test_all_type_variants(self, load_artifacts):
        """Test predictions work correctly for all type variants (L, M, H)."""
        model, engineer = load_artifacts
        base_sample = {
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 10
        }
        
        with TestClient(app) as client:
            for type_val in ['L', 'M', 'H']:
                sample = {**base_sample, "type": type_val}
                
                # Direct prediction
                df = pd.DataFrame([sample])
                df_transformed = engineer.transform(df)
                
                # Verify type encoding
                assert 'type' in df_transformed.columns, "type column missing after transform"
                encoded_type = int(df_transformed['type'].iloc[0])
                
                # Verify encoding is correct
                type_mapping = {'L': 0, 'M': 1, 'H': 2}
                assert encoded_type == type_mapping[type_val], \
                    f"Type encoding incorrect. Expected {type_mapping[type_val]}, got {encoded_type}"
                
                # Test API
                response = client.post("/predict", json=sample)
            assert response.status_code == 200, f"API failed for type {type_val}"
            
            result = response.json()
            assert "prediction" in result
            assert "probability" in result
            assert result["prediction"] in [0, 1]
            assert 0 <= result["probability"] <= 1
    
    def test_feature_engineering_creates_derived_features(self, load_artifacts, sample_data):
        """Test that feature engineering creates the expected derived features."""
        model, engineer = load_artifacts
        
        for sample in sample_data:
            df = pd.DataFrame([sample])
            df_transformed = engineer.transform(df)
            
            # Check that derived features are present
            assert 'temp_difference' in df_transformed.columns, "temp_difference missing"
            assert 'power_factor' in df_transformed.columns, "power_factor missing"
            assert 'strain_wear_product' in df_transformed.columns, "strain_wear_product missing"
            
            # Verify calculations
            expected_temp_diff = sample['process_temperature'] - sample['air_temperature']
            actual_temp_diff = df_transformed['temp_difference'].iloc[0]
            assert abs(actual_temp_diff - expected_temp_diff) < 1e-6, \
                "temp_difference calculation incorrect"
            
            expected_power = sample['torque'] * sample['rotational_speed']
            actual_power = df_transformed['power_factor'].iloc[0]
            assert abs(actual_power - expected_power) < 1e-6, \
                "power_factor calculation incorrect"
            
            expected_strain = sample['torque'] * sample['tool_wear']
            actual_strain = df_transformed['strain_wear_product'].iloc[0]
            assert abs(actual_strain - expected_strain) < 1e-6, \
                "strain_wear_product calculation incorrect"
