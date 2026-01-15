import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pickle
from api.main import app
from api.schemas import MachineData


class TestSchemas:
    """Tests for Pydantic schemas."""
    
    def test_machine_data_valid(self):
        """Test MachineData validation with valid inputs."""
        data = {
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 10,
            "type": "M"
        }
        
        machine_data = MachineData(**data)
        
        assert machine_data.air_temperature == 300.0
        assert machine_data.process_temperature == 310.0
        assert machine_data.rotational_speed == 1500
        assert machine_data.torque == 40.0
        assert machine_data.tool_wear == 10
        assert machine_data.type == "M"
    
    def test_machine_data_invalid_type(self):
        """Test that invalid type values are rejected."""
        data = {
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 10,
            "type": "X"  # Invalid type
        }
        
        with pytest.raises(ValueError):
            MachineData(**data)
    
    def test_machine_data_missing_field(self):
        """Test that missing required fields raise validation error."""
        data = {
            "air_temperature": 300.0,
            # Missing other fields
        }
        
        with pytest.raises(ValueError):
            MachineData(**data)
    
    def test_machine_data_wrong_types(self):
        """Test that wrong data types raise validation error."""
        data = {
            "air_temperature": "not a number",
            "process_temperature": 310.0,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 10,
            "type": "M"
        }
        
        with pytest.raises(ValueError):
            MachineData(**data)


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def mock_models(self):
        """Mock loaded models."""
        mock_engineer = MagicMock()
        mock_model = MagicMock()
        
        # Mock transform to return a DataFrame with expected columns
        import pandas as pd
        mock_engineer.transform.return_value = pd.DataFrame({
            'type': [1],
            'air_temperature': [300.0],
            'process_temperature': [310.0],
            'rotational_speed': [1500],
            'torque': [40.0],
            'tool_wear': [10],
            'temp_difference': [10.0],
            'power_factor': [60000.0],
            'strain_wear_product': [400.0]
        })
        
        # NEW: Mock feature_names_ attribute to match expected features
        mock_engineer.feature_names_ = [
            'type', 'air_temperature', 'process_temperature', 
            'rotational_speed', 'torque', 'tool_wear',
            'temp_difference', 'power_factor', 'strain_wear_product'
        ]
        
        # Mock predictions
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.98, 0.02]]
        
        return {
            'feature_engineer': mock_engineer,
            'model': mock_model
        }
    
    def test_predict_endpoint_success(self, mock_models):
        """Test /predict endpoint with valid data."""
        with patch('api.main.models', mock_models):
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
            assert "prediction" in data
            assert "probability" in data
            assert data["prediction"] in [0, 1]
            assert 0 <= data["probability"] <= 1
    
    def test_predict_endpoint_invalid_schema(self, mock_models):
        """Test /predict endpoint with invalid schema."""
        with patch('api.main.models', mock_models):
            client = TestClient(app)
            
            payload = {
                "air_temperature": 300.0,
                # Missing required fields
            }
            
            response = client.post("/predict", json=payload)
            
            assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_type_value(self, mock_models):
        """Test /predict endpoint with invalid type value."""
        with patch('api.main.models', mock_models):
            client = TestClient(app)
            
            payload = {
                "air_temperature": 300.0,
                "process_temperature": 310.0,
                "rotational_speed": 1500,
                "torque": 40.0,
                "tool_wear": 10,
                "type": "Z"  # Invalid
            }
            
            response = client.post("/predict", json=payload)
            
            assert response.status_code == 422
    
    def test_predict_endpoint_models_not_loaded(self):
        """Test /predict endpoint when models are not loaded."""
        with patch('api.main.models', {}):
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
            
            assert response.status_code == 503  # Service unavailable
    
    def test_predict_endpoint_prediction_error(self, mock_models):
        """Test /predict endpoint handles prediction errors."""
        # Make model.predict raise an exception
        mock_models['model'].predict.side_effect = Exception("Prediction failed")
        
        with patch('api.main.models', mock_models):
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
            
            assert response.status_code == 500
    
    def test_predict_endpoint_returns_correct_format(self, mock_models):
        """Test that /predict endpoint returns expected format."""
        with patch('api.main.models', mock_models):
            client = TestClient(app)
            
            payload = {
                "air_temperature": 300.0,
                "process_temperature": 310.0,
                "rotational_speed": 1500,
                "torque": 40.0,
                "tool_wear": 10,
                "type": "L"
            }
            
            response = client.post("/predict", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check types
            assert isinstance(data["prediction"], int)
            assert isinstance(data["probability"], float)
    
    def test_predict_endpoint_failure_prediction(self, mock_models):
        """Test /predict endpoint with failure prediction."""
        # Mock a failure prediction
        mock_models['model'].predict.return_value = [1]
        mock_models['model'].predict_proba.return_value = [[0.2, 0.8]]
        
        with patch('api.main.models', mock_models):
            client = TestClient(app)
            
            payload = {
                "air_temperature": 300.0,
                "process_temperature": 320.0,  # High temp
                "rotational_speed": 1500,
                "torque": 65.0,  # High torque
                "tool_wear": 250,  # High wear
                "type": "H"
            }
            
            response = client.post("/predict", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == 1
            assert data["probability"] == 0.8
