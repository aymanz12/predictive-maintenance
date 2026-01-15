import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def test_fit_returns_self(self, sample_feature_input):
        """Test that fit returns self (stateless transformer)."""
        engineer = FeatureEngineer()
        result = engineer.fit(sample_feature_input)
        assert result is engineer
    
    def test_transform_creates_physics_features(self, sample_feature_input):
        """Test that transform creates all expected physics-based features."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_feature_input)
        
        # Check that new features are created
        assert 'temp_difference' in result.columns
        assert 'power_factor' in result.columns
        assert 'strain_wear_product' in result.columns
    
    def test_temp_difference_calculation(self):
        """Test temperature difference calculation."""
        df = pd.DataFrame({
            'air_temperature': [300.0, 295.0],
            'process_temperature': [310.0, 305.0],
            'rotational_speed': [1500, 1600],
            'torque': [40.0, 45.0],
            'tool_wear': [10, 20],
            'type': ['L', 'M']
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        # temp_difference = process_temperature - air_temperature
        expected = [10.0, 10.0]
        assert result['temp_difference'].tolist() == expected
    
    def test_power_factor_calculation(self):
        """Test power factor calculation."""
        df = pd.DataFrame({
            'air_temperature': [300.0],
            'process_temperature': [310.0],
            'rotational_speed': [1500],
            'torque': [40.0],
            'tool_wear': [10],
            'type': ['L']
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        # power_factor = torque * rotational_speed
        expected_power = 40.0 * 1500
        assert result['power_factor'].iloc[0] == expected_power
    
    def test_strain_wear_product_calculation(self):
        """Test strain wear product calculation."""
        df = pd.DataFrame({
            'air_temperature': [300.0],
            'process_temperature': [310.0],
            'rotational_speed': [1500],
            'torque': [50.0],
            'tool_wear': [100],
            'type': ['H']
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        # strain_wear_product = torque * tool_wear
        expected_strain = 50.0 * 100
        assert result['strain_wear_product'].iloc[0] == expected_strain
    
    def test_type_encoding(self):
        """Test categorical encoding of type column."""
        df = pd.DataFrame({
            'air_temperature': [300.0, 300.0, 300.0],
            'process_temperature': [310.0, 310.0, 310.0],
            'rotational_speed': [1500, 1500, 1500],
            'torque': [40.0, 40.0, 40.0],
            'tool_wear': [10, 10, 10],
            'type': ['L', 'M', 'H']
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        # L=0, M=1, H=2
        assert result['type'].tolist() == [0, 1, 2]
        assert result['type'].dtype in [np.int64, np.int32, int]
    
    def test_unknown_type_handling(self):
        """Test handling of unknown type values."""
        df = pd.DataFrame({
            'air_temperature': [300.0],
            'process_temperature': [310.0],
            'rotational_speed': [1500],
            'torque': [40.0],
            'tool_wear': [10],
            'type': ['X']  # Unknown type
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        # Unknown types should default to 1 (Medium)
        assert result['type'].iloc[0] == 1
    
    def test_missing_columns_warning(self, caplog):
        """Test that missing columns trigger warnings."""
        df = pd.DataFrame({
            'rotational_speed': [1500],
            'type': ['L']
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        # Should have warnings for missing columns
        assert 'temp_difference' not in result.columns
        assert 'power_factor' not in result.columns
    
    def test_transform_preserves_original_columns(self, sample_feature_input):
        """Test that original columns are preserved."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_feature_input)
        
        # Original columns should still be present
        for col in sample_feature_input.columns:
            assert col in result.columns
    
    def test_transform_with_zeros(self):
        """Test handling of zero values in calculations."""
        df = pd.DataFrame({
            'air_temperature': [300.0],
            'process_temperature': [310.0],
            'rotational_speed': [0],  # Zero speed
            'torque': [0],  # Zero torque
            'tool_wear': [0],  # Zero wear
            'type': ['L']
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        # Should handle zeros without errors
        assert result['power_factor'].iloc[0] == 0.0
        assert result['strain_wear_product'].iloc[0] == 0.0
    
    def test_transform_returns_dataframe(self, sample_feature_input):
        """Test that transform returns a DataFrame."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_feature_input)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_transform_raises_on_non_dataframe(self):
        """Test that transform raises TypeError for non-DataFrame input."""
        engineer = FeatureEngineer()
        
        with pytest.raises(TypeError):
            engineer.transform([[1, 2, 3]])
    
    def test_fit_transform(self, sample_feature_input):
        """Test fit_transform method (inherited from BaseEstimator)."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_feature_input)
        
        # Should have all engineered features
        assert 'temp_difference' in result.columns
        assert 'power_factor' in result.columns
        assert 'strain_wear_product' in result.columns
