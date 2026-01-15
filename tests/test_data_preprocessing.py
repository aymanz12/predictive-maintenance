import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from src.data.preprocessing import clean_column_names, preprocess_and_split, load_config


class TestCleanColumnNames:
    """Tests for clean_column_names function."""
    
    def test_removes_temperature_units(self):
        """Test that [K] is removed from column names."""
        df = pd.DataFrame(columns=['Air temperature [K]', 'Process temperature [K]'])
        result = clean_column_names(df)
        
        assert 'air_temperature' in result.columns
        assert 'process_temperature' in result.columns
        assert 'Air temperature [K]' not in result.columns
    
    def test_removes_rpm_units(self):
        """Test that [rpm] is removed."""
        df = pd.DataFrame(columns=['Rotational speed [rpm]'])
        result = clean_column_names(df)
        
        assert 'rotational_speed' in result.columns
    
    def test_removes_torque_units(self):
        """Test that [Nm] is removed."""
        df = pd.DataFrame(columns=['Torque [Nm]'])
        result = clean_column_names(df)
        
        assert 'torque' in result.columns
    
    def test_removes_tool_wear_units(self):
        """Test that [min] is removed."""
        df = pd.DataFrame(columns=['Tool wear [min]'])
        result = clean_column_names(df)
        
        assert 'tool_wear' in result.columns
    
    def test_converts_to_lowercase(self):
        """Test that column names are converted to lowercase."""
        df = pd.DataFrame(columns=['Type', 'UDI', 'Product ID'])
        result = clean_column_names(df)
        
        assert 'type' in result.columns
        assert 'udi' in result.columns
        assert 'product_id' in result.columns
    
    def test_replaces_spaces_with_underscores(self):
        """Test that spaces are replaced with underscores."""
        df = pd.DataFrame(columns=['Machine failure', 'Product ID'])
        result = clean_column_names(df)
        
        assert 'machine_failure' in result.columns
        assert 'product_id' in result.columns
    
    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        df = pd.DataFrame(columns=[' Column Name ', 'Another  Column  '])
        result = clean_column_names(df)
        
        # Multiple spaces should be replaced with single underscore
        assert 'column_name' in result.columns
        assert 'another__column' in result.columns  # Two underscores for two spaces


class TestDataPreprocessing:
    """Tests for preprocessing functions."""
    
    @patch('src.data.preprocessing.load_config')
    @patch('src.data.preprocessing.pd.read_csv')
    @patch('src.data.preprocessing.train_test_split')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_preprocess_and_split_basic_flow(
        self, 
        mock_makedirs, 
        mock_exists, 
        mock_split, 
        mock_read_csv, 
        mock_load_config,
        sample_raw_data
    ):
        """Test the basic flow of preprocess_and_split."""
        # Setup mocks
        mock_load_config.return_value = {
            'data': {
                'raw_path': 'data/raw/test.csv',
                'processed_train_path': 'data/processed/train.csv',
                'processed_test_path': 'data/processed/test.csv'
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42
            }
        }
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_raw_data
        
        # Create split data
        train_df = sample_raw_data.iloc[:4]
        test_df = sample_raw_data.iloc[4:]
        mock_split.return_value = (train_df, test_df)
        
        # Run function
        with patch.object(pd.DataFrame, 'to_csv'):
            preprocess_and_split()
        
        # Verify config was loaded
        mock_load_config.assert_called_once()
        
        # Verify data was read
        mock_read_csv.assert_called_once()
    
    def test_load_config_file_not_found(self):
        """Test that load_config raises error when file not found."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    @patch('src.data.preprocessing.load_config')
    @patch('os.path.exists')
    def test_preprocess_raises_when_raw_not_found(self, mock_exists, mock_load_config):
        """Test that preprocess_and_split raises error when raw data not found."""
        mock_load_config.return_value = {
            'data': {
                'raw_path': 'data/raw/missing.csv',
                'processed_train_path': 'data/processed/train.csv',
                'processed_test_path': 'data/processed/test.csv'
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42
            }
        }
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            preprocess_and_split()
    
    def test_clean_column_names_preserves_data(self, sample_raw_data):
        """Test that cleaning column names preserves the data."""
        original_shape = sample_raw_data.shape
        result = clean_column_names(sample_raw_data)
        
        assert result.shape == original_shape
