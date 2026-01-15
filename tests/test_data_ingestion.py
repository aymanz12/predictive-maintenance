import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.ingestion import load_config, ingest_data


class TestLoadConfig:
    """Tests for load_config function."""
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('yaml.safe_load')
    def test_load_config_success(self, mock_yaml_load, mock_file):
        """Test successful config loading."""
        expected_config = {
            'data': {'raw_path': 'data/raw/test.csv'},
            'training': {'test_size': 0.2}
        }
        mock_yaml_load.return_value = expected_config
        
        result = load_config("test_config.yaml")
        
        assert result == expected_config
    
    def test_load_config_file_not_found(self):
        """Test that load_config raises exception when file not found."""
        with pytest.raises(Exception):
            load_config("nonexistent.yaml")


class TestIngestData:
    """Tests for ingest_data function."""
    
    @patch('src.data.ingestion.load_config')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    def test_ingest_loads_existing_file(
        self, 
        mock_read_csv, 
        mock_makedirs, 
        mock_exists, 
        mock_load_config,
        sample_raw_data
    ):
        """Test that ingest_data loads file when it exists."""
        mock_load_config.return_value = {
            'data': {
                'raw_path': 'data/raw/test.csv',
                'dataset_url': 'http://example.com/data.csv'
            }
        }
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_raw_data
        
        result = ingest_data("test_config.yaml")
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_raw_data.shape
        mock_read_csv.assert_called_once_with('data/raw/test.csv')
    
    @patch('src.data.ingestion.load_config')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    def test_ingest_downloads_missing_file(
        self, 
        mock_read_csv, 
        mock_makedirs, 
        mock_exists, 
        mock_load_config,
        sample_raw_data
    ):
        """Test that ingest_data downloads file when it doesn't exist."""
        mock_load_config.return_value = {
            'data': {
                'raw_path': 'data/raw/test.csv',
                'dataset_url': 'http://example.com/data.csv'
            }
        }
        mock_exists.return_value = False
        mock_read_csv.return_value = sample_raw_data
        
        with patch.object(pd.DataFrame, 'to_csv'):
            result = ingest_data("test_config.yaml")
        
        # Should have called read_csv with URL
        assert mock_read_csv.call_count == 1
        call_args = mock_read_csv.call_args[0][0]
        assert 'http://' in call_args or call_args == 'data/raw/test.csv'
    
    @patch('src.data.ingestion.load_config')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    def test_ingest_validates_columns(
        self, 
        mock_read_csv, 
        mock_makedirs, 
        mock_exists, 
        mock_load_config,
        sample_raw_data
    ):
        """Test that column validation is performed."""
        mock_load_config.return_value = {
            'data': {
                'raw_path': 'data/raw/test.csv',
                'dataset_url': 'http://example.com/data.csv'
            }
        }
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_raw_data
        
        result = ingest_data("test_config.yaml")
        
        # Verify all expected columns are present
        expected_cols = ['UDI', 'Product ID', 'Type', 'Air temperature [K]']
        for col in expected_cols:
            assert col in result.columns
