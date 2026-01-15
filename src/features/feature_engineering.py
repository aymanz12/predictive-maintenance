import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging specific to this module
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Encapsulates feature engineering logic for the AI4I dataset.
    Compatible with Scikit-Learn Pipelines.
    """
    
    def __init__(self):
        # Mapping for the 'type' column (Ordinal Encoding)
        self.type_mapping = {'L': 0, 'M': 1, 'H': 2}
        # Default value for unknown types (1 = Medium)
        self.default_type_value = 1
        # Store feature names to ensure consistent ordering (set during transform)
        self.feature_names_ = None 
    
    def fit(self, X, y=None): 
        """
        Stateless transformer, so fit does nothing.
        """
        return self

    def transform(self, X):
        """
        Apply physics-based feature engineering and encoding.
        """
        # 1. Validation
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(X)}")

        # Work on a copy to avoid side effects
        df = X.copy()
        
        # 2. Physics-Based Feature Construction
        # We check for column existence to make the transformer robust to partial inputs
        
        # Temp Difference: Critical for Heat Dissipation Failure (HDF)
        if 'process_temperature' in df.columns and 'air_temperature' in df.columns:
            df['temp_difference'] = df['process_temperature'] - df['air_temperature']
        else:
            logger.warning("Missing temperature columns; 'temp_difference' not created.")

        # Power Factor: Critical for Power Failure (PWF)
        if 'torque' in df.columns and 'rotational_speed' in df.columns:
            df['power_factor'] = df['torque'] * df['rotational_speed']
        else:
            logger.warning("Missing torque/speed columns; 'power_factor' not created.")

        # Strain Wear Product: Critical for Overstrain Failure (OSF)
        if 'torque' in df.columns and 'tool_wear' in df.columns:
            df['strain_wear_product'] = df['torque'] * df['tool_wear']
        else:
            logger.warning("Missing torque/wear columns; 'strain_wear_product' not created.")
        
        # 3. Categorical Encoding
        if 'type' in df.columns:
            # Map values, fill missing/unknowns with default (Medium), and force integer type
            df['type'] = df['type'].map(self.type_mapping).fillna(self.default_type_value).astype(int)
        
        # 4. Enforce Consistent Column Ordering
        # Define the standard order for features
        standard_order = [
            'type', 'air_temperature', 'process_temperature', 
            'rotational_speed', 'torque', 'tool_wear',
            'temp_difference', 'power_factor', 'strain_wear_product'
        ]
        
        # Reorder columns to match standard order (only include columns that exist)
        df = df[[col for col in standard_order if col in df.columns]]
        
        # Store feature names on first transform (typically during fit_transform in training)
        if self.feature_names_ is None:
            self.feature_names_ = df.columns.tolist()
            logger.info(f"Feature names set: {self.feature_names_}")
            
        return df

if __name__ == "__main__":
    # Quick sanity check when running this file directly
    try:
        logging.basicConfig(level=logging.INFO)
        
        # Mock data
        df_test = pd.DataFrame({
            'air_temperature': [300, 298],
            'process_temperature': [310, 309],
            'rotational_speed': [1500, 1400],
            'torque': [40, 50],
            'tool_wear': [10, 200],
            'type': ['L', 'H']
        })
        
        engineer = FeatureEngineer()
        df_transformed = engineer.transform(df_test)
        
        print("\n✅ Transformation Successful. Output sample:")
        print(df_transformed[['type', 'power_factor', 'temp_difference']].head())
        
    except Exception as e:
        print(f"❌ Transformation failed: {e}")