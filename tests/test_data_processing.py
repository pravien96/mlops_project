import pytest
import pandas as pd
import numpy as np
from src.data.data_processing import DataProcessor

def test_data_processor_initialization():
    processor = DataProcessor()
    assert processor.scaler is not None

def test_load_data():
    processor = DataProcessor()
    df = processor.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'target' in df.columns
    assert len(df.columns) == 9  # 8 features + 1 target

def test_preprocess_data():
    processor = DataProcessor()
    df = processor.load_data()
    
    X_train, X_test, y_train, y_test = processor.preprocess_data(df)
    
    assert len(X_train) > len(X_test)  # Training set should be larger
    assert X_train.shape[1] == 8  # 8 features
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
