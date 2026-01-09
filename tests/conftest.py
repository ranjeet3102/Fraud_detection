import pytest
import pandas as pd
from src.data_pipeline import clean_data, preprocess_target
from src.feature_engineering import scale_features
from src.model_training import train_model

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "gender": [1, 0],
        "age": [25, 45],
        "previously_insured": [0, 1],
        "vehicle_age": [1, 2],
        "vehicle_damage": [1, 0],
        "policy_sales_channel": [26, 152],
        "is_fraud": [0, 1]
    })

@pytest.fixture
def cleaned_df(sample_df):
    return clean_data(sample_df)

@pytest.fixture
def split_data(cleaned_df):
    X, y = preprocess_target(cleaned_df)
    return X, y

@pytest.fixture
def scaled_data(split_data):
    X, _ = split_data
    return scale_features(X)
