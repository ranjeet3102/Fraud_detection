import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from sklearn.preprocessing import LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.rename(columns={'Response':"is_fraud"})
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    label_enc_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
    for col in label_enc_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    schema = DataFrameSchema({
        "Gender": Column(int, Check.isin([0, 1])),
        "Age": Column(int, Check.ge(18)),
        "Previously_Insured": Column(int, Check.isin([0, 1])),
        "Vehicle_Age": Column(int, Check.isin([0, 1, 2])),  # after label encoding
        "Vehicle_Damage": Column(int, Check.isin([0, 1])),
        "Policy_Sales_Channel": Column(float),  # might still be float at this point
        "is_fraud": Column(int, Check.isin([0, 1]))
    })
    return schema.validate(df)

def preprocess_target(df: pd.DataFrame, target_column="is_fraud"):
    y = df[target_column]
    X = df.drop(column=[target_column])
    return X,y