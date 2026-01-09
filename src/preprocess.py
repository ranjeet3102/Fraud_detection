import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    df = df[df["gender"].isin(["Male", "Female"])]
    df = df[df["vehicle_age"].isin(["< 1 Year", "1-2 Year", "> 2 Years"])]
    df = df[df["vehicle_damage"].isin(["Yes", "No"])]

    label_encoders = {}
    for col in ["gender", "vehicle_age", "vehicle_damage"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df = df.rename(columns={"response": "is_fraud"})

    df = df[df["driving_license"].isin([0, 1])]
    df = df[df["previously_insured"].isin([0, 1])]
    df = df[df["is_fraud"].isin([0, 1])]

    df = df[(df["age"] >= 0) & (df["age"] < 120)]
    df = df[(df["annual_premium"] > 1000) & (df["annual_premium"] < 100000)]
    df = df[(df["vintage"] > 0) & (df["vintage"] < 500)]

    df["region_code"] = df["region_code"].astype("Int64")
    df["policy_sales_channel"] = df["policy_sales_channel"].astype("Int64")
    df["annual_premium"] = df["annual_premium"].astype("Int64")

    return df
