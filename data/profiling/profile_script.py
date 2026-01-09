import pandas as pd
from ydata_profiling import ProfileReport
import os
from pathlib import Path

df = pd.read_csv("E:/PROJECTS/insurance_fraud_detection/data/raw/train.csv")
profile_path = os.path.abspath("E:/PROJECTS/insurance_fraud_detection/data/profiling/ydata_profiling_report.html")

profile = ProfileReport(df, title="Insurance Sataset Profiling", explorative=True)
Path(os.path.dirname(profile_path)).mkdir(parents=True, exist_ok=True)

profile.to_file(profile_path)

print("Profiling completeAd")