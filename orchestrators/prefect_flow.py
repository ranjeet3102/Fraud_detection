from prefect import flow, task
from src.main import run_pipeline

@flow(name="fraud-detection-pipeline")
def prefect_fraud_flow():
    run_pipeline()

if __name__ == "__main__":
    prefect_fraud_flow()
