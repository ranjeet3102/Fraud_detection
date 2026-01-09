import mlflow
import mlflow.sklearn

def log_model(model, run_name="XGB_Fraud_Model", metrics=None):
    """Log a trained model and optional metrics to MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(model.get_params())

        mlflow.sklearn.log_model(model, artifact_path="model")

        if metrics:
            mlflow.log_metrics(metrics)

        print(f"Model logged under run: {mlflow.active_run().info.run_id}")
