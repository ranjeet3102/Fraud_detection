from src.data_pipeline import load_data, clean_data, validate_data, preprocess_target
from src.feature_engineering import scale_features
from src.model_training import train_model
from src.evaluation import evaluate_model, shap_explain
from src.utils import get_logger
from src.preprocess import preprocess_data

logger = get_logger()

def run_pipeline():
    logger.info("Loading raw data...")
    df = load_data("data/raw/claims.csv")
    
    logger.info("Cleaning data...")
    df = clean_data(df)

    logger.info("Preprocessing raw features...")
    df = preprocess_data(df)

    logger.info("Validating schema...")
    df = validate_data(df)

    logger.info("Splitting X and y...")
    X, y = preprocess_target(df)

    logger.info("Scaling features...")
    X_scaled = scale_features(X)

    logger.info("Training model...")
    model = train_model(X_scaled, y)

    logger.info("Evaluating model...")
    evaluate_model(model, X_scaled, y)

    logger.info("Explaining model predictions...")
    shap_explain(model, X_scaled)

if __name__ == "__main__":
    run_pipeline()
