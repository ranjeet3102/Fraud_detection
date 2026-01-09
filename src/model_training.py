import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import joblib
import os

def train_model(X, y, model_path="../models/final_model.pkl"):
    param_grid = {
        'n_estimators':[100, 150],
        'max_depth' : [3,5],
        'learning_rate' : [0.01, 0.1],
    }
    
    skf = StratifiedKFold(n_split = 5, shuffle = True, random_state=42)
    
    clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    
    search = RandomizedSearchCV(
        clf,
        param_distributions=param_grid,
        cv = skf,
        scoring='roc-auc',
        n_jobs = -1,
        verbose=1,
        random_state=42
    )
    
    search.fit(X, y)
    best_model = search.best_estimator_
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    return best_model