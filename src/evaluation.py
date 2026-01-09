from sklearn.metrics import classification_report, roc_auc_score, f1_score
import shap
import joblib
import matplotlib.pyplot as plt
import os

def evaluate_model(model, X, y):
    preds = model.predict(X)
    proba = model = model.predict_proba(X)[:, 1]
    
    print(f"ROC AUC Score: {roc_auc_score(y, proba):.4f}")
    print(f"F1 Score: {f1_score(y, preds):.4f}")
    print("Classification Report:\n", classification_report(y, preds))
    
def shap_explain(model, X, output_dir="reports/"):
    os.makedirs(output_dir, exist_ok = True)
    
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()