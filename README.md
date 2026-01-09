
# Insurance Fraud Detection System

An end-to-end, production-ready machine learning system to detect fraudulent insurance claims using XGBoost, Streamlit for visualization, FastAPI for serving, and complete test automation. Built with industry best practices and modular architecture.

## Project Overview

Insurance fraud causes billions in losses globally every year. This project aims to proactively detect fraudulent claims using a supervised machine learning approach trained on policyholder data.


## Key Features

✅ Production-grade model using XGBoost  
✅ Real-time Streamlit dashboard for non-technical users  
✅ Modular, testable Python code with PyTest  
✅ Model & scaler serialization using `joblib`  
✅ Pipeline-driven data preprocessing and prediction  
✅ CI-friendly structure (Makefile, requirements.txt, .gitignore)  
✅ Easily deployable (via Docker or Streamlit Sharing)

---

## Project Structure

```

insurance-fraud-detection/
│
├── models/                 # Trained model (.pkl) and scaler
├── dashboards/             # Streamlit frontend
│   └── streamlit\_dashboard.py
├── data/                   # Sample datasets (not tracked by Git)
├── tests/                  # Unit tests (PyTest)
├── scripts/                # Training script, utilities
│   └── train\_model.py
├── config.yaml             # Configuration settings
├── Makefile                # Command-line automation
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Tooling and formatting config
└── README.md               # This file

````

---

## How It Works

### Data Flow

1. User uploads data or inputs values manually in the Streamlit dashboard.
2. Features are transformed using a pre-fitted `StandardScaler`.
3. Model predicts whether the policyholder is likely to commit fraud.
4. Output includes prediction and fraud probability.

---

## Input Features

| Feature               | Description                                 |
|----------------------|---------------------------------------------|
| `age`                | Age of policyholder                         |
| `policy_sales_channel` | Encoded channel used for sales              |
| `gender`             | 0 = Female, 1 = Male                         |
| `previously_insured`| 1 = Yes, 0 = No (previous insurance held?)  |
| `vehicle_age`        | Encoded: 0 = <1yr, 1 = 1–2yr, 2 = >2yr      |
| `vehicle_damage`     | 0 = No, 1 = Yes (vehicle was damaged?)      |

---

## Quickstart

### 1. Install Dependencies

make install

Or manually:

pip install -r requirements.txt

### 2. Train the Model

make train

> This saves the pipeline as `models/final_model.pkl`.

### 3. Run Streamlit Dashboard

make run

Or directly:

streamlit run dashboards/streamlit_dashboard.py


## Testing

Run all unit tests using:

make test

## Configuration

All environment-independent configs are managed via `config.yaml`:

```yaml
model_path: models/final_model.pkl
scaler_path: models/scaler.pkl
required_features:
  - age
  - policy_sales_channel
  - gender
  - previously_insured
  - vehicle_age
  - vehicle_damage
```

---

## Code Quality

This project uses:

* [`ruff`](https://github.com/astral-sh/ruff) for linting & formatting
* `Makefile` targets for formatting & checks

make lint      # Check code quality
make format    # Auto-format code


## Security & Privacy

* No personal identifiable information (PII) is stored or logged.
* Model & scaler files are saved securely and not exposed to public.
* Code is structured to support future integrations with authentication or RBAC systems.


## Future Improvements

* Deploy REST API using FastAPI + Docker
* Add SHAP explainability to the dashboard
* Train ensemble or deep learning models
* Integrate database (PostgreSQL or BigQuery)
* Schedule batch predictions using Airflow / Prefect
* CI/CD pipeline via GitHub Actions


## Author

**Sanjay** – Aspiring Data Scientist on a journey to build industry-grade ML pipelines.
*Project built with a focus on clarity, modularity, and real-world applicability.*


## License

MIT License. Free to use for educational or commercial purposes.

#Here is a Blog about this Project:
https://fraud-detection-end2end.hashnode.dev/building-a-production-grade-insurance-fraud-detection-system-with-streamlit-xgboost-and-mlops-slug-insurance-fraud-detection-mlops

#Linked-IN:
www.linkedin.com/in/sanjayram-data

