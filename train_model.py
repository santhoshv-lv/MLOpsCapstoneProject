# src/model/train_model.py
import os, sys, joblib, warnings
import pandas as pd
import mlflow
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.logger import setup_logger
logger = setup_logger(__name__)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES = ['R_Score', 'F_Score', 'M_Score']
TARGET = 'is_high_value'
MODEL_PATH = "src/model/final_model.pkl"
N_TRIALS = 20

def objective(trial, X_train, X_test, y_train, y_test):
    C_param = trial.suggest_loguniform('C', 1e-5, 1e2)
    penalty_param = trial.suggest_categorical('penalty', ['l2', 'none'])
    # solver 'lbfgs' ignores 'none' in older sklearn; use 'saga' to support none if needed.
    solver = 'lbfgs' if penalty_param == 'l2' else 'saga'
    model = LogisticRegression(C=C_param, penalty=None if penalty_param == 'none' else penalty_param,
                               solver=solver, max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("f1_score", f1)
        mlflow.set_tag("trial_id", trial.number)
    return f1

def train_and_save_model():
    logger.info("Starting training pipeline")
    data_path = "data/processed/rfm_features.csv"
    if not os.path.exists(data_path):
        logger.error("RFM file not found. Run process_rfm.py first.")
        raise FileNotFoundError
    data = pd.read_csv(data_path)
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    mlflow.set_experiment("Retail-Customer-Segmentation")
    with mlflow.start_run(run_name="Optuna_Hyperparameter_Search") as parent_run:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=N_TRIALS)
        best_params = study.best_params
        logger.info(f"Best trial value: {study.best_value}, params: {best_params}")

        final_model = LogisticRegression(
            **{k: v for k, v in best_params.items() if k != 'penalty'},
            penalty=None if best_params.get('penalty') == 'none' else best_params.get('penalty'),
            solver='saga' if best_params.get('penalty') == 'none' else 'lbfgs',
            max_iter=1000, random_state=42, class_weight='balanced'
        )
        final_model.fit(X, y)

        y_pred_test = final_model.predict(X_test)
        final_f1 = f1_score(y_test, y_pred_test)
        final_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])

        mlflow.log_metric("final_f1_score", final_f1)
        mlflow.log_metric("final_roc_auc", final_auc)
        mlflow.set_tag("model_type", "LogisticRegression")

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        # register model (if you run mlflow server with registry); here we just log
        try:
            mlflow.sklearn.log_model(sk_model=final_model, artifact_path="model",
                                     registered_model_name="High_Value_Classifier")
        except Exception:
            logger.info("Model registered step skipped (no MLflow registry connected).")

    logger.info("Training pipeline completed.")

if __name__ == "__main__":
    train_and_save_model()
