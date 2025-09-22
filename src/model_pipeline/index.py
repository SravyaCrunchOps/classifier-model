import joblib
import json
import os
from model_pipeline.model_training import model_training
from model_pipeline.model_evaluation import evaluation
from model_pipeline.model_validation import validation
from model_pipeline.model_tuning import hyperparameter_tuning

MODELS_PATH = "../models"
LOGS_PATH = "../logs"
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

def model_pipeline(X_train, X_test, y_train, y_test):
    # model training
    classifier_model = model_training(X_train, y_train)
    y_pred = classifier_model.predict(X_test)

    # ṃodel evaluation
    accuracy_metric = evaluation(y_test, y_pred)

    # model validation
    accuracy_metric_cv, cv = validation(classifier_model)

    # hyperparameter tuning
    best_model = hyperparameter_tuning(X_train, y_train, cv)
    y_pred_ht = best_model.predict(X_test)

    accuracy_metric_ht = evaluation(y_test, y_pred_ht)

    # Convert all parameter values to strings for JSON serialization
    params_serializable = {k: str(v) for k, v in best_model.get_params().items()}
    metric_log = {
        "accuracy": accuracy_metric,
        "accuracy_cv": accuracy_metric_cv,
        "accuracy_tuning": accuracy_metric_ht,
        "parameters": params_serializable
    }
    with open(f"{LOGS_PATH}/metric.json", "w") as f:
        json.dump(metric_log, f, indent=4)  # pretty print

    # model deployment
    joblib.dump(best_model, f'{MODELS_PATH}/classifier_model.pkl')

