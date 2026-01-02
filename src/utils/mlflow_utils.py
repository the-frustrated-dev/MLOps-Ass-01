import os
import mlflow
import time

if os.getenv("CI", "false").lower() == "false":
    from dotenv import load_dotenv
    load_dotenv()

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def start_run(cfg, run_name, params):
    mlflow.set_experiment(cfg["experiment_name"])
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(params=params)

def log_metrics(metrics):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def log_model(model, model_name, X_test, metrics):
    # mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(sk_model=model, name=model_name, input_example=X_test, tags=metrics)

def end_run():
    mlflow.end_run()