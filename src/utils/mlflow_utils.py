import mlflow

def start_run(cfg, run_name, params):
    mlflow.set_experiment(cfg["experiment_name"])
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(params=params)

def log_metrics(metrics):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def log_model(model, model_name, X_test):
    # mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(model, name=model_name, input_example=X_test)

def end_run():
    mlflow.end_run()