
from src.data.load_data import load_data
from src.features.preprocess import preprocess
from src.models.factory import create_model
from src.training.evaluate import evaluate_model

from src.utils.config_loader import load_cfg
from src.utils.mlflow_utils import *

# data_cfg = load_cfg("config/data.yaml")
models_cfg = load_cfg("config/models.yaml")
training_cfg = load_cfg("config/training.yaml")

X_train, X_test, y_train, y_test = load_data(training_cfg["data"]) # todo
X_train, X_test = preprocess(X_train, X_test)

for model_name, model_cfg in models_cfg["models"].items():
    print(f"Training: {model_name}")
    
    model = create_model(model_cfg=model_cfg)
    
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    start_run(cfg=training_cfg, run_name=model_name, params=model.get_params())
    log_metrics(metrics)
    log_model(model, model_name, X_test)
    end_run()