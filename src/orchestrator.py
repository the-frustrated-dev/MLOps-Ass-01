import os
import tempfile
from collections import defaultdict

import yaml
import mlflow
import pandas as pd

from data.data_utils import fetch_data, load_data
from features.preprocess import preprocess
from models.factory import create_model
from training.train import train_model, evaluate_model
from models.registry import promote_model

if os.getenv("CI", "false").lower() == "false":
    from dotenv import load_dotenv
    load_dotenv()

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


## 1. Read Configurations ##
with open("config/data.yaml") as f:
    data_cfg = yaml.safe_load(f)
with open("config/models.yaml") as f:
    models_cfg = yaml.safe_load(f)

## 2. Fetch The Dataset ##
fetch_data(data_cfg) # downloading handled

## 3. Load & Split The Dataset ##
X_train, X_test, y_train, y_test = load_data(data_cfg)

## 4. Preprocess the data ##
X_train, X_test = preprocess(X_train, X_test)

## 5. Train & Evaluate Models ##
track_best_models = defaultdict(list)
for experiment_name, model_info in models_cfg["experiments"].items():
    experiment = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        model = create_model(model_info["class"], params=model_info.get("params"))
        
        train_model(model, X_train, y_train.to_numpy()) 
        
        with tempfile.TemporaryDirectory() as temp:
            metrics = evaluate_model(model, X_test, y_test, temp, report_name_prefix=experiment_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            # mlflow.log_artifacts(temp) # logged figures in evaluate_model()

        example_col = data_cfg["load"]["column_names"][::].remove(data_cfg["load"]["target_col"])
        
        model_info = mlflow.sklearn.log_model(
            model, 
            artifact_path=experiment_name, # maybe give a better name 
            input_example=pd.DataFrame(X_test, columns=example_col).iloc[:10],
            # tags=metrics, # tag the model with metrics
        )

        # model_uri = f"runs:/{run.info.run_id}/model"

        track_best_models[experiment_name].append((model_info, metrics, model_info.artifact_path))

# Find Best Model of Current Runs
_, (mi, met, uri) = max(((exp, model_info_list[0]) for exp, model_info_list in track_best_models.items()), key=lambda x: x[1][1]['f1_weighted']) # based on f1-score

## 6. Promote Models to  ModelRegistry if Better
promote_model(mi, met, uri)