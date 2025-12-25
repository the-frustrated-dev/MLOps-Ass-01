import joblib
import yaml
import mlflow
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def run_train():
    with open("config/params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    data = joblib.load("data/features/train_test_data.pkl")
    X_train, y_train = data["train"]
    
    # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    with mlflow.start_run():
        if params["model_choice"] == "random_forest":
            model = RandomForestClassifier(**params["hyperparams"]["rf"])
        else:
            model = LogisticRegression(**params["hyperparams"]["lr"])
            
        model.fit(X_train, y_train)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")
        
        # mlflow.log_params(params["hyperparams"])
        # mlflow.sklearn.log_model(model, "model", registered_model_name="HeartDiseaseModel")

if __name__ == "__main__":
    run_train()