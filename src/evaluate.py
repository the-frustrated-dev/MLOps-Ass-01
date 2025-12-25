import joblib
import json
import os
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def run_evaluate():
    # Load the data bundle created in features.py
    data = joblib.load("data/features/train_test_data.pkl")
    X_test, y_test = data["test"]
    
    # Load the trained model
    model = joblib.load("models/model.joblib")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Create metrics dictionary
    metrics = {
        "accuracy": float(acc),
        "f1_score": float(f1)
    }
    
    # Save metrics for DVC and MLflow
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Evaluation complete. Accuracy: {acc:.4f}")

    df_cm = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
    df_cm.to_csv("confusion_matrix.csv", index=False)

if __name__ == "__main__":
    run_evaluate()