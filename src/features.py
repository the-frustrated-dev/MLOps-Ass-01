import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def run_features():
    os.makedirs("data/features", exist_ok=True)
    df = pd.read_csv("data/processed/cleaned_heart_disease.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    data_bundle = {
        "train": (X_train_scaled, y_train),
        "test": (X_test_scaled, y_test),
        "scaler": scaler
    }
    joblib.dump(data_bundle, "data/features/train_test_data.pkl")

if __name__ == "__main__":
    run_features()