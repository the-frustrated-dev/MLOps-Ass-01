import pandas as pd
import joblib
import os

def test_preprocess_output():
    df = pd.read_csv("data/processed/cleaned_heart_disease.csv")
    assert not df.isnull().values.any()
    assert df['target'].nunique() == 2

def test_feature_output():
    data = joblib.load("data/features/train_test_data.pkl")
    assert "train" in data
    assert len(data["train"][0]) > 0

def test_model_exists():
    assert os.path.exists("models/model.joblib")