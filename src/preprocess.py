import os
import zipfile

import yaml
import numpy as np
import pandas as pd

def preprocess():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    zip_path = os.path.join(config['data']['target_dir'], config['data']['dataset_name'])
    extract_dir = "data/raw"
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Load one dataset (will explore after EDA) - currently enabling the pipeline
    df = pd.read_csv(f"{extract_dir}/processed.cleveland.data", header=None)
    # Define columns based on UCI metadata
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'] # (num) column is the Target
    
    # Basic preprocessing - these will change after EDA
    df = df.replace('?', pd.NA).dropna()
    # df['target'] = (df['target'] > 0).astype(int)
    
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(f"{processed_dir}/cleaned_heart_disease.csv", index=False)

if __name__ == "__main__":
    preprocess()