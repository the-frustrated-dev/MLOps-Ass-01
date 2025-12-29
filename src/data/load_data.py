import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(cfg):
    df = pd.read_csv(cfg["path"], names=cfg["cols"])
    df.replace("?", np.nan, inplace=True)
    df = df.astype(float)
    df.fillna(df.median(), inplace=True)
    X = df.drop(cfg["target_col"], axis=1)
    y = df[cfg["target_col"]]

    return train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y
    )