import os
from zipfile import ZipFile

import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# def load_data(cfg):
#     df = pd.read_csv(cfg["path"], names=cfg["cols"])
#     df.replace("?", np.nan, inplace=True)
#     df = df.astype(float)
#     df.fillna(df.median(), inplace=True)
#     X = df.drop(cfg["target_col"], axis=1)
#     y = df[cfg["target_col"]]

#     return train_test_split(
#         X, y,
#         test_size=cfg["test_size"],
#         random_state=cfg["random_state"],
#         stratify=y
#     )

def fetch_data(config, chunk_size=1024 * 32):
    src_url = config["data"]["url"]
    dest = os.path.join(config["data"]["target_dir"], config["data"]["dataset_name"])
    os.makedirs(config["data"]["target_dir"], exist_ok=True)
    if os.path.exists(dest):
        print("Dataset already present. Skipping Download!")
        return
    with requests.get(src_url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    with ZipFile(dest) as dataset:
        dataset.extract(config["load"]["target_file"], path=config["data"]["target_dir"])

def load_data(config):
    dataset_file = os.path.join(config["data"]["target_dir"], config["load"]["target_file"])
    df = pd.read_csv(dataset_file, names=config["load"]["column_names"])
    df.replace("?", np.nan, inplace=True)
    df = df.astype(float)
    df.fillna(df.median(), inplace=True)
    df[config["load"]["target_col"]] = df[config["load"]["target_col"]].apply(lambda x: 1 if x > 0 else 0)
    X = df.drop(config["load"]["target_col"], axis=1)
    y = df[config["load"]["target_col"]]

    return train_test_split(
        X, y,
        test_size=config["load"]["test_size"],
        random_state=config["load"]["random_state"],
        stratify=y
    )