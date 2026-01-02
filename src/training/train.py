import mlflow

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import LearningCurveDisplay
from sklearn.calibration import CalibrationDisplay

import seaborn as sns
from matplotlib import pyplot as plt
# from src.data.load_data import load_data
# from src.features.preprocess import preprocess
# from src.models.factory import create_model
# from src.training.evaluate import evaluate_model

# from src.utils.config_loader import load_cfg
# from src.utils.mlflow_utils import *

# from src.models.registry import promote_best_model

# # data_cfg = load_cfg("config/data.yaml")
# models_cfg = load_cfg("config/models.yaml")
# training_cfg = load_cfg("config/training.yaml")

# X_train, X_test, y_train, y_test = load_data(training_cfg["data"]) # todo
# X_train, X_test = preprocess(X_train, X_test)

# for model_name, model_cfg in models_cfg["models"].items():
#     print(f"Training: {model_name}")
    
#     model = create_model(model_cfg=model_cfg)
    
#     model.fit(X_train, y_train)

#     metrics = evaluate_model(model, X_test, y_test)

#     start_run(cfg=training_cfg, run_name=model_name, params=model.get_params())
#     log_metrics(metrics)
#     log_model(model, model_name, X_test, metrics)
#     end_run()

# promote_best_model(training_cfg["experiment_name"], training_cfg["metric"])


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test, out_dir, report_name_prefix="report_"):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T # Exclude 'support' row
    cls = sns.heatmap(df_report, annot=True, cmap='crest')
    # cls.figure.savefig(f"{out_dir}/{report_name_prefix}_classification_report.png")
    mlflow.log_figure(cls.figure, f"{report_name_prefix}_classification_report.png")
    
    cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    # cm.im_.figure.savefig(f"{out_dir}/{report_name_prefix}_confusion_matrix.png")
    mlflow.log_figure(cm.im_.figure, f"{report_name_prefix}_confusion_matrix.png")

    pr = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    # pr.figure_.savefig(f"{out_dir}/{report_name_prefix}_precision_recall.png")
    mlflow.log_figure(pr.figure_, f"{report_name_prefix}_precision_recall.png")
    
    roc = RocCurveDisplay.from_estimator(model, X_test, y_test)
    # roc.figure_.savefig(f"{out_dir}/{report_name_prefix}_roc.png")
    mlflow.log_figure(roc.figure_, f"{report_name_prefix}_roc.png")
    
    cal = CalibrationDisplay.from_estimator(model, X_test, y_test)
    # cal.figure_.savefig(f"{out_dir}/{report_name_prefix}_calibration.png")
    mlflow.log_figure(cal.figure_, f"{report_name_prefix}_calibration.png")
    
    lc = LearningCurveDisplay.from_estimator(model, X_test, y_test, shuffle=True) # keep shuffle True, we have very few samples
    # lc.figure_.savefig(f"{out_dir}/{report_name_prefix}_learning_curve.png")
    mlflow.log_figure(lc.figure_, f"{report_name_prefix}_learning_curve.png")

    plt.show()

    results = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average="weighted", zero_division=0),
        'recall':    recall_score(y_test, y_pred, average="weighted"),
        'f1_weighted':  f1_score(y_test, y_pred, average="weighted"),
    }
    return results