import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(
f"""
Accuracy:  {accuracy_score(y_test, y_pred):.4f}
Precision: {precision_score(y_test, y_pred, average="weighted", zero_division=0):.4f}
Recall:    {recall_score(y_test, y_pred, average="weighted"):.4f}
F1-Score:  {f1_score(y_test, y_pred, average="weighted"):.4f}
{f"ROC-AUC:   {roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")}" if hasattr(model, "predict_proba") else ""}
"""
    )
    print(classification_report(y_test, y_pred, zero_division=0))

    if hasattr(model, "predict_proba"):
        classes = np.unique(y_test)
        y_val_bin = label_binarize(y_test, classes=classes)
        y_proba = model.predict_proba(X_test)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class: {cls} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [1, 0], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi Class ROC Curves (One vs Rest)")
    plt.legend()
    # plt.savefig() # TODO
    # plt.show()
    results = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average="weighted", zero_division=0),
        'Recall':    recall_score(y_test, y_pred, average="weighted"),
        'F1-Score':  f1_score(y_test, y_pred, average="weighted"),
    }
    if hasattr(model, "predict_proba"):
        results["ROC-AUC"] = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
    
    return results