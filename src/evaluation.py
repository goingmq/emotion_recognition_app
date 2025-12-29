import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def evaluate_and_store_results(model, X_test, y_test, label_encoder, results_list, class_reports, model_name):
    """
    Evaluate a model and store:
    - overall metrics (accuracy / precision / recall / f1 / roc-auc)
    - per-class F1 (for comparison graph)
    """

    # predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    rocauc = roc_auc_score(y_test, y_pred_prob, average="weighted", multi_class="ovr")

    # store results
    results_list.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": rocauc
    })

    # per-class report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    class_reports[model_name] = report_dict
