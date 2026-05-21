"""
Model evaluation utilities.

Key metrics explained
---------------------
Accuracy   : (correct predictions) / (total predictions)
             Misleading on imbalanced data — a model that always
             predicts 'positive' gets ~55% accuracy but is useless.

Precision  : Of all reviews predicted as class X, how many actually are X?
             High precision → few false positives.

Recall     : Of all actual class X reviews, how many did we catch?
             High recall → few false negatives (missed detections).

F1-score   : Harmonic mean of Precision and Recall.
             Balances both — the go-to metric for imbalanced classes.

macro avg  : Unweighted average across all classes.
             Treats every class equally regardless of size.
             Best for imbalanced datasets where minority class matters.

weighted avg: Average weighted by class support (sample count).
             Dominated by majority classes.

Confusion Matrix
  Row   = actual class
  Column = predicted class
  Diagonal = correct predictions; off-diagonal = errors.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)


def evaluate(model, X_test, y_test) -> dict:
    """
    Run model.predict and compute a standard set of metrics.
    Returns a dict including 'y_pred' for downstream use.
    """
    y_pred = model.predict(X_test)
    return {
        'y_pred':           y_pred,
        'accuracy':         round(accuracy_score(y_test, y_pred), 4),
        'f1_macro':         round(f1_score(y_test, y_pred, average='macro'), 4),
        'f1_weighted':      round(f1_score(y_test, y_pred, average='weighted'), 4),
        'precision_macro':  round(precision_score(y_test, y_pred, average='macro'), 4),
        'recall_macro':     round(recall_score(y_test, y_pred, average='macro'), 4),
    }


def classification_report_df(y_test, y_pred, labels=None) -> pd.DataFrame:
    """Return sklearn's classification_report as a tidy DataFrame."""
    report = classification_report(y_test, y_pred,
                                   target_names=labels,
                                   output_dict=True)
    return pd.DataFrame(report).T.round(3)


def confusion_matrix_df(y_test, y_pred, labels) -> pd.DataFrame:
    """
    Return the confusion matrix as a labelled DataFrame.
    Rows = actual class, Columns = predicted class.
    """
    cm = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(
        cm,
        index=[f'Actual: {l}' for l in labels],
        columns=[f'Pred: {l}' for l in labels],
    )


def compare_models(results: dict) -> pd.DataFrame:
    """
    Summarise per-model metrics in one DataFrame sorted by f1_macro.

    Parameters
    ----------
    results : {model_name: metrics_dict}  (output of evaluate())
    """
    rows = {name: {k: v for k, v in m.items() if k != 'y_pred'}
            for name, m in results.items()}
    return pd.DataFrame(rows).T.sort_values('f1_macro', ascending=False)


def check_threshold(metrics: dict, thresholds: dict) -> tuple:
    """
    Check whether every metric meets its minimum threshold.

    How to set thresholds
    ---------------------
    Think about the cost of each error type in your business context:

    * Missing a NEGATIVE review (false negative) = customer churn risk
      → raise recall_macro threshold.
    * Flagging a POSITIVE review as negative (false positive) =
      wasted moderation effort → raise precision_macro threshold.
    * For a balanced safety bar, use f1_macro >= 0.80 as a starting point
      and adjust after reviewing the confusion matrix.

    Returns
    -------
    (passed: bool, failed_metric, actual_value, required_value)
    passed=True means all thresholds are satisfied.
    """
    for metric, required in thresholds.items():
        actual = metrics.get(metric)
        if actual is not None and actual < required:
            return False, metric, actual, required
    return True, None, None, None
