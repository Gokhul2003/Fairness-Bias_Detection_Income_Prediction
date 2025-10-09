from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    true_positive_rate
)
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def fairness_summary(y_true, y_pred, sensitive_feats):
    # Compute base metrics safely
    mf = MetricFrame(metrics={
        'selection_rate': selection_rate,
        'fpr': false_positive_rate,
        'tpr': true_positive_rate
    }, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feats)

    # Handle fairness metrics separately to avoid missing argument errors
    dpd = demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feats
    )
    eod = equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feats
    )

    overall = {
        'accuracy': float((y_true == y_pred).mean()),
        'demographic_parity_difference': float(dpd),
        'equalized_odds_difference': float(eod)
    }

    by_group = mf.by_group.to_dict()
    return {'overall': overall, 'by_group': by_group}


def confusion_matrices(y_true, y_pred, sensitive_feats):
    groups = np.unique(sensitive_feats)
    out = {}
    for g in groups:
        mask = (sensitive_feats == g)
        cm = confusion_matrix(y_true[mask], y_pred[mask])
        out[str(g)] = cm.tolist()
    return out
