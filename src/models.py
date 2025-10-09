from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def get_models():
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
    }
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(n_estimators=200)
    except Exception:
        pass
    return models

def train_and_eval(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        probs = None
        try:
            probs = m.predict_proba(X_test)[:,1]
        except Exception:
            probs = None
        results[name] = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'f1': float(f1_score(y_test, preds, zero_division=0)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, probs)) if probs is not None else None,
            'preds': preds,
            'model': m
        }
    return results
