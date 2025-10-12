# fairness_metrics.py
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import pandas as pd

def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def compute_bias(models, X_test, y_test, protected_attributes):
    bias_summary = []
    X_test_df = pd.DataFrame(X_test, columns=X_test.columns)
    y_test_series = pd.Series(y_test)
    
    for attr in protected_attributes:
        sensitive_test = X_test_df[attr]
        # Using RandomForest as reference model for bias calculation
        y_pred = models["RandomForest"].predict(X_test)
        
        dp_diff = demographic_parity_difference(y_test_series, y_pred, sensitive_features=sensitive_test)
        eo_diff = equalized_odds_difference(y_test_series, y_pred, sensitive_features=sensitive_test)
        
        dp_percent = round(dp_diff*100,2)
        eo_percent = round(eo_diff*100,2)
        bias_score = dp_percent + eo_percent
        
        # Detailed explanation for each attribute
        explanation = ""
        if attr == "age":
            explanation = f"Older people (30–50 yrs) predicted high income {dp_percent}% of the time; younger (18–30 yrs) rarely. Model accuracy {eo_percent}% for older, very low for younger."
        elif attr == "education":
            explanation = f"Higher-educated people predicted high income {dp_percent}% of the time; lower-educated less. Accuracy gap EO%={eo_percent}%."
        elif attr == "race":
            explanation = f"One racial group predicted high income {dp_percent}% more; accuracy gap EO%={eo_percent}%."
        elif attr == "sex":
            explanation = f"One gender predicted high income {dp_percent}% more; accuracy gap EO%={eo_percent}%."
        elif attr == "marital_status":
            explanation = f"Certain marital status predicted high income {dp_percent}% more; accuracy gap EO%={eo_percent}%."
        elif attr == "occupation":
            explanation = f"Certain occupations predicted high income {dp_percent}% more; accuracy gap EO%={eo_percent}%."

        bias_summary.append({
            "Attribute": attr,
            "DP%": dp_percent,
            "EO%": eo_percent,
            "Bias Score": bias_score,
            "Explanation": explanation
        })
    return pd.DataFrame(bias_summary)
