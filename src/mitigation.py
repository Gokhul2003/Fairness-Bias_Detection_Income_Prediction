from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression

def run_exponentiated_gradient(X_train, y_train, sensitive_train, constraint='demographic_parity', eps=0.01):
    if constraint == 'equalized_odds':
        cons = EqualizedOdds()
    else:
        cons = DemographicParity()
    base = LogisticRegression(max_iter=1000)
    mitigator = ExponentiatedGradient(base, cons, eps=eps)
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    return mitigator
