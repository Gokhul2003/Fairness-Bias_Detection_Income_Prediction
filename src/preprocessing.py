import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

def detect_target(df):
    for c in ['income','Income','target']:
        if c in df.columns:
            return c
    return df.columns[-1]

def preprocess(df, target_col=None, protected_attrs=['gender','sex','race'], test_size=0.2, random_state=42):
    df = df.copy()
    if target_col is None:
        target_col = detect_target(df)
    df = df.drop_duplicates().fillna('Unknown')
    protected = None
    for p in protected_attrs:
        if p in df.columns:
            protected = p
            break
    y_raw = df[target_col].astype(str).copy()
    # Map common income labels to binary
    y = (y_raw.str.contains('>') | y_raw.str.contains('50')).astype(int)
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
    ], remainder='drop')
    X_trans = preprocessor.fit_transform(X)
    prot_series = df[protected] if protected is not None else pd.Series(['Unknown']*len(df))
    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(X_trans, y, prot_series, test_size=test_size, random_state=random_state, stratify=y if len(set(y))>1 else None)
    return X_train, X_test, y_train.values, y_test.values, prot_train.values, prot_test.values, preprocessor, protected, target_col
