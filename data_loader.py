# data_loader.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE
    smote_available = True
except ModuleNotFoundError:
    smote_available = False

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

def load_adult_data():
    train = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        names=columns, sep=",\s*", engine="python"
    )
    test = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        names=columns, sep=",\s*", engine="python", skiprows=1
    )
    test['income'] = test['income'].str.replace('.', '', regex=False)
    df = pd.concat([train, test], ignore_index=True)
    return df

def preprocess_data(df, target_col="income"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical columns
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)
    
    # Handle imbalance with SMOTE
    if smote_available:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
