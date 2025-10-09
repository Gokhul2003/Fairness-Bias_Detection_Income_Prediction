import pandas as pd, os
def load_dataset(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'income_data.csv')
    df = pd.read_csv(path)
    return df
