import pandas as pd

def load_data(filepath):
    """
    Load dataset from a given CSV file.
    """
    return pd.read_csv(filepath)
