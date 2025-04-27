import pandas as pd

def load_data(filepath):
    """
    Load dataset from a given CSV file.

    Parameters:
        filepath (str): Path to the CSV file to load.

    Returns:
        pandas.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(filepath)
