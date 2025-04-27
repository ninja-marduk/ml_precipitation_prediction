import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the precipitation dataset for time series modeling.

    Parameters:
        filepath (str): Path to the CSV file containing precipitation data.

    Returns:
        pandas.DataFrame: Processed dataframe with datetime index and additional features.
    """
    # Load data
    df = pd.read_csv(filepath)

    # Melt the DataFrame to have a 'Month' column
    df_melted = df.melt(id_vars=['SUBDIVISION', 'YEAR', 'Latitude', 'Longitude'],
                        value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
                        var_name='Month', value_name='Precipitation')

    # Convert 'YEAR' and 'Month' to datetime
    df_melted['Date'] = pd.to_datetime(df_melted['YEAR'].astype(str) + df_melted['Month'], format='%Y%b')

    # Convert Month to numeric representation
    month_dict = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    df_melted['month_num'] = df_melted['Month'].map(month_dict)

    # Add cyclical features for month
    df_melted['month_sin'] = np.sin(2 * np.pi * df_melted['month_num'] / 12)
    df_melted['month_cos'] = np.cos(2 * np.pi * df_melted['month_num'] / 12)

    # Add day of year features (assuming middle of month)
    df_melted['doy'] = df_melted['Date'].dt.dayofyear
    df_melted['doy_sin'] = np.sin(2 * np.pi * df_melted['doy'] / 365)
    df_melted['doy_cos'] = np.cos(2 * np.pi * df_melted['doy'] / 365)

    # Set Date as index for time-series analysis
    df_melted = df_melted.set_index('Date')

    return df_melted
