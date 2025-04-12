import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the precipitation dataset for time series modeling.
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
    month_mapping = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                     'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    df_melted['Month'] = df_melted['Month'].map(month_mapping)

    # Sort the data by Date for each location
    df_melted = df_melted.sort_values(by=['Latitude', 'Longitude', 'Date'])

    # Feature engineering: Create lag features for the precipitation
    df_melted['Precipitation_lag_1'] = df_melted.groupby(['Latitude', 'Longitude'])['Precipitation'].shift(1)
    df_melted['Precipitation_lag_2'] = df_melted.groupby(['Latitude', 'Longitude'])['Precipitation'].shift(2)

    # Drop missing values created by lags
    df_melted = df_melted.dropna()

    # Define features and target
    X = df_melted[['Latitude', 'Longitude', 'Precipitation_lag_1', 'Precipitation_lag_2', 'YEAR', 'Month']]
    y = df_melted['Precipitation']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
