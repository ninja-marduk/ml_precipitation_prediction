import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define the directory and file pattern
directory = '/Users/riperez/Conda/anaconda3/doc/precipitation'
file_pattern = 'chirps-v2.0.20*.days_p05.nc'

# Load and concatenate all NetCDF files
file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith('chirps-v2.0.20') and f.endswith('.days_p05.nc')]

# Check if any files were found
if not file_paths:
    raise FileNotFoundError(f"No files matching the pattern '{file_pattern}' were found in the directory '{directory}'.")

datasets = [xr.open_dataset(fp) for fp in file_paths]

# Handle the case where there is only one dataset
if len(datasets) == 1:
    data = datasets[0]
else:
    data = xr.concat(datasets, dim='time')

# Display basic information about the dataset
print("Dataset Info:")
print(data)

# Convert to a Pandas DataFrame for easier analysis
df = data.to_dataframe().reset_index()

# Display basic information about the DataFrame
print("\nDataFrame Info:")
print(df.info())

print("\nDataFrame Head:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Correlation matrix
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Distribution of numerical features
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Time series analysis (if applicable)
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    plt.figure(figsize=(12, 6))
    df.resample('M').mean().plot()
    plt.title("Monthly Average Precipitation")
    plt.ylabel("Precipitation")
    plt.show()

# Categorical feature analysis
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(df[col], order=df[col].value_counts().index)
    plt.title(f"Count of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
