import pandas as pd
from sklearn.impute import KNNImputer

def vshd_imputation(data):
    """
    Perform Variable Specific Hot Deck (VSHD) imputation on the dataset.
    """
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return imputed_data
