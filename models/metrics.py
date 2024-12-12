import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_test, y_pred_rf, y_pred_svm, y_pred_dt):
    """
    Calculate accuracy, precision, recall, and F1 score for different models.
    Ensure both true values and predictions are binary categorical values.
    """
    # Convertimos los valores a binarios categóricos para garantizar consistencia
    y_test = pd.Series(y_test).astype(str)
    y_pred_rf = pd.Series(y_pred_rf).astype(str)
    y_pred_svm = pd.Series(y_pred_svm).astype(str)
    y_pred_dt = pd.Series(y_pred_dt).astype(str)

    metrics = {
        'RandomForest': {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf, average='macro', zero_division=1),
            'recall': recall_score(y_test, y_pred_rf, average='macro', zero_division=1),
            'f1_score': f1_score(y_test, y_pred_rf, average='macro', zero_division=1)
        },
        'SVM': {
            'accuracy': accuracy_score(y_test, y_pred_svm),
            'precision': precision_score(y_test, y_pred_svm, average='macro', zero_division=1),
            'recall': recall_score(y_test, y_pred_svm, average='macro', zero_division=1),
            'f1_score': f1_score(y_test, y_pred_svm, average='macro', zero_division=1)
        },
        'DecisionTree': {
            'accuracy': accuracy_score(y_test, y_pred_dt),
            'precision': precision_score(y_test, y_pred_dt, average='macro', zero_division=1),
            'recall': recall_score(y_test, y_pred_dt, average='macro', zero_division=1),
            'f1_score': f1_score(y_test, y_pred_dt, average='macro', zero_division=1)
        }
    }
    return metrics

def export_metrics_to_csv(metrics, file_path='data/model_metrics.csv'):
    """
    Export model metrics to a CSV file.
    """
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv(file_path)
    print(f'Metrics exported to {file_path}')
