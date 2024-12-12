from models.shem_model import build_shem_model
from models.metrics import calculate_metrics, export_metrics_to_csv
from preprocessing.data_preprocessing import load_and_preprocess_data  # Updated import

# Load and preprocess the dataset
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/Precipitation_Data_LL.csv')

# Build and train SHEM model adapted for time series
model = build_shem_model(X_train, y_train)

# Generate predictions
y_pred_rf = model.predict(X_test)

# Calculate metrics
metrics = calculate_metrics(y_test, y_pred_rf, y_pred_rf, y_pred_rf)  # Replace with predictions from other models if available

# Export metrics to CSV
export_metrics_to_csv(metrics)

print("Model training complete and metrics exported.")
