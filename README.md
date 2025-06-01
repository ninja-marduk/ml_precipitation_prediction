# Precipitation Prediction Models Comparison

This repository contains implementations of various deep learning models for precipitation prediction, including ConvLSTM, ConvGRU, and hybrid Topoclus-CEEMDAN-TVF-AFC models.

## Models Implemented

1. **ConvLSTM**: Convolutional LSTM model for spatial-temporal precipitation prediction
2. **ConvGRU**: Convolutional GRU model for spatial-temporal precipitation prediction
3. **Hybrid Topoclus Model**: A combination of Topoclus-CEEMDAN-TVF-AFC with ConvBiGRU-AE and ConvLSTM-AE

## Features

- Complete training and evaluation pipeline
- Multiple evaluation metrics (RMSE, MAE, MAPE%, r, R²)
- Visualization capabilities:
  - Prediction maps (white to blue scale)
  - Error maps (white to red scale)
- Support for 12-month predictions
- Early stopping and model checkpointing
- Data preprocessing and standardization
- Integration with elevation and cluster information

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml_precipitation_prediction.git
cd ml_precipitation_prediction
```

2. Create a conda environment:
```bash
conda create -n precipitation_prediction python=3.9
conda activate precipitation_prediction
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The code expects the following data files in the `data` directory:
- `precipitation.nc`: NetCDF file containing precipitation data
- `elevation.npy`: NumPy file containing elevation data
- `clusters.npy`: NumPy file containing cluster information

## Usage

1. Ensure your data files are in the correct location and format.

2. Run the model comparison:
```bash
python models/precipitation_model_comparison.py
```

3. Results will be saved in the `results/model_comparison` directory:
- Model checkpoints (*.pth files)
- Prediction maps (PNG format)
- Error maps (PNG format)
- Metrics comparison CSV file

## Configuration

You can modify the model configuration in the `precipitation_model_comparison.py` script:

```python
config = {
    'input_channels': 1,
    'hidden_channels': 64,
    'output_channels': 1,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 100
}
```

## Output

The code generates:
1. Monthly prediction maps for each model
2. Monthly error maps for each model
3. A comprehensive CSV file with performance metrics
4. Trained model checkpoints

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_reference,
    title={Your Paper Title},
    author={Your Name},
    journal={Journal Name},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
