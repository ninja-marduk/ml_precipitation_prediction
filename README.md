# ML stacking precipitation prediction
Precipitation prediction models



## Instructions
Place your dataset in the `data/ directory`.
Run `pip install -r requirements.txt` to install the necessary packages.
Execute `python .\main.py` to run the entire workflow, from data loading to model training and evaluation.

# Structure
precipitation_prediction/  
│  
├── data/  
│   ├── Precipitation_Data_LL.csv          # Precipitation data provided for India with coordinates and dates  
│   └── model_metrics.csv             # Output file with model metrics  
│
├── models/  
│   ├── imputation.py                 # Module for imputing missing values (if needed)  
│   ├── feature_selection.py          # Module for feature selection (might not be needed for time series)  
│   ├── shem_model.py                 # Module with the Stacked Heterogeneous Ensemble Model for time series  
│   ├── metrics.py                    # Module for calculating and exporting model metrics  
│  
├── utils/  
│   └── data_loader.py                # Utility module for loading data (if used for other formats)  
│
├── preprocessing/  
│   └── data_preprocessing.py         # Preprocessing module to prepare time series and spatial data  
│  
├── main.py                           # Main script to execute the data flow, training, and evaluation  
├── requirements.txt                  # File with project dependencies (required libraries)  
└── README.md                         # Project documentation, usage instructions, etc.  
