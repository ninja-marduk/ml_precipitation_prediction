
# Example usage of meta-analysis
import pandas as pd
import numpy as np

def run_meta_analysis_example():
    # Load extracted data
    data = pd.read_csv("meta_analysis_template.csv")
    
    # Calculate log-ratios
    data['Log_Ratio'] = np.log(data['Hybrid_Error'] / data['Baseline_Error'])
    data['Improvement_Pct'] = (np.exp(data['Log_Ratio']) - 1) * 100
    
    # Group by category
    summary = data.groupby('Hybrid_Category')['Log_Ratio'].agg(['median', 'std', 'count'])
    print(summary)
    
    return data

if __name__ == "__main__":
    run_meta_analysis_example()
