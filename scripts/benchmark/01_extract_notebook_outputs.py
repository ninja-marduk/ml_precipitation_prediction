"""
Benchmark Analysis Script 1: Extract Notebook Outputs

Part of V2 vs V3 Comparative Analysis Pipeline
Extracts metrics and results from Jupyter notebook cell outputs
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'docs' / 'models' / 'comparative' / 'data'
NOTEBOOKS_DIR = PROJECT_ROOT / 'models'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)


def parse_jupyter_notebook(notebook_path: Path) -> Dict:
    """
    Parse Jupyter notebook and extract structure.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        Dictionary containing notebook structure
    """
    logger.info(f"Parsing notebook: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    logger.info(f"  Notebook version: {notebook.get('nbformat', 'unknown')}.{notebook.get('nbformat_minor', 'unknown')}")
    logger.info(f"  Total cells: {len(notebook.get('cells', []))}")

    return notebook


def extract_cell_outputs(notebook: Dict) -> List[Dict]:
    """
    Extract outputs from all code cells.

    Args:
        notebook: Parsed notebook dictionary

    Returns:
        List of cell output dictionaries
    """
    outputs = []
    cells = notebook.get('cells', [])

    for idx, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            cell_outputs = cell.get('outputs', [])

            for output in cell_outputs:
                output_data = {
                    'cell_index': idx,
                    'output_type': output.get('output_type', 'unknown'),
                    'text': '',
                    'data': {}
                }

                # Extract text output (stdout)
                if 'text' in output:
                    if isinstance(output['text'], list):
                        output_data['text'] = ''.join(output['text'])
                    else:
                        output_data['text'] = output['text']

                # Extract display data (tables, plots, etc.)
                if 'data' in output:
                    output_data['data'] = output['data']

                # Extract execution results
                if output.get('output_type') == 'execute_result':
                    if 'data' in output:
                        output_data['data'] = output['data']

                outputs.append(output_data)

    logger.info(f"  Extracted {len(outputs)} outputs from code cells")
    return outputs


def extract_metrics_from_text(text: str) -> List[Dict]:
    """
    Parse text output for metric values.

    Args:
        text: Text output from cell

    Returns:
        List of extracted metrics
    """
    metrics = []

    # Pattern for metric lines: "Metric_Name: value" or "Metric_Name = value"
    metric_patterns = [
        r'(\w+)\s*[=:]\s*([0-9.]+)',
        r'RMSE\s*[=:]\s*([0-9.]+)',
        r'MAE\s*[=:]\s*([0-9.]+)',
        r'R\^?2\s*[=:]\s*([0-9.]+)',
        r'R²\s*[=:]\s*([0-9.]+)',
        r'Bias\s*[=:]\s*([+-]?[0-9.]+)',
    ]

    for pattern in metric_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if len(match.groups()) == 2:
                metric_name = match.group(1)
                metric_value = match.group(2)
            else:
                metric_name = pattern.split('\\')[0].strip()
                metric_value = match.group(1)

            try:
                metrics.append({
                    'metric': metric_name,
                    'value': float(metric_value)
                })
            except ValueError:
                continue

    return metrics


def process_notebook(notebook_path: Path, version: str) -> pd.DataFrame:
    """
    Process a notebook and extract all metrics.

    Args:
        notebook_path: Path to notebook file
        version: 'V2' or 'V3'

    Returns:
        DataFrame with extracted metrics
    """
    logger.info(f"Processing {version} notebook...")

    notebook = parse_jupyter_notebook(notebook_path)
    outputs = extract_cell_outputs(notebook)

    all_metrics = []

    for output in outputs:
        if output['text']:
            metrics = extract_metrics_from_text(output['text'])
            for metric in metrics:
                all_metrics.append({
                    'version': version,
                    'cell_index': output['cell_index'],
                    'metric_name': metric['metric'],
                    'metric_value': metric['value']
                })

    df = pd.DataFrame(all_metrics)

    if not df.empty:
        logger.info(f"  Extracted {len(df)} metric values")
        logger.info(f"  Unique metrics: {df['metric_name'].unique().tolist()}")
    else:
        logger.warning(f"  No metrics extracted from {version} notebook")

    return df


def save_extraction_log(v2_df: pd.DataFrame, v3_df: pd.DataFrame):
    """Save extraction summary log."""
    log_path = DATA_DIR / 'extraction_log.txt'

    with open(log_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("NOTEBOOK EXTRACTION LOG\n")
        f.write("="*60 + "\n\n")

        f.write("V2 Notebook Extraction:\n")
        f.write(f"  Total metrics extracted: {len(v2_df)}\n")
        if not v2_df.empty:
            f.write(f"  Unique metric types: {v2_df['metric_name'].nunique()}\n")
            f.write(f"  Cells with outputs: {v2_df['cell_index'].nunique()}\n")
        f.write("\n")

        f.write("V3 Notebook Extraction:\n")
        f.write(f"  Total metrics extracted: {len(v3_df)}\n")
        if not v3_df.empty:
            f.write(f"  Unique metric types: {v3_df['metric_name'].nunique()}\n")
            f.write(f"  Cells with outputs: {v3_df['cell_index'].nunique()}\n")
        f.write("\n")

        f.write("="*60 + "\n")

    logger.info(f"Extraction log saved to {log_path}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Script 1: Extract Notebook Outputs")
    logger.info("="*60)

    # Define notebook paths
    v2_notebook = NOTEBOOKS_DIR / 'base_models_Conv_STHyMOUNTAIN_V2.ipynb'
    v3_notebook = NOTEBOOKS_DIR / 'base_models_Conv_STHyMOUNTAIN_V3_FNO.ipynb'

    # Check if notebooks exist
    if not v2_notebook.exists():
        logger.error(f"V2 notebook not found: {v2_notebook}")
        return
    if not v3_notebook.exists():
        logger.error(f"V3 notebook not found: {v3_notebook}")
        return

    # Process notebooks
    v2_df = process_notebook(v2_notebook, 'V2')
    v3_df = process_notebook(v3_notebook, 'V3')

    # Save results
    v2_output = DATA_DIR / 'v2_notebook_outputs.csv'
    v3_output = DATA_DIR / 'v3_notebook_outputs.csv'

    v2_df.to_csv(v2_output, index=False)
    v3_df.to_csv(v3_output, index=False)

    logger.info(f"V2 outputs saved to {v2_output}")
    logger.info(f"V3 outputs saved to {v3_output}")

    # Save extraction log
    save_extraction_log(v2_df, v3_df)

    logger.info("="*60)
    logger.info("Completed successfully")
    logger.info("="*60)


if __name__ == '__main__':
    main()
