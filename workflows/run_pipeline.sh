#!/bin/bash
# ML Precipitation Prediction - Pipeline Shell Wrapper
#
# Usage:
#   bash workflows/run_pipeline.sh                  # Run all stages
#   bash workflows/run_pipeline.sh --from 7          # From stage 7
#   bash workflows/run_pipeline.sh --stages 8 9      # Specific stages
#   bash workflows/run_pipeline.sh --skip-gpu         # Skip GPU stages
#   bash workflows/run_pipeline.sh --dry-run          # Preview only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "  ML Precipitation Prediction Pipeline"
echo "  Project root: $PROJECT_ROOT"
echo "=================================================="

# Activate conda environment if available
if command -v conda &> /dev/null; then
    CONDA_ENV="${CONDA_ENV:-precipitation}"
    if conda env list | grep -q "$CONDA_ENV"; then
        echo "Activating conda environment: $CONDA_ENV"
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
    fi
fi

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found in PATH"
    exit 1
fi

echo "Python: $(python --version)"
echo "Working directory: $PROJECT_ROOT"
echo ""

# Run the orchestrator with all arguments passed through
cd "$PROJECT_ROOT"
python workflows/run_pipeline.py "$@"
