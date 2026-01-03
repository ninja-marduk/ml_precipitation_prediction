import pandas as pd
import numpy as np
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
TEMPLATE_PATH = BASE_DIR / "smote_acc_template.csv"
MODELS_CSV = BASE_DIR / "models_avg_metrics.csv"

MODEL_ALIASES = {
    "SMOTE-km-XGB": "SMOTE-km-XGB",
    "SMOTE-km-RF": "SMOTE-km-RF",
    "SMOTE-km-RNN": "SMOTE-km-RNN",
    "SMOTE-km-LSTM": "SMOTE-km-LSTM",
}

REFERENCE = "Medium- and Long-Term Precipitation Forecasting Method Based on Data Augmentation and Machine Learning Algorithms"


def compute_r_squared(acc_values: pd.Series) -> float:
    acc_values = pd.to_numeric(acc_values, errors="coerce").dropna()
    if acc_values.empty:
        return np.nan
    r_mean = acc_values.mean()
    return float(np.clip(r_mean, -1, 1) ** 2)


def append_entries(model: str, r2: float, models_csv: Path) -> None:
    # We add R^2 entry; RMSE/MAE are not available in the paper
    # CSV uses comma as separator and dot decimals; respect existing style
    with models_csv.open("a", encoding="utf-8") as f:
        f.write(f"{REFERENCE}\tR^2\t{r2:.4f}\t\t{model} \n")


def main() -> None:
    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Template not found: {TEMPLATE_PATH}")

    df = pd.read_csv(TEMPLATE_PATH)
    if df.empty:
        raise SystemExit("Template has no rows; fill acc_r values first.")

    for model, group in df.groupby("model"):
        r2 = compute_r_squared(group["acc_r"])
        if np.isnan(r2):
            print(f"Skipping {model}: no ACC values provided")
            continue
        append_entries(MODEL_ALIASES.get(model, model), r2, MODELS_CSV)
        print(f"Added R^2={r2:.4f} for {model}")

    # Run normalization and analysis to refresh outputs
    subprocess.run(["python", str(BASE_DIR / "final_normalization.py")], check=True)
    subprocess.run(["python", str(BASE_DIR / "updated_analysis_normalized.py")], check=True)


if __name__ == "__main__":
    main()
