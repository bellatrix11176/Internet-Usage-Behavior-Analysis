"""
Internet Behaviors — Chi-Square Feature Association (Attribute Weights)

Goal:
- Compute chi-square association between each predictor and the target label
  (Online_Shopping), producing a ranked "attribute weights" table.

Outputs (recreated every run):
- outputs/internet_behaviors_deidentified_clean.csv
- outputs/chi2_attribute_weights.csv
- outputs/chi2_attribute_weights.png
- outputs/results_summary.txt

Notes:
- Chi-square requires categorical variables. Numeric features are discretized
  into quantile bins before testing.
- The chi2 statistic is used as the "weight" (higher = stronger association).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


# =========================
# SETTINGS (EDIT THESE IF NEEDED)
# =========================
LABEL_COL = "Online_Shopping"  # target used for Chi2 weighting

POSITIVE_LABELS = {"yes", "y", "1", "true", "t"}
NEGATIVE_LABELS = {"no", "n", "0", "false", "f"}

# Predictors included in the chi-square weighting run
PREDICTOR_COLS = [
    "Facebook",
    "Online_Gaming",
    "Other_Social_Network",
    "Twitter",
    "Read_News",
    "Hours_Per_Day",
    "Years_on_Internet",
]

# Remove sensitive fields for a shareable cleaned dataset (kept OUT of chi2 by default)
SENSITIVE_COLS = [
    "Birth_Year",
    "Race",
    "Gender",
    "Marital_Status",
    "Preferred_Browser",
    "Preferred_Search_Engine",
    "Preferred_Email",
]

NUMERIC_BINS = 4
PVALUE_WEAK_THRESHOLD = 0.20

DEFAULT_DATA_PATH = Path("data") / "Internet-Behaviors.csv"
OUTPUT_DIR = Path("outputs")


# =========================
# HELPERS
# =========================
def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_yes_no(series: pd.Series) -> pd.Series:
    """
    Normalize common Yes/No formats to 1/0 Int64.
    """
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float64")

    out[s.isin(POSITIVE_LABELS)] = 1
    out[s.isin(NEGATIVE_LABELS)] = 0

    if out.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.where(numeric.isin([0, 1]))
        out = out.fillna(numeric)

    return out.astype("Int64")


def safe_category(series: pd.Series) -> pd.Series:
    """
    Convert a series to categorical if appropriate.
    """
    if isinstance(series.dtype, CategoricalDtype):
        return series.astype("category")
    if series.dtype == "object" or series.dtype == "bool":
        return series.astype("category")
    return series


def discretize_numeric(series: pd.Series, bins: int = 4) -> pd.Series:
    """
    Discretize numeric series into equal-frequency bins (qcut).
    """
    s = pd.to_numeric(series, errors="coerce")

    if s.dropna().nunique() <= 1:
        return pd.Series(["(constant)"] * len(series), index=series.index, dtype="category")

    try:
        binned = pd.qcut(s, q=bins, duplicates="drop")
    except Exception:
        binned = pd.cut(s, bins=bins)

    return binned.astype("category")


def minimal_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning:
    - Trim strings
    - Avoid aggressive dropping
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.strip()
    return out


def compute_chi2_weight(df: pd.DataFrame, feature: str, label: str) -> dict:
    """
    Chi-square test between feature and label using contingency table.
    """
    contingency = pd.crosstab(df[feature], df[label])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"attribute": feature, "chi2": 0.0, "p_value": 1.0, "dof": 0}

    chi2, p, dof, _ = chi2_contingency(contingency)
    return {"attribute": feature, "chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def save_bar_chart(weights_df: pd.DataFrame, out_path: Path) -> None:
    """
    Bar chart of Chi2 weights.
    """
    top = weights_df.sort_values("chi2", ascending=False)

    plt.figure()
    plt.bar(top["attribute"], top["chi2"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Chi-Square Weight (Chi2 Statistic)")
    plt.title("Attribute Weights by Chi-Square Statistic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_summary(weights_df: pd.DataFrame, out_path: Path) -> None:
    """
    Plain-English interpretation of results.
    """
    top = weights_df.sort_values("chi2", ascending=False)

    lines = []
    lines.append("Internet Behaviors — Chi-Square Feature Association Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Label (target): {LABEL_COL} (normalized to 0/1)")
    lines.append("")
    lines.append("How to read this:")
    lines.append("- Higher chi2 => stronger association with the label (not causation).")
    lines.append("- Lower p-value => stronger evidence the association is not random.")
    lines.append("- High p-value (near 1.0) => weak/no evidence the feature matters here.")
    lines.append("")

    lines.append("Ranked results (highest to lowest chi2):")
    for _, r in top.iterrows():
        lines.append(f"  - {r['attribute']}: chi2={r['chi2']:.3f}, p={r['p_value']:.3f}, dof={int(r['dof'])}")
    lines.append("")

    weak = top[top["p_value"] >= PVALUE_WEAK_THRESHOLD]
    lines.append(f"Weak-evidence features (p >= {PVALUE_WEAK_THRESHOLD:.2f}): {len(weak)}")
    if len(weak) > 0:
        lines.append("These should not be over-trusted as important predictors in this dataset.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_data_path(user_path: str | None) -> Path:
    """
    Resolve dataset path:
    - If user passes --data, use it
    - Else use default data/Internet-Behaviors.csv
    - Give a helpful error message if not found
    """
    path = Path(user_path) if user_path else DEFAULT_DATA_PATH
    if path.exists():
        return path

    # Helpful fallback: search for any CSV in ./data
    data_dir = Path("data")
    if data_dir.exists():
        csvs = list(data_dir.glob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]

    raise FileNotFoundError(
        f"Cannot find dataset at:\n  {path.resolve()}\n\n"
        "Fix options:\n"
        "1) Put your CSV into the repo's /data folder (recommended)\n"
        "2) Or run with: python src/python_internet_behaviors.py --data path/to/yourfile.csv\n"
    )


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Chi-square feature association weights for Internet Behaviors.")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset CSV (defaults to data/Internet-Behaviors.csv)")
    args = parser.parse_args()

    ensure_dirs()

    data_path = resolve_data_path(args.data)
    print(f"✅ Using dataset:\n{data_path.resolve()}\n")

    df = pd.read_csv(data_path)
    df = minimal_cleaning(df)

    # Validate label
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Normalize label to 0/1 and drop rows where label is missing
    df[LABEL_COL] = normalize_yes_no(df[LABEL_COL])
    df = df.dropna(subset=[LABEL_COL]).copy()

    # Ensure predictors exist
    missing_preds = [c for c in PREDICTOR_COLS if c not in df.columns]
    if missing_preds:
        raise ValueError(
            "Some expected predictor columns are missing:\n"
            f"{missing_preds}\n\n"
            "If your column names differ, update PREDICTOR_COLS in the script."
        )

    # Build analysis frame: label + predictors
    work = df[[LABEL_COL] + PREDICTOR_COLS].copy()

    # Prepare predictors:
    # - numeric -> discretize
    # - yes/no-ish -> category
    for col in PREDICTOR_COLS:
        if col in ("Hours_Per_Day", "Years_on_Internet"):
            work[col] = discretize_numeric(work[col], bins=NUMERIC_BINS)
        else:
            if work[col].dtype == "object":
                s = work[col].astype(str).str.strip().str.lower()
                mapped = s.map({
                    "yes": "Yes", "y": "Yes", "1": "Yes", "true": "Yes",
                    "no": "No", "n": "No", "0": "No", "false": "No"
                })
                if mapped.notna().any():
                    work[col] = mapped.fillna(work[col].astype(str)).astype("category")
                else:
                    work[col] = safe_category(work[col])
            else:
                work[col] = safe_category(work[col])

    # Label as category
    work[LABEL_COL] = work[LABEL_COL].astype(int).astype("category")

    # =========================
    # CHI-SQUARE WEIGHTS
    # =========================
    results = [compute_chi2_weight(work, feature, LABEL_COL) for feature in PREDICTOR_COLS]
    weights_df = pd.DataFrame(results).sort_values("chi2", ascending=False)

    # Save weights
    out_weights_csv = OUTPUT_DIR / "chi2_attribute_weights.csv"
    weights_df.to_csv(out_weights_csv, index=False)

    # Save chart
    out_chart = OUTPUT_DIR / "chi2_attribute_weights.png"
    save_bar_chart(weights_df, out_chart)

    # =========================
    # DE-IDENTIFIED CLEAN EXPORT
    # =========================
    deid = df.copy()
    drop_cols = [c for c in SENSITIVE_COLS if c in deid.columns]
    deid = deid.drop(columns=drop_cols, errors="ignore")

    out_clean_csv = OUTPUT_DIR / "internet_behaviors_deidentified_clean.csv"
    deid.to_csv(out_clean_csv, index=False)

    # Summary
    out_summary = OUTPUT_DIR / "results_summary.txt"
    write_summary(weights_df, out_summary)

    # Console preview
    print("Top attributes by Chi² weight:")
    print(weights_df.head(10).to_string(index=False))
    print("\nBottom attributes by Chi² weight:")
    print(weights_df.tail(10).to_string(index=False))

    print("\n✅ Outputs created:")
    print(f"- {out_clean_csv}")
    print(f"- {out_weights_csv}")
    print(f"- {out_chart}")
    print(f"- {out_summary}")


if __name__ == "__main__":
    main()
