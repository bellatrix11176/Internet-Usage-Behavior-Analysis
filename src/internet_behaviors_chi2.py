"""
Internet Behaviors — Chi-Square Feature Association (Attribute Weights)

Goal:
- Compute chi-square association between each predictor and the target label
  (Online_Shopping), producing a ranked "attribute weights" table.

Data Integrity:
- Invalid sentinel codes (e.g., 99) are treated as MISSING and excluded from
  chi-square contingency tables (so they never influence weights).

Outputs (recreated every run in <repo_root>/outputs):
- internet_behaviors_deidentified_clean.csv
- chi2_attribute_weights.csv
- chi2_attribute_weights.png
- results_summary.txt

Notes:
- Chi-square requires categorical variables. Numeric features are discretized
  into quantile bins before testing.
- The chi2 statistic is used as the "weight" (higher = stronger association).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


# =========================
# SETTINGS
# =========================
LABEL_COL = "Online_Shopping"

POSITIVE_LABELS = {"yes", "y", "1", "true", "t"}
NEGATIVE_LABELS = {"no", "n", "0", "false", "f"}

# Common invalid/sentinel tokens that should NOT be treated as real data.
# (Add more here if your dataset uses other codes.)
INVALID_TOKENS = {"99", "98", "-1", "na", "n/a", "null", "none", ""}

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

DEFAULT_DATA_REL = Path("data") / "Internet-Behaviors.csv"
OUTPUTS_REL = Path("outputs")


# =========================
# PATH HELPERS (GITHUB-SAFE)
# =========================
def find_repo_root(start: Path) -> Path:
    """
    Find the repository root by walking upward from this script.
    We consider a folder the 'root' if it contains a 'data' folder,
    or a README.md, or pyproject.toml.
    """
    start = start.resolve()
    candidates = [start] + list(start.parents)

    for p in candidates:
        if (p / "data").exists() and (p / "data").is_dir():
            return p
        if (p / "README.md").exists():
            return p
        if (p / "pyproject.toml").exists():
            return p

    # Fallback: parent of the script directory
    return start.parent


def resolve_data_path(repo_root: Path, user_path: str | None) -> Path:
    """
    Resolve dataset path in a way that works anywhere:
    - If user provides --data, use it (absolute or relative to current working dir).
    - Else use <repo_root>/data/Internet-Behaviors.csv.
    - If that doesn't exist, try: <repo_root>/data/*.csv if exactly one is present.
    """
    if user_path:
        p = Path(user_path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Cannot find dataset at:\n  {p}\n\n"
            "Fix options:\n"
            "1) Double-check the path you passed to --data\n"
            "2) Or move the dataset into the repo's /data folder\n"
        )

    # Default location inside repo
    default_path = (repo_root / DEFAULT_DATA_REL).resolve()
    if default_path.exists():
        return default_path

    # Helpful fallback: if /data exists and has exactly one CSV, use it
    data_dir = repo_root / "data"
    if data_dir.exists() and data_dir.is_dir():
        csvs = list(data_dir.glob("*.csv"))
        if len(csvs) == 1:
            return csvs[0].resolve()

    raise FileNotFoundError(
        "Cannot find dataset.\n\n"
        f"Repo root:\n  {repo_root}\n\n"
        f"Expected default:\n  {default_path}\n\n"
        "Fix options:\n"
        "1) Put your CSV into: <repo_root>/data/\n"
        "2) Or run with: python src/internet_behaviors_chi2.py --data path/to/yourfile.csv\n"
    )


def ensure_dirs(outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)


# =========================
# DATA CLEANING / PREP
# =========================
def minimal_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning:
    - Strip whitespace from strings
    - Do not aggressively drop rows
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.strip()
    return out


def normalize_yes_no(series: pd.Series) -> pd.Series:
    """
    Normalize common Yes/No formats to 1/0 Int64.
    Invalid tokens become <NA>.
    """
    s = series.astype(str).str.strip().str.lower()

    # Convert invalid tokens to NaN
    s = s.where(~s.isin(INVALID_TOKENS), other=np.nan)

    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[s.isin(POSITIVE_LABELS)] = 1
    out[s.isin(NEGATIVE_LABELS)] = 0

    # If some values are numeric 0/1 already
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
    Invalid tokens become NaN and are excluded by crosstab/dropna.
    """
    # Replace invalid tokens if series is object-like
    if series.dtype == "object":
        s_str = series.astype(str).str.strip().str.lower()
        series = series.where(~s_str.isin(INVALID_TOKENS), other=np.nan)

    s = pd.to_numeric(series, errors="coerce")

    if s.dropna().nunique() <= 1:
        return pd.Series(["(constant)"] * len(series), index=series.index, dtype="category")

    try:
        binned = pd.qcut(s, q=bins, duplicates="drop")
    except Exception:
        binned = pd.cut(s, bins=bins)

    return binned.astype("category")


def normalize_feature_to_category(series: pd.Series) -> pd.Series:
    """
    Normalize a feature for chi-square:
    - Convert invalid tokens (e.g., 99) to NaN
    - If values look yes/no-ish => map to Yes/No category
    - Else keep as categorical
    """
    if series.dtype == "object":
        s = series.astype(str).str.strip().str.lower()
        s = s.where(~s.isin(INVALID_TOKENS), other=np.nan)

        mapped = s.map({
            "yes": "Yes", "y": "Yes", "1": "Yes", "true": "Yes", "t": "Yes",
            "no": "No", "n": "No", "0": "No", "false": "No", "f": "No",
        })

        # If mapping worked for at least some rows, use it; keep the rest as original strings
        if mapped.notna().any():
            out = mapped
        else:
            out = s

        return out.astype("category")

    # Numeric/bool columns: treat invalid sentinel 99 as NaN if present
    s_num = pd.to_numeric(series, errors="coerce")
    s_num = s_num.where(~s_num.isin([99, 98, -1]), other=np.nan)
    return s_num.astype("category")


# =========================
# CHI-SQUARE
# =========================
def compute_chi2_weight(df: pd.DataFrame, feature: str, label: str) -> dict:
    """
    Chi-square test between feature and label using contingency table.
    NaNs are excluded so invalid tokens never influence chi2.
    """
    tmp = df[[feature, label]].dropna()
    contingency = pd.crosstab(tmp[feature], tmp[label])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"attribute": feature, "chi2": 0.0, "p_value": 1.0, "dof": 0, "n_used": int(len(tmp))}

    chi2, p, dof, _ = chi2_contingency(contingency)
    return {
        "attribute": feature,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "n_used": int(len(tmp)),
    }


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
    lines.append("=" * 66)
    lines.append("")
    lines.append(f"Label (target): {LABEL_COL} (normalized to 0/1)")
    lines.append("")
    lines.append("How to read this:")
    lines.append("- Higher chi2 => stronger association with the label (not causation).")
    lines.append("- Lower p-value => stronger evidence the association is not random.")
    lines.append("- High p-value (near 1.0) => weak/no evidence the feature matters here.")
    lines.append("- Invalid sentinel values (e.g., 99) were treated as missing and excluded.")
    lines.append("")

    lines.append("Ranked results (highest to lowest chi2):")
    for _, r in top.iterrows():
        lines.append(
            f"  - {r['attribute']}: chi2={r['chi2']:.3f}, p={r['p_value']:.3f}, dof={int(r['dof'])}, n_used={int(r['n_used'])}"
        )

    lines.append("")
    weak = top[top["p_value"] >= PVALUE_WEAK_THRESHOLD]
    lines.append(f"Weak-evidence features (p >= {PVALUE_WEAK_THRESHOLD:.2f}): {len(weak)}")
    if len(weak) > 0:
        lines.append("These should not be over-trusted as important predictors in this dataset.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# MAIN
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chi-square feature association weights for Internet Behaviors."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset CSV (defaults to repo_root/data/Internet-Behaviors.csv)",
    )
    args = parser.parse_args()

    # Locate repo root from script location, NOT from where you launched python
    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir)

    outputs_dir = (repo_root / OUTPUTS_REL).resolve()
    ensure_dirs(outputs_dir)

    data_path = resolve_data_path(repo_root, args.data)
    print(f"✅ Repo root:\n{repo_root}\n")
    print(f"✅ Using dataset:\n{data_path}\n")
    print(f"✅ Outputs dir:\n{outputs_dir}\n")

    df = pd.read_csv(data_path)
    df = minimal_cleaning(df)

    # Validate label
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Normalize label and drop missing label rows
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

    # Build analysis frame
    work = df[[LABEL_COL] + PREDICTOR_COLS].copy()

    # Prepare predictors for chi-square
    for col in PREDICTOR_COLS:
        if col in ("Hours_Per_Day", "Years_on_Internet"):
            work[col] = discretize_numeric(work[col], bins=NUMERIC_BINS)
        else:
            work[col] = normalize_feature_to_category(work[col])

    # Label categorical
    work[LABEL_COL] = work[LABEL_COL].astype(int).astype("category")

    # Compute chi-square weights
    results = [compute_chi2_weight(work, feature, LABEL_COL) for feature in PREDICTOR_COLS]
    weights_df = pd.DataFrame(results).sort_values("chi2", ascending=False)

    # Save weights table
    out_weights_csv = outputs_dir / "chi2_attribute_weights.csv"
    weights_df.to_csv(out_weights_csv, index=False)

    # Save chart
    out_chart = outputs_dir / "chi2_attribute_weights.png"
    save_bar_chart(weights_df, out_chart)

    # De-identified dataset export (remove sensitive columns)
    deid = df.copy()
    drop_cols = [c for c in SENSITIVE_COLS if c in deid.columns]
    deid = deid.drop(columns=drop_cols, errors="ignore")

    out_clean_csv = outputs_dir / "internet_behaviors_deidentified_clean.csv"
    deid.to_csv(out_clean_csv, index=False)

    # Summary
    out_summary = outputs_dir / "results_summary.txt"
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
