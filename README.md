# Internet Behaviors — Data Cleaning & Chi-Square Feature Association

This project demonstrates **foundational data science skills** using a real-world behavioral dataset, with a focus on:

- Data cleaning and integrity
- Handling invalid / sentinel values correctly
- Chi-Square statistical testing for feature association
- Reproducible analysis using Python

The goal is to evaluate which internet behavior attributes are most strongly associated with **Online Shopping** behavior, while avoiding common analytical pitfalls that can distort results.

---

## Project Overview

Many real-world datasets contain **invalid placeholder values** (such as `99`) that represent *unknown*, *not applicable*, or *missing* data. If these values are mistakenly treated as real numeric or categorical data, they can **corrupt statistical results**.

This project explicitly identifies and removes such invalid values **before** performing Chi-Square tests, ensuring that only meaningful observations influence the analysis.

---

## Dataset

**Internet-Behaviors.csv**

- Behavioral indicators (e.g., Twitter usage, Online Gaming, Reading News)
- Time-based metrics (e.g., Years on Internet, Hours per Day)
- Target label: `Online_Shopping` (Yes / No)

---

## Key Data Integrity Decision: Handling Invalid Values (`99`)

During exploratory analysis, it was identified that certain behavior fields (notably **Twitter**) contained the value `99`.

### Why this matters

- `99` does **not** represent real behavior
- It is a sentinel code meaning *Unknown / Not Reported*
- Treating `99` as a real category or numeric value artificially inflates Chi-Square statistics

### What was done

- All invalid sentinel values (`99`, `98`, `-1`, `"NA"`, etc.) are explicitly converted to **missing values**
- Missing values are **excluded from contingency tables**
- Invalid entries are **never discretized**, **never categorized**, and **never included in Chi-Square math**

This guarantees that the statistical associations reflect **actual behavior**, not data-entry artifacts.

---

## Analysis Methodology

1. **Minimal Cleaning**
   - Trim whitespace
   - Preserve rows whenever possible
   - Avoid aggressive row deletion

2. **Label Normalization**
   - `Online_Shopping` normalized to binary (0 / 1)

3. **Feature Preparation**
   - Numeric features discretized into quantile bins (Chi-Square requirement)
   - Categorical features standardized
   - Invalid values removed before testing

4. **Chi-Square Testing**
   - Performed independently for each feature
   - Chi² statistic used as the attribute weight
   - p-values retained for statistical interpretation

---

## Outputs (Auto-Generated)

Each run recreates the following files in the `outputs/` directory:

- `chi2_attribute_weights.csv` — ranked Chi-Square results
- `chi2_attribute_weights.png` — bar chart visualization
- `internet_behaviors_deidentified_clean.csv` — cleaned, shareable dataset
- `results_summary.txt` — plain-English interpretation

---

## How to Run

```bash
python src/internet_behaviors_chi2.py
```
🌐 **PixelKraze Analytics (Portfolio):** https://pixelkraze.com/?utm_source=github&utm_medium=readme&utm_campaign=portfolio&utm_content=homepage

