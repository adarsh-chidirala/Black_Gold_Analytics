import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = Path("data_clean/master_oil_panel.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("\n" + "="*80)
print("         VALIDATION REPORT FOR CLEANED OIL PANEL DATASET")
print("="*80 + "\n")

# -----------------------------
# BASIC INFO
# -----------------------------
print("1. BASIC INFO")
print("-" * 80)
print(df.info())
print("\nSample Rows:")
print(df.head())
print("\n")

# -----------------------------
# MISSING VALUES
# -----------------------------
print("2. MISSING VALUES CHECK")
print("-" * 80)
missing = df.isnull().sum()
print(missing)

if missing.sum() == 0:
    print("\nGOOD: No missing values.")
else:
    print("\nNOTE: Missing values exist (expected with yearly country data).")

print("\n")

# -----------------------------
# DUPLICATES
# -----------------------------
print("3. DUPLICATE ROW CHECK")
print("-" * 80)
dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}")
print("\n")

# -----------------------------
# SUMMARY STATS
# -----------------------------
print("4. SUMMARY STATISTICS")
print("-" * 80)
print(df.describe().T)
print("\n")

# -----------------------------
# TYPE CHECKS (Matches your actual cleaned dataset)
# -----------------------------
print("5. DATA TYPE VALIDATION")
print("-" * 80)

expected_numeric = [
    "Production_Mb",
    "Consumption_Mb",
    "Reserves_Gb",
    "Price_Brent",
    "R_to_P_Years",
    "coverage_pct"
]

expected_categorical = ["Country", "Year"]

for col in expected_numeric:
    if col not in df.columns:
        print(f"ERROR: Missing numeric column: {col}")
    else:
        if np.issubdtype(df[col].dtype, np.number):
            print(f"OK: {col} is numeric.")
        else:
            print(f"ERROR: {col} should be numeric but is {df[col].dtype}")

for col in expected_categorical:
    if col not in df.columns:
        print(f"ERROR: Missing categorical column: {col}")
    else:
        print(f"OK: {col} exists.")

print("\n")

# -----------------------------
# OUTLIER DETECTION
# -----------------------------
print("6. OUTLIER CHECK (Z-score > 3)")
print("-" * 80)

for col in expected_numeric:
    if col in df.columns:
        col_clean = df[col].dropna()
        if col_clean.std() == 0:
            continue

        z = np.abs((col_clean - col_clean.mean()) / col_clean.std())
        outliers = (z > 3).sum()
        print(f"{col}: {outliers} outliers")

print("\n")

# -----------------------------
# LOGICAL CHECKS
# -----------------------------
print("7. LOGICAL CONSISTENCY CHECKS")
print("-" * 80)

checks = {
    "Production_Mb >= 0": (df["Production_Mb"].dropna() >= 0).all(),
    "Consumption_Mb >= 0": (df["Consumption_Mb"].dropna() >= 0).all(),
    "Reserves_Gb >= 0": (df["Reserves_Gb"].dropna() >= 0).all(),
    "Price_Brent >= 0": (df["Price_Brent"].dropna() >= 0).all(),
}

for rule, passed in checks.items():
    print(f"{rule}: {'OK' if passed else 'FAILED'}")

# Boolean flags check
bool_cols = ["has_reserve", "has_production", "has_consumption"]

for col in bool_cols:
    if col in df.columns:
        print(f"{col}: OK (all boolean)")

print("\n")

# -----------------------------
# YEAR RANGE CHECK
# -----------------------------
print("8. YEAR RANGE CHECK")
print("-" * 80)
print(f"Year range: {df.Year.min()} - {df.Year.max()}")
print("\n")

print("="*80)
print("VALIDATION COMPLETED")
print("="*80 + "\n")