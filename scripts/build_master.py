import pandas as pd
import numpy as np
from pathlib import Path
import re

# --- CONFIGURATION ---
DATA_DIR = Path("data_raw")
OUTPUT_PATH = Path("data_clean/master_oil_panel.csv")

FILES = {
    "production": "Oil - Production barrels - Sheet1.csv",
    "consumption": "Oil - Consumption barrels - Sheet1.csv",
    "reserves": "Oil - Proved reserves history - Sheet1.csv",
    "prices": "Oil - Spot crude prices - Sheet1.csv"
}

# --- UTILITIES ---

def find_header_row(df, keywords=["Country", "Year", "US dollars"]):
    for i, row in df.head(10).iterrows():
        row_str = row.astype(str).str.cat(sep=" ")
        if any(k in row_str for k in keywords):
            return i
    return 0

def clean_and_melt(df, value_name, id_vars="Country"):
    year_cols = [c for c in df.columns if str(c).strip().replace('.','').isdigit()]
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols,
                      var_name="Year", value_name=value_name)
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce").astype('Int64')
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")
    return df_long.dropna(subset=["Year", value_name])

def clean_countries(df):
    drop_patterns = ["Total", "World", "OECD", "European Union", "OPEC", "Non-OPEC", "of which:", "Other", "USSR"]
    pattern = "|".join([re.escape(x) for x in drop_patterns])
    mask = ~df["Country"].astype(str).str.contains(pattern, case=False, regex=True)
    df = df[mask].copy()
    df["Country"] = df["Country"].str.strip()
    df["Country"] = df["Country"].replace({
        "US": "United States", "Russian Federation": "Russia", "Viet Nam": "Vietnam"
    })
    return df

# --- MAIN EXECUTION ---
print("🚀 Starting Data Pipeline (Corrected Transposed Prices)...")

# 1. PRODUCTION
print(f"   Reading: {FILES['production']}")
df_raw = pd.read_csv(DATA_DIR / FILES['production'], header=None)
df_prod = pd.read_csv(DATA_DIR / FILES['production'], header=find_header_row(df_raw, ["Country"]))
df_prod_long = clean_and_melt(clean_countries(df_prod), "Production_Raw")
df_prod_long["Production_Mb"] = df_prod_long["Production_Raw"] * 0.365

# 2. CONSUMPTION
print(f"   Reading: {FILES['consumption']}")
df_raw = pd.read_csv(DATA_DIR / FILES['consumption'], header=None)
df_cons = pd.read_csv(DATA_DIR / FILES['consumption'], header=find_header_row(df_raw, ["Country"]))
df_cons_long = clean_and_melt(clean_countries(df_cons), "Consumption_Raw")
df_cons_long["Consumption_Mb"] = df_cons_long["Consumption_Raw"] * 0.365

# 3. RESERVES
print(f"   Reading: {FILES['reserves']}")
df_raw = pd.read_csv(DATA_DIR / FILES['reserves'], header=None)
df_res = pd.read_csv(DATA_DIR / FILES['reserves'], header=find_header_row(df_raw, ["Country"]))
df_res_long = clean_and_melt(clean_countries(df_res), "Reserves_Gb")

# 4. PRICES (NEW TRANSPOSED LOGIC)
print(f"   Reading: {FILES['prices']}")
# Read header=0 because row 0 contains the years (1972, 1973...)
df_prices_raw = pd.read_csv(DATA_DIR / FILES['prices'], header=0)

# Rename the first column to 'Crude_Type' to verify we have the right file
df_prices_raw.rename(columns={df_prices_raw.columns[0]: 'Crude_Type'}, inplace=True)

# Transpose: Switch rows and columns so Years become rows
df_prices_T = df_prices_raw.set_index('Crude_Type').T
df_prices_T.reset_index(inplace=True)
df_prices_T.rename(columns={'index': 'Year'}, inplace=True)

# Clean up Year column (remove decimals like 1976.0 -> 1976)
df_prices_T['Year'] = pd.to_numeric(df_prices_T['Year'], errors='coerce')

# Select only Year and Brent
# Note: Column name might have spaces, so we strip them
df_prices_T.columns = df_prices_T.columns.str.strip()

if 'Brent' in df_prices_T.columns:
    df_prices = df_prices_T[['Year', 'Brent']].copy()
    df_prices.columns = ['Year', 'Price_Brent']
    # Force Price to numeric (coercing '-' to NaN)
    df_prices['Price_Brent'] = pd.to_numeric(df_prices['Price_Brent'], errors='coerce')
    df_prices = df_prices.dropna(subset=['Year'])
else:
    print("⚠️ WARNING: 'Brent' column not found in transposed price data. Check CSV format.")
    df_prices = pd.DataFrame(columns=['Year', 'Price_Brent'])

# 5. MERGE
print("   Merging datasets...")
master = pd.merge(df_prod_long[["Country", "Year", "Production_Mb"]],
                  df_cons_long[["Country", "Year", "Consumption_Mb"]],
                  on=["Country", "Year"], how="outer")
master = pd.merge(master, df_res_long[["Country", "Year", "Reserves_Gb"]],
                  on=["Country", "Year"], how="outer")
master = pd.merge(master, df_prices, on="Year", how="left")

# 6. CALCULATE METRICS
print("   Calculating R/P Ratio...")
master["R_to_P_Years"] = (master["Reserves_Gb"] * 1000) / master["Production_Mb"]
master.loc[master["Production_Mb"] == 0, "R_to_P_Years"] = np.nan

# --- ROUND ALL FLOAT COLUMNS TO 4 DECIMALS ---
float_cols = master.select_dtypes(include=['float64', 'float32']).columns
master[float_cols] = master[float_cols].round(4)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
master.to_csv(OUTPUT_PATH, index=False)
print(f"✅ DONE! Saved to: {OUTPUT_PATH}")
# Verify we actually have prices now
print("\nSample Data with Prices:")
print(master.dropna(subset=["Price_Brent"]).head(3))