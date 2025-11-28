# robustness.py
"""
Simple two-plot design + minimal robustness check.
Outputs:
  - outputs/plots/global_prod_price_timeseries.png
  - outputs/plots/scatter_lnprod_lnprice_with_fit.png
  - outputs/robustness.txt

How to run:
  python3 robustness.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# ---- CONFIG ----
DATA_PATH = Path("data_clean/master_oil_panel.csv")
OUT_DIR = Path("outputs")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- LOAD DATA ----
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows from {DATA_PATH}")

# ---- FILTER: keep countries with decent coverage ----
min_years = 15
coverage = df.dropna(subset=['Production_Mb', 'Price_Brent']).groupby('Country')['Year'].nunique()
good_countries = coverage[coverage >= min_years].index.tolist()
df = df[df['Country'].isin(good_countries)].copy()
print(f"Using {len(good_countries)} countries with >= {min_years} years of data")

# ---- TRANSFORMATIONS ----
# log transforms (log1p handles zeros)
df['ln_Production'] = np.log1p(df['Production_Mb'])
df['ln_Consumption'] = np.log1p(df['Consumption_Mb'])
df['ln_Price'] = np.log1p(df['Price_Brent'])
df['ln_Reserves'] = np.log1p(df['Reserves_Gb'])

# sort for lagging / differencing
df = df.sort_values(['Country', 'Year']).reset_index(drop=True)

# ---- PLOT 1: Global production (sum) vs mean Brent price (time series) ----
global_prod = df.groupby('Year')['Production_Mb'].sum()
global_price = df.groupby('Year')['Price_Brent'].mean()

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(global_prod.index, global_prod.values, label='Global Production (Mb/year)', linewidth=2)
ax1.set_xlabel('Year')
ax1.set_ylabel('Production (Mb/year)')
ax1.grid(axis='y', alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(global_price.index, global_price.values, color='C1', label='Brent Price (USD)', linewidth=2)
ax2.set_ylabel('Brent Price (USD)')

# legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('Global Production vs Brent Price (mean)')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "global_prod_price_timeseries.png", dpi=150)
plt.close()
print("Saved plot:", PLOTS_DIR / "global_prod_price_timeseries.png")

# ---- PLOT 2: Scatter ln(Production) vs ln(Price) with linear fit (sampled) ----
# sample to keep plot readable
sample = df.dropna(subset=['ln_Production','ln_Price']).sample(n=min(2000, len(df.dropna(subset=['ln_Production','ln_Price']))), random_state=1)

plt.figure(figsize=(8,6))
sns.regplot(data=sample, x='ln_Price', y='ln_Production', scatter_kws={'s':10, 'alpha':0.4}, line_kws={'color':'C1'})
plt.xlabel('ln(Price + 1)')
plt.ylabel('ln(Production + 1)')
plt.title('Scatter: ln(Production) vs ln(Price) (sample) with OLS fit')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scatter_lnprod_lnprice_with_fit.png", dpi=150)
plt.close()
print("Saved plot:", PLOTS_DIR / "scatter_lnprod_lnprice_with_fit.png")

# ---- MINIMAL ROBUSTNESS CHECKS ----
# 1) Fixed effects OLS (country + year dummies)
# Use log variables; interpret ln_Price coef as elasticity (log-log)
prod_fe_df = df.dropna(subset=['ln_Production','ln_Price','ln_Reserves'])
formula_fe = 'ln_Production ~ ln_Price + ln_Reserves + C(Country) + C(Year)'
fe_res = smf.ols(formula_fe, data=prod_fe_df).fit()
fe_res_clustered = fe_res.get_robustcov_results(cov_type='cluster', groups=prod_fe_df['Country'])

# 2) First-difference regression (within-country first difference)
# diff by country removes country fixed effects non-parametrically
df['d_ln_Production'] = df.groupby('Country')['ln_Production'].diff()
df['d_ln_Price'] = df.groupby('Country')['ln_Price'].diff()
df['d_ln_Reserves'] = df.groupby('Country')['ln_Reserves'].diff()

fd_df = df.dropna(subset=['d_ln_Production','d_ln_Price','d_ln_Reserves'])
formula_fd = 'd_ln_Production ~ d_ln_Price + d_ln_Reserves'
fd_res = smf.ols(formula_fd, data=fd_df).fit(cov_type='HC1')  # robust SE (heteroskedasticity-consistent)

# ---- SAVE ROBUSTNESS RESULTS ----
out_txt = OUT_DIR / "robustness.txt"
with open(out_txt, 'w') as f:
    f.write("Robustness check: Price elasticity of production\n")
    f.write("="*72 + "\n\n")
    f.write("1) Fixed Effects OLS (country + year dummies)\n")
    f.write("- model formula: " + formula_fe + "\n\n")
    f.write(fe_res_clustered.summary().as_text())
    f.write("\n\n" + "-"*72 + "\n\n")
    f.write("2) First-difference regression (within-country changes)\n")
    f.write("- model formula: " + formula_fd + "\n\n")
    f.write(fd_res.summary().as_text())
print("Saved robustness results to:", out_txt)

# ---- PRINT SHORT INTERPRETATION TO CONSOLE ----
print("\n--- Short robustness summary ---")

# Get FE price elasticity safely
if hasattr(fe_res_clustered.params, "index"):  # pandas index → named params
    fe_coef = fe_res_clustered.params.get("ln_Price", "NA")
else:  # fallback → raw numpy array
    try:
        price_index = fe_res_clustered.model.exog_names.index("ln_Price")
        fe_coef = fe_res_clustered.params[price_index]
    except Exception:
        fe_coef = "NA"

# FD is always a Series → safe
fd_coef = fd_res.params.get("d_ln_Price", "NA")

print("FE ln_Price coef:", fe_coef)
print("FD d_ln_Price coef:", fd_coef)
print("Results saved in:", out_txt)
