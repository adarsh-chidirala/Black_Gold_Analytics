"""
Price Elasticity Pipeline
Single-run script for exploratory analysis and baseline econometric estimation of
price elasticity effects on oil production and consumption.
Outputs:
 - results/plots/global_prod_price_timeseries.png
 - results/plots/scatter_lnprod_lnprice.png
 - results/plots/resid_hist_prod.png
 - results/ols_production_price.txt
 - results/ols_consumption_price.txt
 - results/diagnostics.txt
 - results/sample_transformed_prod.csv

How to run:
    python3 scripts/price_elasticity_pipeline.py

Requirements:
    pandas, numpy, matplotlib, seaborn, statsmodels
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path("data_clean/master_oil_panel.csv")
PLOTS_DIR = Path("results/plots")
OUT_DIR = Path("results")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print("Loaded data:", DATA_PATH, "rows:", len(df))

# Basic preprocessing for econometric analysis
# Keep only countries with reasonable coverage (>= 15 years of production+price data)
min_years = 15
count_years = df.dropna(subset=['Production_Mb','Price_Brent']).groupby('Country')['Year'].nunique()
good_countries = count_years[count_years >= min_years].index.tolist()
print(f"Countries with >= {min_years} years of production+price: {len(good_countries)}")

df = df[df['Country'].isin(good_countries)].copy()

# Create log-level variables (use log1p to handle zeros)
df['ln_Production'] = np.log1p(df['Production_Mb'])
df['ln_Consumption'] = np.log1p(df['Consumption_Mb'])
df['ln_Price'] = np.log1p(df['Price_Brent'])
df['ln_Reserves'] = np.log1p(df['Reserves_Gb'])

# Create lags of price (by country)
df = df.sort_values(['Country','Year'])
df['Price_lag1'] = df.groupby('Country')['ln_Price'].shift(1)
df['Price_lag2'] = df.groupby('Country')['ln_Price'].shift(2)
df['Prod_lag1'] = df.groupby('Country')['ln_Production'].shift(1)

# Keep rows with required variables for baseline regressions
base_prod = df.dropna(subset=['ln_Production','ln_Price','Price_lag1','ln_Reserves'])
base_cons = df.dropna(subset=['ln_Consumption','ln_Price','Price_lag1','ln_Reserves'])

# Quick global time series plots
plt.figure(figsize=(10,5))
global_prod = df.groupby('Year')['Production_Mb'].sum()
global_price = df.groupby('Year')['Price_Brent'].mean()
ax = global_prod.plot(label='Global Production (Mb)', legend=True)
ax2 = ax.twinx()
global_price.plot(ax=ax2, color='C1', label='Brent Price (USD)', legend=True)
ax.set_ylabel('Production (Mb/year)')
ax2.set_ylabel('Price (USD)')
plt.title('Global Production vs Brent Price (mean)')
plt.savefig(PLOTS_DIR / "global_prod_price_timeseries.png", bbox_inches='tight')
plt.close()

# Scatter (country-year) production vs price (log-log sample)
plt.figure(figsize=(8,6))
sample_size = min(2000, len(base_prod))
sns.scatterplot(data=base_prod.sample(sample_size, random_state=1),
                x='ln_Price', y='ln_Production', alpha=0.5)
plt.title('Scatter: ln(Production) vs ln(Price) (sample)')
plt.savefig(PLOTS_DIR / "scatter_lnprod_lnprice.png", bbox_inches='tight')
plt.close()

# Baseline OLS with country and year fixed effects via dummies
print("Running OLS: ln(Production) ~ ln_Price + Price_lag1 + ln_Reserves + Prod_lag1 + FE country + FE year")
formula_prod = 'ln_Production ~ ln_Price + Price_lag1 + ln_Reserves + Prod_lag1 + C(Country) + C(Year)'

# Clean data before model fitting
required_cols = ['ln_Production', 'ln_Price', 'Price_lag1', 'ln_Reserves', 'Prod_lag1', 'Country', 'Year']
base_prod_clean = base_prod.dropna(subset=required_cols)

# Fit the model using the cleaned data
model_prod = smf.ols(formula_prod, data=base_prod_clean).fit(
    cov_type='cluster', cov_kwds={'groups': base_prod_clean['Country']}
)
# model_prod = smf.ols(formula_prod, data=base_prod).fit(cov_type='cluster', cov_kwds={'groups': base_prod['Country']})
with open(OUT_DIR / "ols_production_price.txt", "w") as f:
    f.write(model_prod.summary().as_text())

# Consumption regression
print("Running OLS: ln(Consumption) ~ ln_Price + Price_lag1 + ln_Reserves + FE country + FE year")
formula_cons = 'ln_Consumption ~ ln_Price + Price_lag1 + ln_Reserves + C(Country) + C(Year)'
model_cons = smf.ols(formula_cons, data=base_cons).fit(cov_type='cluster', cov_kwds={'groups': base_cons['Country']})
with open(OUT_DIR / "ols_consumption_price.txt", "w") as f:
    f.write(model_cons.summary().as_text())

# Save diagnostics summary
with open(OUT_DIR / "diagnostics.txt", "w") as f:
    f.write("Baseline Price Elasticity Diagnostics\n\n")
    f.write("Production model summary:\n\n")
    f.write(model_prod.summary().as_text())
    f.write("\n\nConsumption model summary:\n\n")
    f.write(model_cons.summary().as_text())

print("Saved OLS results to results/")

# Quick interpretation printed to console
print("\n--- Quick interpretation ---")
print("Production elasticity estimate (ln_Price):", model_prod.params.get('ln_Price', np.nan))
print("Consumption elasticity estimate (ln_Price):", model_cons.params.get('ln_Price', np.nan))

# Plot residuals for production model
resid = model_prod.resid
plt.figure(figsize=(8,4))
plt.hist(resid.dropna(), bins=50)
plt.title('Residual distribution: Production model')
plt.savefig(PLOTS_DIR / "resid_hist_prod.png", bbox_inches='tight')
plt.close()

# Save a small sample of transformed data for inspection
base_prod[['Country','Year','Production_Mb','Price_Brent','ln_Production','ln_Price','Price_lag1','ln_Reserves']].sample(20).to_csv(OUT_DIR / "sample_transformed_prod.csv", index=False)

print("Plots saved to:", PLOTS_DIR)
print("Results saved to:", OUT_DIR)
print("Script finished.")
