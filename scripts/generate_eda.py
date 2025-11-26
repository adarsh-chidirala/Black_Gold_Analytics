import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
INPUT_FILE = Path("data_clean/master_oil_panel.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)  # Creates the folder if it doesn't exist

# Set plot style for professional look
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# --- LOAD DATA ---
print("📊 Loading Master Dataset...")
df = pd.read_csv(INPUT_FILE)

# --- CHART 1: GLOBAL PRODUCTION vs CONSUMPTION ---
print("   Generating Chart 1: Global Trends...")
# Group by Year to get global totals
global_totals = df.groupby("Year")[["Production_Mb", "Consumption_Mb"]].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(global_totals["Year"], global_totals["Production_Mb"], label="Production", color="black", linewidth=2)
plt.plot(global_totals["Year"], global_totals["Consumption_Mb"], label="Consumption", color="grey", linestyle="--", linewidth=2)
plt.title("Global Oil Production vs. Consumption (1980-2023)")
plt.ylabel("Million Barrels per Year")
plt.xlabel("Year")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_global_trends.png")
plt.close()

# --- CHART 2: OIL PRICES (BRENT) ---
print("   Generating Chart 2: Oil Prices...")
# Prices are constant for a year across all countries, so we take the mean to get 1 value per year
prices = df.groupby("Year")["Price_Brent"].mean().dropna()

plt.figure(figsize=(10, 6))
plt.plot(prices.index, prices.values, color="darkred", linewidth=2)
plt.title("Crude Oil Benchmark Price (Brent)")
plt.ylabel("USD per Barrel")
plt.xlabel("Year")
# Add vertical lines for major events (optional context)
plt.axvline(x=2008, color='black', linestyle=':', alpha=0.5, label="2008 Crisis")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_oil_prices.png")
plt.close()

# --- CHART 3: CORRELATION HEATMAP ---
print("   Generating Chart 3: Correlation...")
# We correlate Global Totals vs Price
global_corr_data = global_totals.merge(prices.rename("Price"), on="Year")
corr_matrix = global_corr_data[["Production_Mb", "Consumption_Mb", "Price"]].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="RdBu", vmin=0, vmax=1, fmt=".2f")
plt.title("Correlation: Global Supply, Demand, and Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "3_correlation_heatmap.png")
plt.close()

# --- CHART 4: GLOBAL R/P RATIO (DEPLETION) ---
print("   Generating Chart 4: Global R/P Ratio...")
# Calculate Global R/P = (Total Reserves / Total Production)
# We calculate this on the aggregate, NOT the average of individual countries
global_reserves = df.groupby("Year")["Reserves_Gb"].sum()
global_prod_for_rp = df.groupby("Year")["Production_Mb"].sum()

# Ratio = (Reserves(Gb) * 1000) / Production(Mb)
global_rp = (global_reserves * 1000) / global_prod_for_rp

plt.figure(figsize=(10, 6))
plt.plot(global_rp.index, global_rp.values, color="black", linewidth=2)
plt.title("Global Reserves-to-Production (R/P) Ratio")
plt.ylabel("Years of Supply Remaining")
plt.xlabel("Year")
# Highlight the data artifact (sparse reporting in recent years)
plt.axvspan(2020, 2023, color='red', alpha=0.1, label="Data Coverage Artifact")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4_global_rp_ratio.png")
plt.close()

# --- CHART 5: TOP 10 PRODUCERS (2022) ---
print("   Generating Chart 5: Top 10 Producers...")
# We use 2022 because it often has more complete reporting than 2023 in this dataset
target_year = 2022
top_producers = df[df["Year"] == target_year].nlargest(10, "Production_Mb")

plt.figure(figsize=(12, 6))
sns.barplot(data=top_producers, x="Production_Mb", y="Country", palette="Blues_d")
plt.title(f"Top 10 Oil Producers ({target_year})")
plt.xlabel("Million Barrels per Year")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "5_top_10_producers.png")
plt.close()

print(f"✅ DONE! All 5 charts saved to: {OUTPUT_DIR}/")