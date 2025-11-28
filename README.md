# 🛢️ Black Gold Analytics  
### *Global Oil Production, Consumption, Prices & Elasticity Analysis (2000–2024)*  

This project builds a fully reproducible data pipeline for constructing, validating, and analyzing a global oil panel dataset from multiple raw sources.  
It includes:

- Clean ETL pipeline  
- Full data validation  
- Econometric models  
- Price elasticity estimation  
- Robustness checks & diagnostics  
- Visualizations and interpretations  

---

## 📁 Project Structure

```
BACK GOLD ANALYTICS/
│
├── data_raw/
├── data_clean/
│    └── master_oil_panel.csv
│
├── outputs/
│    ├── plots/
│    │     ├── global_prod_price_timeseries.png
│    │     └── scatter_lnprod_lnprice_with_fit.png
│    └── robustness/
│          └── robustness.txt
│
├── results/
│    ├── plots/
│    ├── diagnostics.txt
│    ├── ols_production_price.txt
│    └── ols_consumption_price.txt
│
├── scripts/
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Build cleaned dataset

```bash
python3 scripts/build_master.py
```

### 3️⃣ Validate dataset

```bash
python3 scripts/validate_clean_data.py
```

### 4️⃣ Estimate price elasticity

```bash
python3 scripts/price_elasticity_pipeline.py
```

### 5️⃣ Robustness checks

```bash
python3 scripts/robustness.py
```

---

## 📊 Key Plots

### Global Production vs Price  
![Global Production vs Price](outputs/plots/global_prod_price_timeseries.png)

### ln(Production) vs ln(Price)  
![Scatter](outputs/plots/scatter_lnprod_lnprice_with_fit.png)

---

## 📈 Regression Output Summary

- Production elasticity ≈ **0.005–0.02**
- Consumption elasticity ≈ **0.04–0.05**

---

## 🧪 Robustness Summary

See: `outputs/robustness/robustness.txt`
