
import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings

# ---- CONFIG: try common locations ----
POSSIBLE_DATA_DIRS = [Path("data_raw"), Path("/mnt/data"), Path(".")]
DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if p.exists()), Path("."))
OUT_DIR = Path("data_clean")
OUT_DIR_MNT = Path("/mnt/data") / "data_clean" if Path("/mnt/data").exists() else OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_MNT.mkdir(parents=True, exist_ok=True)

FILES = {
    "production": "Oil - Production barrels - Sheet1.csv",
    "consumption": "Oil - Consumption barrels - Sheet1.csv",
    # prefer history or current depending on availability
    "reserves": "Oil - Proved reserves history - Sheet1.csv",
    "prices": "Oil - Spot crude prices - Sheet1.csv"
}

print("Using DATA_DIR:", DATA_DIR.resolve())

# ---- UTILITIES ----
year_regex = re.compile(r'^\s*(1[8-9]\d{2}|20\d{2})\s*(?:\.0+)?\s*$')

def find_header_row_by_years(df, min_year_cols=3, max_rows=20):
    best_i, best_count = 0, 0
    for i, row in df.head(max_rows).iterrows():
        count = sum(bool(year_regex.match(str(v))) for v in row.values)
        if count > best_count:
            best_i, best_count = i, count
    if best_count >= min_year_cols:
        return best_i
    for i, row in df.head(max_rows).iterrows():
        row_str = " ".join(str(x) for x in row.values if pd.notnull(x))
        if re.search(r'\bcountry\b', row_str, flags=re.I) or re.search(r'\byear\b', row_str, flags=re.I):
            return i
    return 0

def robust_read_csv(path):
    try:
        return pd.read_csv(path, header=None, encoding='utf-8', low_memory=False)
    except Exception:
        return pd.read_csv(path, header=None, encoding='latin1', low_memory=False)

def standardize_country(col):
    if pd.isna(col):
        return col
    s = str(col).strip()
    s = re.sub(r'\s+', ' ', s)
    replacements = {
        "US": "United States", "U.S.": "United States", "U.S.A.": "United States",
        "Russian Federation": "Russia", "Viet Nam": "Vietnam", "Korea, Rep.": "South Korea",
        "Korea, Dem. Rep.": "North Korea"
    }
    return replacements.get(s, s)

drop_patterns = [
    r'\btotal\b', r'\bworld\b', r'\boecd\b', r'\beuropean union\b', r'\bopec\b',
    r'\bnon-opec\b', r'\bof which\b', r'other\b', r'\bussr\b', r'aggregate', r'\bsubtotal\b'
]
drop_re = re.compile("|".join(drop_patterns), flags=re.I)

def clean_countries_df(df, country_col='Country'):
    df = df.copy()
    if country_col not in df.columns:
        df = df.rename(columns={df.columns[0]: 'Country'})
        country_col = 'Country'
    df['Country'] = df['Country'].astype(str).map(standardize_country)
    mask = ~df['Country'].str.match(r'^\s*(nan|none)?\s*$', flags=re.I)
    mask &= ~df['Country'].str.contains(drop_re)
    df = df[mask].copy()
    df['Country'] = df['Country'].str.strip()
    return df

def melt_years(df, value_name, id_vars='Country'):
    cols = list(df.columns)
    year_col_map = {}
    for orig in cols:
        s = str(orig).strip()
        m = re.match(r'^(1[8-9]\d{2}|20\d{2})', s)
        if m:
            year_col_map[orig] = int(m.group(1))
        else:
            try:
                if float(s).is_integer():
                    yi = int(float(s))
                    if 1800 <= yi <= 2099:
                        year_col_map[orig] = yi
            except Exception:
                pass
    if not year_col_map:
        raise ValueError("No year columns detected in dataframe. Columns: " + ", ".join(map(str, cols[:20])))
    year_cols_orig = [c for c in cols if c in year_col_map]
    id_vars_final = id_vars if id_vars in df.columns else df.columns[0]
    df_long = df.melt(id_vars=id_vars_final, value_vars=year_cols_orig, var_name='Year_raw', value_name=value_name)
    df_long['Year'] = df_long['Year_raw'].map(lambda x: year_col_map.get(x, None))
    df_long = df_long.drop(columns=['Year_raw'])
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')
    df_long = df_long.dropna(subset=['Year']).reset_index(drop=True)
    df_long['Year'] = df_long['Year'].astype(int)
    return df_long

def detect_and_convert_production_units(series):
    s = series.dropna().abs()
    if s.empty:
        return series, "unknown"
    med = float(s.median())
    if med > 1e4:
        converted = series * 365.0 / 1e6
        return converted, "barrels/day -> Mb/year"
    elif 1e-3 < med <= 1e4:
        if med > 100:
            converted = series * 365.0 / 1e6
            return converted, "likely barrels/day -> Mb/year"
        else:
            return series, "assumed Mb/year (no conversion)"
    else:
        return series, "assumed Mb/year (tiny median)"

def detect_and_convert_reserves_units(series):
    s = series.dropna().abs()
    if s.empty:
        return series, "unknown"
    med = float(s.median())
    if med > 1e4:
        converted = series / 1000.0
        return converted, "Mb -> Gb"
    elif med > 1000:
        converted = series / 1e6
        return converted, "barrels -> Gb (divided by 1e6)"
    else:
        return series, "assumed Gb"

def load_panel_sheet(key, filename):
    path = DATA_DIR / filename
    if not path.exists():
        cand = [p for p in DATA_DIR.iterdir() if filename.lower() in p.name.lower()]
        path = cand[0] if cand else None
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"{key} file not found: {filename} in {DATA_DIR}")
    df_raw = robust_read_csv(path)
    header_row = find_header_row_by_years(df_raw, min_year_cols=3, max_rows=25)
    try:
        df = pd.read_csv(path, header=header_row, low_memory=False)
    except Exception:
        header = df_raw.iloc[header_row].tolist()
        df = df_raw.copy()
        df.columns = header
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns[0].lower() not in ['country', 'economy', 'area', 'entity']:
        for i, col in enumerate(df.columns[:3]):
            colvals = df[col].astype(str).str.lower().head(10).tolist()
            if any(re.search(r'\b(country|economy|area|entity|state|region)\b', v) for v in colvals):
                df = df.rename(columns={col: 'Country'})
                break
        else:
            df = df.rename(columns={df.columns[0]: 'Country'})
    df = clean_countries_df(df, country_col='Country')
    return df

# Process production
print("\\--- Processing Production ---")
df_prod = load_panel_sheet('production', FILES['production'])
prod_long = melt_years(df_prod, 'Production_Raw', id_vars='Country')
prod_long['Country'] = prod_long['Country'].map(standardize_country)
prod_long['Production_Mb'], prod_unit = detect_and_convert_production_units(prod_long['Production_Raw'])
print("Production unit heuristic:", prod_unit)
prod_long.loc[prod_long['Production_Mb'] < 0, 'Production_Mb'] = np.nan

# Process consumption
print("\\--- Processing Consumption ---")
df_cons = load_panel_sheet('consumption', FILES['consumption'])
cons_long = melt_years(df_cons, 'Consumption_Raw', id_vars='Country')
cons_long['Country'] = cons_long['Country'].map(standardize_country)
cons_long['Consumption_Mb'], cons_unit = detect_and_convert_production_units(cons_long['Consumption_Raw'])
print("Consumption unit heuristic:", cons_unit)
cons_long.loc[cons_long['Consumption_Mb'] < 0, 'Consumption_Mb'] = np.nan

# Process reserves
print("\\--- Processing Reserves ---")
res_candidates = [FILES['reserves'], "Oil - Proved reserves - Sheet1.csv", "Oil - Proved reserves - Sheet1.csv"]
found = False
for fname in res_candidates:
    try:
        df_res = load_panel_sheet('reserves', fname)
        print("Using reserves file:", fname)
        found = True
        break
    except FileNotFoundError:
        continue
if not found:
    raise FileNotFoundError("Reserves file not found among expected candidates. Files in data dir: " + ", ".join([p.name for p in DATA_DIR.iterdir()]))
res_long = melt_years(df_res, 'Reserves_Raw', id_vars='Country')
res_long['Country'] = res_long['Country'].map(standardize_country)
res_long['Reserves_Gb'], res_unit = detect_and_convert_reserves_units(res_long['Reserves_Raw'])
print("Reserves unit heuristic:", res_unit)
res_long.loc[res_long['Reserves_Gb'] < 0, 'Reserves_Gb'] = np.nan

# Process prices (transpose-supporting)
print("\\--- Processing Prices ---")
prices_path = DATA_DIR / FILES['prices']
if not prices_path.exists():
    cand = [p for p in DATA_DIR.iterdir() if 'price' in p.name.lower() or 'spot' in p.name.lower()]
    prices_path = cand[0] if cand else prices_path
if not prices_path.exists():
    print("Warning: prices file not found; proceeding without prices.")
    df_prices = pd.DataFrame(columns=['Year','Price_Brent'])
else:
    try:
        df_prices_raw = pd.read_csv(prices_path, header=0, low_memory=False)
    except Exception:
        df_prices_raw = pd.read_csv(prices_path, header=None, low_memory=False)
    df_prices_raw.columns = [str(c).strip() for c in df_prices_raw.columns]
    first_col_sample = df_prices_raw.iloc[:,0].astype(str).head(10).tolist()
    if any(year_regex.match(x) for x in first_col_sample):
        possible_brent = [c for c in df_prices_raw.columns if 'brent' in c.lower()]
        if possible_brent:
            df_prices = df_prices_raw[[df_prices_raw.columns[0], possible_brent[0]]].rename(columns={df_prices_raw.columns[0]:'Year', possible_brent[0]:'Price_Brent'})
            df_prices['Year'] = pd.to_numeric(df_prices['Year'], errors='coerce').astype('Int64')
            df_prices['Price_Brent'] = pd.to_numeric(df_prices['Price_Brent'], errors='coerce')
        else:
            df_prices = df_prices_raw.iloc[:, :2].copy()
            df_prices.columns = ['Year', 'Price_Brent']
            df_prices['Year'] = pd.to_numeric(df_prices['Year'], errors='coerce').astype('Int64')
            df_prices['Price_Brent'] = pd.to_numeric(df_prices['Price_Brent'], errors='coerce')
    else:
        if 'Crude Type' not in df_prices_raw.columns and 'Crude_Type' not in df_prices_raw.columns:
            df_prices_raw = df_prices_raw.rename(columns={df_prices_raw.columns[0]:'Crude_Type'})
        df_prices_T = df_prices_raw.set_index('Crude_Type').T.reset_index().rename(columns={'index':'Year'})
        df_prices_T.columns = [str(c).strip() for c in df_prices_T.columns]
        brent_cols = [c for c in df_prices_T.columns if 'brent' in c.lower()]
        if not brent_cols:
            brent_cols = [c for c in df_prices_T.columns if 'uk' in c.lower() or 'dtd' in c.lower()]
        if not brent_cols:
            print("Warning: Brent-like column not found in price data. Columns:", df_prices_T.columns.tolist()[:20])
            df_prices = pd.DataFrame(columns=['Year','Price_Brent'])
        else:
            col = brent_cols[0]
            df_prices = df_prices_T[['Year', col]].rename(columns={col:'Price_Brent'})
            df_prices['Year'] = df_prices['Year'].astype(str).str.replace(r'\\,?\\.0+$','', regex=True)
            df_prices['Year'] = pd.to_numeric(df_prices['Year'], errors='coerce').astype('Int64')
            df_prices['Price_Brent'] = pd.to_numeric(df_prices['Price_Brent'], errors='coerce')

# ---- MERGE ----
print("\\--- Merging datasets ---")
for df in [prod_long, cons_long, res_long]:
    df['Country'] = df['Country'].astype(str).str.strip()

prod_small = prod_long[['Country','Year','Production_Mb']].copy()
cons_small = cons_long[['Country','Year','Consumption_Mb']].copy()
res_small  = res_long[['Country','Year','Reserves_Gb']].copy()

master = pd.merge(prod_small, cons_small, on=['Country','Year'], how='outer', indicator=True)
print("After prod-cons merge: rows", master.shape[0], "prod only:", (master['_merge']=='left_only').sum(), "cons only:", (master['_merge']=='right_only').sum())
master = master.drop(columns=['_merge'])
master = pd.merge(master, res_small, on=['Country','Year'], how='outer', indicator=True)
print("After adding reserves: rows", master.shape[0], "new only prod/cons:", (master['_merge']=='left_only').sum(), "res only:", (master['_merge']=='right_only').sum())
master = master.drop(columns=['_merge'])
if not (('df_prices' in globals()) and (not df_prices.empty)):
    master['Price_Brent'] = np.nan
else:
    master = pd.merge(master, df_prices[['Year','Price_Brent']].drop_duplicates(subset=['Year']), on='Year', how='left')

for col in ['Production_Mb','Consumption_Mb','Reserves_Gb']:
    master[col] = pd.to_numeric(master[col], errors='coerce')

master['R_to_P_Years'] = np.where(master['Production_Mb']>0, (master['Reserves_Gb'] * 1000.0) / master['Production_Mb'], np.nan)
master.replace([np.inf, -np.inf], np.nan, inplace=True)
master.loc[master['R_to_P_Years'] > 1e6, 'R_to_P_Years'] = np.nan

master['has_reserve'] = master['Reserves_Gb'].notna()
master['has_production'] = master['Production_Mb'].notna()
master['has_consumption'] = master['Consumption_Mb'].notna()
master['coverage_pct'] = master[['has_reserve','has_production','has_consumption']].mean(axis=1)

float_cols = master.select_dtypes(include=['float64','float32']).columns
master[float_cols] = master[float_cols].round(4)

# ---- SAVE ----
out_path = OUT_DIR / "master_oil_panel.csv"
master.to_csv(out_path, index=False)
out_path_mnt = OUT_DIR_MNT / "master_oil_panel.csv"
master.to_csv(out_path_mnt, index=False)

print(f"\\✅ Saved cleaned master panel to: {out_path.resolve()}")
if OUT_DIR_MNT != OUT_DIR:
    print(f"✅ Also saved to: {out_path_mnt.resolve()}")

print("\\--- Quick audit ---")
print("Rows:", len(master))
try:
    ymin = int(master['Year'].dropna().astype(int).min())
    ymax = int(master['Year'].dropna().astype(int).max())
    print("Years covered:", ymin, "-", ymax)
except Exception:
    print("Years covered: unknown")
print("Unique countries:", master['Country'].nunique())
print("\\Missingness (%):")
print((master[['Production_Mb','Consumption_Mb','Reserves_Gb','Price_Brent']].isna().mean()*100).round(2))

print("\\Top 5 rows with prices:")
try:
    print(master.dropna(subset=['Price_Brent']).sort_values(['Year','Country']).head(5).to_string(index=False))
except Exception:
    print("No price rows available.")

rp_issues = master[(master['R_to_P_Years'].isna()) & (master['Reserves_Gb'].notna()) & (master['Production_Mb'].notna())]
print("\\Rows with Reserves and Production present but R/P failed (count):", len(rp_issues))

print("\\Sample R/P extremes (non-null):")
try:
    display = None
    print(master.dropna(subset=['R_to_P_Years']).sort_values('R_to_P_Years').head(3).to_string(index=False))
    print(master.dropna(subset=['R_to_P_Years']).sort_values('R_to_P_Years', ascending=False).head(3).to_string(index=False))
except Exception:
    pass

