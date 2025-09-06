#%%
# ABOUTME: Residual Institutional Ownership (RIO) predictors following Nagel 2005, Table 2B, 2, 2, 2E
# ABOUTME: RIO_MB, RIO_Disp, RIO_Turnover, RIO_Volatility combining institutional ownership with market-to-book, forecast dispersion, turnover, and volatility
"""
Usage:
    python3 Predictors/ZZ1_RIO_MB_RIO_Disp_RIO_Turnover_RIO_Volatility.py

Inputs:
    - IBES_EPS_Unadj.parquet: IBES forecast data with columns [tickerIBES, time_avail_m, stdev]
    - MSigma_InstitutionalOwnership.parquet: Institutional ownership data
    - SignalMasterTable.parquet: Monthly master table with market cap and other variables
    - MSigma_Vol_m.parquet: Monthly volume data
    - m_crsp.parquet: CRSP monthly returns

Outputs:
    - RIO_MB.csv: RIO quintile for stocks in highest MB quintile
    - RIO_Disp.csv: RIO quintile for stocks in high forecast dispersion quintiles  
    - RIO_Turnover.csv: RIO quintile for stocks in highest turnover quintile
    - RIO_Volatility.csv: RIO quintile for stocks in highest volatility quintile
    
All predictors use residual institutional ownership (RIO) which controls for size effects in institutional holdings
"""

import polars as pl
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.save_standardized import save_predictor
from utils.stata_fastxtile import fastxtile
from utils.asrol import asrol_pl

print("=" * 80)
print("🏗️  ZZ1_RIO_MB_RIO_Disp_RIO_Turnover_RIO_Volatility.py")
print("Generating Real Investment Opportunities (RIO) predictors")
print("=" * 80)

print("📊 Preparing IBES data...")

# Prep IBES data
ibes_eps = pl.read_parquet("../pyData/Intermediate/IBES_EPS_Unadj.parquet")
temp_ibes = ibes_eps.filter(pl.col("fpi") == "1").select(["tickerIBES", "time_avail_m", "stdev"])
print(f"IBES EPS data: {len(temp_ibes):,} observations")

print("📊 Loading main data sources...")

# DATA LOAD
signal_master = pl.read_parquet("../pyData/Intermediate/SignalMasterTable.parquet")
df = signal_master.select(["permno", "tickerIBES", "time_avail_m", "exchcd", "mve_c"])
print(f"SignalMasterTable: {len(df):,} observations")

# Merge all data sources
tr_13f = pl.read_parquet("../pyData/Intermediate/TR_13F.parquet")
df = df.join(tr_13f.select(["permno", "time_avail_m", "instown_perc"]), on=["permno", "time_avail_m"], how="left")

m_compustat = pl.read_parquet("../pyData/Intermediate/m_aCompustat.parquet")
df = df.join(m_compustat.select(["permno", "time_avail_m", "at", "ceq", "txditc"]), on=["permno", "time_avail_m"], how="left")

crsp = pl.read_parquet("../pyData/Intermediate/monthlyCRSP.parquet")
df = df.join(crsp.select(["permno", "time_avail_m", "vol", "shrout", "ret"]), on=["permno", "time_avail_m"], how="left")

df = df.join(temp_ibes, on=["tickerIBES", "time_avail_m"], how="left")

print(f"After merging all data sources: {len(df):,} observations")

print("🔍 Applying size filters...")

# Calculate 20th percentile of NYSE/AMEX market cap for filtering
df = df.with_columns(
    pl.when((pl.col("exchcd") == 1) | (pl.col("exchcd") == 2))
    .then(pl.col("mve_c"))
    .otherwise(None)
    .quantile(0.2)
    .over("time_avail_m")
    .alias("p20_nyse")
)

# Filter out bottom size quintile (at or below NYSE 20th percentile)
df = df.filter(pl.col("mve_c") > pl.col("p20_nyse"))
df = df.drop("p20_nyse")
print(f"After filtering bottom size quintile: {len(df):,} observations")

print("🏛️ Computing Residual Institutional Ownership (RIO)...")

# Compute RIO following Stata's sequential logic
df = df.with_columns(
    pl.when(pl.col("instown_perc").is_null())
    .then(None)
    .otherwise(pl.col("instown_perc") / 100)
    .alias("temp")
)

df = df.with_columns(
    pl.when(pl.col("temp").is_null())
    .then(0.0)
    .otherwise(pl.col("temp"))
    .alias("temp")
)

df = df.with_columns(
    pl.when(pl.col("temp") > 0.9999)
    .then(0.9999)
    .otherwise(pl.col("temp"))
    .alias("temp")
)

df = df.with_columns(
    pl.when(pl.col("temp") < 0.0001)
    .then(0.0001)
    .otherwise(pl.col("temp"))
    .alias("temp")
)

# Calculate RIO
df = df.with_columns(
    (
        (pl.col("temp") / (1 - pl.col("temp"))).log() + 
        23.66 - 
        2.89 * pl.col("mve_c").log() + 
        0.08 * (pl.col("mve_c").log()).pow(2)
    ).alias("RIO")
)

# Create 6-month lag using calendar-based approach
df = df.sort(["permno", "time_avail_m"])

# Convert to pandas for calendar-based lag
df_pandas = df.to_pandas()

# Calculate the exact 6-month lag date for each observation
df_pandas['lag_date'] = df_pandas['time_avail_m'] - pd.DateOffset(months=6)

# Create lookup for RIO values by permno and date
rio_lookup = df_pandas.set_index(['permno', 'time_avail_m'])['RIO']

# Get RIOlag by looking up RIO at lag_date
df_pandas['RIOlag'] = df_pandas.apply(
    lambda row: rio_lookup.get((row['permno'], row['lag_date']), None), 
    axis=1
)

# Drop the temporary lag_date column and convert back to polars
df_pandas = df_pandas.drop(columns=['lag_date'])
df = pl.from_pandas(df_pandas)

# Create RIO quintiles - must convert to pandas for fastxtile
df_pandas = df.to_pandas()
df_pandas['cat_RIO'] = fastxtile(df_pandas, 'RIOlag', by='time_avail_m', n=5)
df = pl.from_pandas(df_pandas)

print("📊 Computing characteristic variables...")

# Forecast dispersion, market-to-book, turnover, volatility sorts
df = df.with_columns(
    pl.when(pl.col("txditc").is_null()).then(0.0).otherwise(pl.col("txditc")).alias("txditc")
)

df = df.with_columns(
    pl.when((pl.col("ceq") + pl.col("txditc")) < 0)
    .then(None)
    .otherwise(pl.col("mve_c") / (pl.col("ceq") + pl.col("txditc")))
    .alias("MB")
)

df = df.with_columns(
    pl.when(pl.col("stdev") > 0)
    .then(pl.col("stdev") / pl.col("at"))
    .otherwise(None)
    .alias("Disp")
)

df = df.with_columns(
    (pl.col("vol") / pl.col("shrout")).alias("Turnover")
)

# Use polars asrol for rolling volatility
df = asrol_pl(
    df,
    group_col='permno',
    time_col='time_avail_m',
    freq='1mo',
    window=12,
    value_col='ret',
    stat='std',
    new_col_name='Volatility',
    min_samples=6
)

# Drop rows missing mve_c (asrol may fill gaps too aggressively)
df = df.filter(pl.col("mve_c").is_not_null())

print("🏷️ Creating characteristic quintiles and RIO interactions...")

# Create characteristic quintiles and RIO interactions
variables = ["MB", "Disp", "Volatility", "Turnover"]

# Convert to pandas for fastxtile operations
df_pandas = df.to_pandas()

for var in variables:
    df_pandas[f'cat_{var}'] = fastxtile(df_pandas, var, by='time_avail_m', n=5)
    df_pandas[f'RIO_{var}'] = df_pandas['cat_RIO'].where(df_pandas[f'cat_{var}'] == 5)

df = pl.from_pandas(df_pandas)

# Patch for Dispersion
df = df.with_columns(
    pl.when((pl.col("cat_Disp") >= 4) & (pl.col("cat_Disp").is_not_null()))
    .then(pl.col("cat_RIO"))
    .otherwise(pl.col("RIO_Disp"))
    .alias("RIO_Disp")
)

print("💾 Saving RIO predictors...")

# Save all RIO predictors
rio_predictors = ["RIO_MB", "RIO_Disp", "RIO_Turnover", "RIO_Volatility"]

for predictor in rio_predictors:
    result = df.select(["permno", "time_avail_m", predictor])
    valid_result = result.filter(pl.col(predictor).is_not_null())
    
    print(f"Generated {predictor}: {len(valid_result):,} observations")
    if len(valid_result) > 0:
        print(f"  Value distribution:")
        print(valid_result.group_by(predictor).agg(pl.len().alias("count")).sort(predictor))
    
    save_predictor(result, predictor)
    print(f"✅ {predictor}.csv saved successfully")

print("🎉 All RIO predictors completed!")