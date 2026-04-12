import pandas as pd
import numpy as np
import holidays
import warnings
from astral import LocationInfo
from astral.sun import sun

warnings.filterwarnings('ignore')


def load_and_clean_data(dataset_path: str = "data/df_clean.csv"):
    print("--- 1. LOADING AND FILTERING DATA ---")

    df_clean = pd.read_csv(dataset_path)
    print(f"ORIGINAL COLUMNS:{df_clean.columns.tolist()}")

    if 'time' in df_clean.columns:
        df_clean['time'] = pd.to_datetime(df_clean['time'], utc=True)
        df_clean = df_clean.set_index('time').sort_index()

    cols_to_keep = [
        'total load actual', 
        'price day ahead', 
        'price actual', 
        'temp_national'
    ]
    
    final_cols = [c for c in cols_to_keep if c in df_clean.columns]
    df_clean = df_clean[final_cols]

    print(f"Data loading complete. Retained columns: {df_clean.columns.tolist()}\n")
    return df_clean


def engineer_features(df):
    print("--- 2. FEATURE ENGINEERING ---")
    df = df.copy()

    # ── 2a. Price Lagging ─────────────────────────────────────────────────
    if 'price day ahead' in df.columns and 'price actual' in df.columns:
        df['lag_price_24'] = df['price actual'].shift(24)
                
        df.drop(columns=['price day ahead', 'price actual'], inplace=True)

    # ── 2b. System Memory ─────────────────────────────────────────────────
    print("Generating system memory features (lags & rolling)...")
    df['load_lag_24']  = df['total load actual'].shift(24)
    df['load_lag_168'] = df['total load actual'].shift(168)
    
    df['load_rolling_mean_7d']  = df['load_lag_24'].rolling(168).mean()
    df['load_rolling_mean_24h'] = df['load_lag_24'].rolling(24).mean()
    df['load_std_24h'] = df['total load actual'].shift(24).rolling(24).std()
    df['load_diff_24h'] = df['total load actual'].shift(24) - df['total load actual'].shift(48)

    # ── 2c. Smart Weather ─────────────────────────────────────────────────
    print("Building smart weather features (HDH, CDH)...")
    
    df['HDH'] = (15.0 - df['temp_national']).clip(lower=0)   # Frío
    df['CDH'] = (df['temp_national'] - 24.0).clip(lower=0)   # Calor
    df['temp_lag_24'] = df['temp_national'].shift(24)

    # ── 2d. Human Behavior (Calendar Encodings) ───────────────────────────
    print("Creating calendar and cyclical features...")
    df['hour']  = df.index.hour
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek

    df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # ── 2e. Human Behavior (Holidays) ─────────────────────────────────────
    print("Adding holiday and bridge-day flags...")
    years_list = list(range(2014, 2020))
    es_nat = holidays.Spain(years=years_list)

    df['is_holiday'] = df.index.map(lambda x: int(x in es_nat))
    df['next_day_is_holiday'] = df['is_holiday'].shift(-24, fill_value=0)

    holiday_dates = set(pd.Timestamp(d, tz='UTC').date() for d in es_nat.keys())

    def _is_bridge(ts):
        d = ts.date()
        dow = ts.dayofweek   
        if dow == 0:  # Lunes
            return int((d + pd.Timedelta(days=1)) in holiday_dates)
        if dow == 4:  # Viernes
            return int((d - pd.Timedelta(days=1)) in holiday_dates)
        return 0

    df['is_bridge_day'] = df.index.map(_is_bridge)

    # ── 2f. Solar Features ────────────────────────────────────────────────
    print("Adding sunrise/sunset and day-length features...")
    madrid = LocationInfo("Madrid", "Spain", "Europe/Madrid", 40.4168, -3.7038)
    _solar_cache = {}

    def _get_solar(date_key):
        if date_key not in _solar_cache:
            _solar_cache[date_key] = sun(madrid.observer, date=date_key)
        return _solar_cache[date_key]

    local_index = df.index.tz_convert('Europe/Madrid')
    sunrise_hours, sunset_hours, day_length = [], [], []

    for ts in local_index:
        s = _get_solar(ts.date())
        sr = s['sunrise'].astimezone(ts.tzinfo).hour + s['sunrise'].astimezone(ts.tzinfo).minute / 60
        ss = s['sunset'].astimezone(ts.tzinfo).hour  + s['sunset'].astimezone(ts.tzinfo).minute  / 60
        sunrise_hours.append(sr)
        sunset_hours.append(ss)
        day_length.append(ss - sr)

    df['sunrise_hour'] = sunrise_hours
    df['sunset_hour']  = sunset_hours
    df['day_length_h'] = day_length   

    df['is_daylight'] = (
        (df['hour'] >= df['sunrise_hour'].astype(int)) &
        (df['hour'] <  df['sunset_hour'].astype(int))
    ).astype(int)

    # ── 2g. Final cleanup ─────────────────────────────────────────────────
    print("Dropping rows with NaNs from lag initialisation...")
    df.drop(columns=['hour', 'month', 'week_of_year'], inplace=True, errors='ignore')
    
    df_final = df.dropna().copy()

    print("Feature engineering complete.")
    print(f"   Shape inicial: {df.shape}")
    print(f"   Shape final:   {df_final.shape} | Variables mantenidas: {df_final.shape[1]}")
    
    return df_final