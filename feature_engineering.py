import pandas as pd
import numpy as np
import holidays
import warnings
from astral import LocationInfo
from astral.sun import sun

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Population-based weights for national temperature aggregate
# Source: INE 2016 municipal census (closest mid-period estimate)
CITY_WEIGHTS = {
    'Madrid':    0.47,
    'Barcelona': 0.24,
    'Valencia':  0.12,
    'Seville':   0.10,
    'Bilbao':    0.07,
}

# Generation columns that are ACTUALS → not available at prediction time.
# Strategy: apply lag_24 (safe for day-ahead) and lag_168 (same hour last week).
# Columns that are already forecast variables are dropped entirely in load_and_clean_data.
GENERATION_COLS = [
    'generation biomass',
    'generation fossil brown coal/lignite',
    'generation fossil gas',
    'generation fossil hard coal',
    'generation fossil oil',
    'generation hydro pumped storage consumption',
    'generation hydro run-of-river and poundage',
    'generation hydro water reservoir',
    'generation nuclear',
    'generation other',
    'generation other renewable',
    'generation solar',
    'generation waste',
    'generation wind onshore',
]

# Cross-border flow columns (actuals, same leakage issue)
FLOW_COLS = [
    'total load actual',        # <-- target, handled separately
    'price day ahead',          # known D-1 from market clearing at ~12:00 CET
]


# ---------------------------------------------------------------------------
# STEP 1 — LOAD & CLEAN
# ---------------------------------------------------------------------------

def load_and_clean_data(
    energy_path: str = "data/energy_dataset.csv",
    weather_path: str = "data/weather_features.csv",
):
    print("--- 1. LOADING AND CLEANING DATA ---")

    # ── Energy ────────────────────────────────────────────────────────────
    print("Processing Energy Dataset...")
    df_energy = pd.read_csv(energy_path)

    # Drop forecast variables (leakage) and structurally empty columns
    cols_to_drop = [
        'forecast wind offshore eday ahead',
        'total load forecast',          # kept separately in evaluate.py for TSO benchmark
        'forecast solar day ahead',
        'forecast wind onshore day ahead',
        'generation hydro pumped storage aggregated',
        'generation marine',
        'generation wind offshore',
        'generation fossil peat',
        'generation fossil oil shale',
        'generation fossil coal-derived gas',
        'generation geothermal',
    ]
    df_energy = df_energy.drop(columns=cols_to_drop, errors='ignore')

    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
    df_energy = df_energy.set_index('time').sort_index()
    df_energy.interpolate(method='time', limit_direction='forward', inplace=True)

    # ── Weather ───────────────────────────────────────────────────────────
    print("Processing Weather Dataset...")
    df_weather = pd.read_csv(weather_path)

    # Cast int64 → float64 before any arithmetic
    for col in df_weather.select_dtypes(include=[np.int64]).columns:
        df_weather[col] = df_weather[col].astype(np.float64)

    df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True)
    df_weather = df_weather.drop(columns=['dt_iso']).set_index('time')

    cities = ['Valencia', 'Madrid', ' Barcelona', 'Bilbao', 'Seville']
    city_dataframes = []

    for city in cities:
        df_city = df_weather[df_weather['city_name'] == city].copy()
        df_city = df_city[~df_city.index.duplicated(keep='first')]

        # Kelvin → Celsius
        for tcol in ['temp', 'temp_min', 'temp_max']:
            df_city[tcol] = df_city[tcol] - 273.15

        # Pressure outliers (hPa)
        df_city.loc[~df_city['pressure'].between(900, 1060), 'pressure'] = np.nan
        df_city['pressure'] = df_city['pressure'].interpolate(method='time')

        # Wind speed outliers
        df_city.loc[df_city['wind_speed'] > 50, 'wind_speed'] = np.nan
        df_city['wind_speed'] = df_city['wind_speed'].interpolate(method='time')

        drop_weather = ['rain_3h', 'weather_main', 'weather_description',
                        'weather_icon', 'city_name']
        df_city.drop(columns=[c for c in drop_weather if c in df_city.columns],
                     inplace=True)

        city_dataframes.append((city.strip(), df_city))

    print("Data loading and cleaning complete.\n")
    return df_energy, city_dataframes


# ---------------------------------------------------------------------------
# STEP 2 — FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def merge_and_engineer_features(df_energy, city_dataframes):
    print("--- 2. FEATURE ENGINEERING ---")

    # ── 2a. Merge weather cities ──────────────────────────────────────────
    print("Merging city weather datasets...")
    df_weather_all = pd.DataFrame(index=df_energy.index)

    for city_name, df_city in city_dataframes:
        clean_name = city_name.lower()
        df_c = (df_city[~df_city.index.duplicated(keep='first')]
                .reindex(df_energy.index)
                .ffill().bfill())
        df_c.columns = [f"{col}_{clean_name}" for col in df_c.columns]
        df_weather_all = df_weather_all.join(df_c)

    df = df_energy.join(df_weather_all, how='left')
    df.interpolate(method='time', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # ── 2b. LAG generation & price columns (core leakage fix) ─────────────
    # At day-ahead prediction time (issued ~12:00 CET for next 24h), the most
    # recent confirmed actual generation values are from the PREVIOUS day.
    # → lag_24: "same hour yesterday" — always safe for day-ahead
    # → lag_168: "same hour last week" — captures weekly dispatch pattern
    #
    # price day ahead IS known at prediction time (market clears D-1 ~12:00),
    # so it could in theory be used as-is. We lag it anyway for robustness,
    # since the market price for the EXACT hour being predicted is borderline.
    print("Applying lag_24 and lag_168 to all generation & price columns...")

    gen_cols_present = [c for c in GENERATION_COLS if c in df.columns]

    for col in gen_cols_present:
        df[f'{col}_lag_24']  = df[col].shift(24)
        df[f'{col}_lag_168'] = df[col].shift(168)
        df.drop(columns=[col], inplace=True)   # remove raw (leaky) version

    # Price day ahead: lag_24 only (market result from yesterday)
    if 'price day ahead' in df.columns:
        df['price_lag_24'] = df['price day ahead'].shift(24)
        df.drop(columns=['price day ahead'], inplace=True)

    # ── 2c. Target lags (safe by construction) ────────────────────────────
    print("Generating safe target lags...")
    df['load_lag_24']  = df['total load actual'].shift(24)
    df['load_lag_48']  = df['total load actual'].shift(48)
    df['load_lag_168'] = df['total load actual'].shift(168)

    df['load_rolling_mean_24h'] = df['total load actual'].shift(24).rolling(24).mean()
    df['load_rolling_mean_7d']  = df['total load actual'].shift(24).rolling(168).mean()
    df['load_rolling_std_7d']   = df['total load actual'].shift(24).rolling(168).std()

    # ── 2d. National temperature aggregate ───────────────────────────────
    print("Building population-weighted national temperature...")
    df['temp_national'] = sum(
        df[f'temp_{city.lower()}'] * w for city, w in CITY_WEIGHTS.items()
    )
    df['temp_national_sq'] = df['temp_national'] ** 2

    # Heating / Cooling Degree Hours — nonlinear HVAC proxy
    # These are far more predictive than raw temperature for demand
    df['HDH'] = (18.0 - df['temp_national']).clip(lower=0)   # heating demand
    df['CDH'] = (df['temp_national'] - 21.0).clip(lower=0)   # cooling demand

    # Per-city temperature lags
    for city_name, _ in city_dataframes:
        cn = city_name.lower()
        tcol = f'temp_{cn}'
        if tcol in df.columns:
            df[f'{tcol}_lag_24']  = df[tcol].shift(24)
            df[f'{tcol}_lag_168'] = df[tcol].shift(168)

    # ── 2e. Calendar features ─────────────────────────────────────────────
    print("Creating calendar and cyclical features...")
    df['hour']  = df.index.hour
    df['month'] = df.index.month

    # day_type: 0=weekday, 1=Saturday, 2=Sunday
    df['day_type'] = df.index.dayofweek.map({0:0,1:0,2:0,3:0,4:0,5:1,6:2})

    # Cyclical encodings
    df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Week of year (cyclic) — captures annual seasonality more smoothly than month
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # ── 2f. Holiday features ──────────────────────────────────────────────
    print("Adding national, regional and bridge-day holiday flags...")
    years_list = list(range(2014, 2020))

    es_nat = holidays.Spain(years=years_list)
    es_mad = holidays.Spain(years=years_list, prov='MD')
    es_cat = holidays.Spain(years=years_list, prov='CT')
    es_val = holidays.Spain(years=years_list, prov='VC')
    es_and = holidays.Spain(years=years_list, prov='AN')
    es_bas = holidays.Spain(years=years_list, prov='PV')

    # Base holiday flags (all known in advance — no leakage)
    df['is_holiday']           = (df.index.map(lambda x: x in es_nat)).astype(int)
    df['is_holiday_madrid']    = (df.index.map(lambda x: x in es_mad)).astype(int)
    df['is_holiday_barcelona'] = (df.index.map(lambda x: x in es_cat)).astype(int)
    df['is_holiday_valencia']  = (df.index.map(lambda x: x in es_val)).astype(int)
    df['is_holiday_seville']   = (df.index.map(lambda x: x in es_and)).astype(int)
    df['is_holiday_bilbao']    = (df.index.map(lambda x: x in es_bas)).astype(int)

    # "Tomorrow is a holiday" — legitimate forward-looking calendar knowledge
    df['next_day_is_holiday'] = df['is_holiday'].shift(-24, fill_value=0)

    # Bridge days ("puente"): holiday on Tue/Thu → Mon/Fri is a de-facto day off
    # This creates weekend-like demand patterns the model won't otherwise catch
    holiday_dates = set(pd.Timestamp(d, tz='UTC').date() for d in es_nat.keys())

    def _is_bridge(ts):
        d = ts.date()
        dow = ts.dayofweek   # 0=Mon … 6=Sun
        if dow == 0:  # Monday → check Tuesday
            return int((d + pd.Timedelta(days=1)) in holiday_dates)
        if dow == 4:  # Friday → check Thursday
            return int((d - pd.Timedelta(days=1)) in holiday_dates)
        return 0

    df['is_bridge_day'] = df.index.map(_is_bridge)

    # ── 2g. Solar features (deterministic, zero leakage) ──────────────────
    print("Adding sunrise/sunset and day-length features...")
    madrid = LocationInfo("Madrid", "Spain", "Europe/Madrid", 40.4168, -3.7038)

    _solar_cache = {}

    def _get_solar(date_key):
        if date_key not in _solar_cache:
            s = sun(madrid.observer, date=date_key)
            _solar_cache[date_key] = s
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
    df['day_length_h'] = day_length   # proxy for lighting demand and solar output potential

    # Is it daylight right now? (binary — useful for solar generation lag features)
    df['is_daylight'] = (
        (df['hour'] >= df['sunrise_hour'].astype(int)) &
        (df['hour'] <  df['sunset_hour'].astype(int))
    ).astype(int)

    # ── 2h. Final cleanup ─────────────────────────────────────────────────
    print("Dropping rows with NaNs from lag initialisation (first 8 days)...")
    df_final = df.dropna().copy()

    print(f"✅ Feature engineering complete. Final shape: {df_final.shape}")
    print(f"   Columns: {df_final.shape[1]} | Rows: {df_final.shape[0]:,}")
    return df_final