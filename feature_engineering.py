import os
import pandas as pd
import numpy as np
import holidays
import warnings

warnings.filterwarnings('ignore')


def load_and_clean_data(energy_path="data/energy_dataset.csv", weather_path="data/weather_features.csv"):
    print("--- 1. LOADING AND CLEANING DATA ---")
    
    # --- Energy Dataset Processing ---
    print("Processing Energy Dataset...")
    df_energy = pd.read_csv(energy_path)
    
    # Drop forecast variables (to avoid data leakage) and uninformative/empty columns
    cols_to_drop = [
        'forecast wind offshore eday ahead', 'total load forecast', 
        'forecast solar day ahead', 'forecast wind onshore day ahead',
        'generation hydro pumped storage aggregated', 'generation marine',
        'generation wind offshore', 'generation fossil peat',
        'generation fossil oil shale', 'generation fossil coal-derived gas',
        'generation geothermal'
    ]
    df_energy = df_energy.drop(columns=cols_to_drop, errors='ignore')
    
    # Set temporal index (UTC)
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
    df_energy = df_energy.set_index('time')
    
    # Impute missing values using linear interpolation
    df_energy.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)
    
    # --- Weather Dataset Processing ---
    print("Processing Weather Dataset...")
    df_weather = pd.read_csv(weather_path)
    
    # Fix datatypes and timezone alignment (convert dt_iso to UTC datetime)
    cols = df_weather.select_dtypes(include=[np.int64]).columns
    for col in cols:
        df_weather[col] = df_weather[col].values.astype(np.float64)

    df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True)
    df_weather = df_weather.drop(['dt_iso'], axis=1)
    df_weather = df_weather.set_index('time')
    
    # Split by city
    cities = ['Valencia', 'Madrid', ' Barcelona', 'Bilbao', 'Seville']
    city_dataframes = []
    
    for city in cities:
        df_city = df_weather[df_weather['city_name'] == city].copy()
        df_city.drop_duplicates(inplace=True)
        
        # 1. Convert Kelvin to Celsius
        df_city['temp']     = df_city['temp'] - 273.15
        df_city['temp_min'] = df_city['temp_min'] - 273.15
        df_city['temp_max'] = df_city['temp_max'] - 273.15
        
        # 2. Clean Pressure Outliers (Valid range: 900 - 1060 hPa)
        mask_pressure = ~df_city['pressure'].between(900, 1060)
        df_city.loc[mask_pressure, 'pressure'] = np.nan
        df_city['pressure'] = df_city['pressure'].interpolate(method='time')
        
        # 3. Clean Wind Speed Outliers (Max cap: 50 m/s)
        mask_wind = df_city['wind_speed'] > 50
        df_city.loc[mask_wind, 'wind_speed'] = np.nan
        df_city['wind_speed'] = df_city['wind_speed'].interpolate(method='time')
        
        # 4. Drop inconsistent/categorical columns
        cols_to_drop_weather = ['rain_3h', 'weather_main', 'weather_description', 'weather_icon', 'city_name']
        df_city.drop(columns=[c for c in cols_to_drop_weather if c in df_city.columns], inplace=True)
        
        # Clean up the city name for the dictionary key
        clean_name = city.strip()
        city_dataframes.append((clean_name, df_city))
        
    print("Data loading and cleaning complete.\n")
    return df_energy, city_dataframes


def merge_and_engineer_features(df_energy, city_dataframes):
    print("--- 2. FEATURE ENGINEERING ---")
    print("Merging individual city weather datasets...")
    
    # Initialize a dataframe to hold all city weather data
    df_weather_all = pd.DataFrame(index=df_energy.index)
    
    # Iterate through the list of tuples: [('Madrid', df_mad), ('Barcelona', df_bar), ...]
    for city_name, df_city in city_dataframes:
        # Clean the city name to use as a column suffix (e.g., ' Barcelona' -> 'barcelona')
        clean_name = city_name.strip().lower()
        
        # Remove duplicates in the temporal index to avoid merge errors
        df_city_clean = df_city[~df_city.index.duplicated(keep='first')].copy()
        # Reindex to match the energy index exactly
        df_city_clean = df_city_clean.reindex(df_energy.index)
        # Forward and backward fill internal missing values
        df_city_clean = df_city_clean.ffill().bfill()
        
        # Rename columns to maintain identity (e.g., 'temp' -> 'temp_madrid')
        df_city_clean.columns = [f"{col}_{clean_name}" for col in df_city_clean.columns]
        
        # Join to our master weather dataframe
        df_weather_all = df_weather_all.join(df_city_clean)
        
    # Merge Energy + All Cities Weather
    df = df_energy.join(df_weather_all, how='left')
    
    # Final handling of missing values after merging
    df.interpolate(method='time', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    print("Creating temporal variables and cyclical encoding...")
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    # df['dayofweek'] = df.index.dayofweek
    
    # Custom day type mapping: 0 = Weekday (Mon-Fri), 1 = Saturday, 2 = Sunday
    df['day_type'] = df.index.dayofweek.map({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:2})    
    # Cyclical encoding (Sine and Cosine)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    # df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print("Incorporating external covariates: National and Regional Holidays...")
    years_list = [2014, 2015, 2016, 2017, 2018, 2019]
    
    # Spain ISO 3166-2 Province Codes for the holidays library:
    es_holidays = holidays.Spain(years=years_list)
    es_holidays_mad = holidays.Spain(years=years_list, prov='MD')
    es_holidays_cat = holidays.Spain(years=years_list, prov='CT')
    es_holidays_val = holidays.Spain(years=years_list, prov='VC')
    es_holidays_and = holidays.Spain(years=years_list, prov='AN')
    es_holidays_bas = holidays.Spain(years=years_list, prov='PV')
    
    df['is_holiday'] = df.index.map(lambda x: 1 if x in es_holidays else 0)
    df['is_holiday_madrid'] = df.index.map(lambda x: 1 if x in es_holidays_mad else 0)
    df['is_holiday_barcelona'] = df.index.map(lambda x: 1 if x in es_holidays_cat else 0)
    df['is_holiday_valencia'] = df.index.map(lambda x: 1 if x in es_holidays_val else 0)
    df['is_holiday_seville'] = df.index.map(lambda x: 1 if x in es_holidays_and else 0)
    df['is_holiday_bilbao'] = df.index.map(lambda x: 1 if x in es_holidays_bas else 0)
    
    print("Generating safe lags for Day-Ahead prediction...")
    # Lags for the target variable
    df['load_lag_24'] = df['total load actual'].shift(24)
    df['load_lag_48'] = df['total load actual'].shift(48)
    df['load_lag_168'] = df['total load actual'].shift(168)
    
    # Create temperature lags for EACH city
    for city_name, _ in city_dataframes:
        clean_name = city_name.strip().lower()
        temp_col = f'temp_{clean_name}'
        
        if temp_col in df.columns:
            df[f'{temp_col}_lag_24'] = df[temp_col].shift(24)
            df[f'{temp_col}_lag_168'] = df[temp_col].shift(168)
            
    print("Generating safe rolling window statistics...")
    # Calculate rolling mean on the ALREADY 24h-shifted series to avoid Data Leakage
    df['load_rolling_mean_24h'] = df['total load actual'].shift(24).rolling(window=24).mean()
    df['load_rolling_mean_7d'] = df['total load actual'].shift(24).rolling(window=168).mean()
        
    print("Dropping rows with NaNs generated by lags (first 8 days)...")
    df_final = df.dropna().copy()
    
    print(f"✅ Process completed. Final dimensions: {df_final.shape}")
    return df_final