import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def temporal_train_test_split(df, target_col='total load actual', cutoff_date='2018-01-01 00:00:00+00:00'):
    print(f"Splitting data at cutoff date: {cutoff_date}")
    
    features = [col for col in df.columns if col != target_col]
    
    X = df[features]
    y = df[target_col]
    
    X_train = X.loc[X.index < cutoff_date]
    y_train = y.loc[y.index < cutoff_date]
    
    X_test = X.loc[X.index >= cutoff_date]
    y_test = y.loc[y.index >= cutoff_date]
    
    print(f"  Training set: {X_train.shape[0]:,} hours ({X_train.index.min().date()} to {X_train.index.max().date()})")
    print(f"  Testing set:  {X_test.shape[0]:,} hours ({X_test.index.min().date()} to {X_test.index.max().date()})")
    
    return X_train, X_test, y_train, y_test


def evaluate_tso_baseline(y_test, energy_csv_path="data/energy_dataset.csv"):
    print("\nCalculating official TSO baseline metrics...")
    
    df_orig = pd.read_csv(energy_csv_path)
    df_orig['time'] = pd.to_datetime(df_orig['time'], utc=True)
    df_orig.set_index('time', inplace=True)
    
    tso_forecast = df_orig.loc[y_test.index, 'total load forecast']
    
    if tso_forecast.isna().sum() > 0:
        tso_forecast = tso_forecast.interpolate(method='time')
        
    mae = mean_absolute_error(y_test, tso_forecast)
    mape = mean_absolute_percentage_error(y_test, tso_forecast) * 100
    rmse = np.sqrt(mean_squared_error(y_test, tso_forecast))
    print(f"OFFICIAL TSO BASELINE (2018):")
    print(f"   MAE:  {mae:,.2f} MWh")
    print(f"   MAPE: {mape:.3f} %")
    print(f"   RMSE: {rmse:,.2f} MWh")
    
    metrics = {'MAE': mae, 'MAPE': mape, 'RMSE': rmse}
    return tso_forecast, metrics


def plot_forecast_vs_actual(y_actual, y_pred, model_name="TSO Forecast", window_hours=168):
    plt.figure(figsize=(15, 5))
    plt.plot(y_actual.iloc[:window_hours].index, y_actual.iloc[:window_hours], 
             label='Actual Demand', color='black', linewidth=2)
    plt.plot(y_pred.iloc[:window_hours].index, y_pred.iloc[:window_hours], 
             label=model_name, color='red', linestyle='dashed')
    
    plt.title(f'Actual Demand vs {model_name} (First {window_hours} hours)')
    plt.ylabel('Total Load (MWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()