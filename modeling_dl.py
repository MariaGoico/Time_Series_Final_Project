import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, TCNModel, NBEATSModel, NHiTSModel, TFTModel
from darts.metrics import mae, mape, rmse
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PREPARE DARTS DATA DIRECTLY FROM YOUR X/y SPLIT
# ==========================================
def prepare_darts_from_split(X_train, y_train, X_test, y_test):
    print("--- 1. PREPARING DARTS DATA FROM EXISTING TRAIN/TEST SPLIT ---")
    
    # Combine X and y temporarily to format them for Darts
    df_train = X_train.copy()
    df_train['target'] = y_train.values
    
    df_test = X_test.copy()
    df_test['target'] = y_test.values
    
    # Darts requires timezone-naive datetime indices
    df_train.index = df_train.index.tz_localize(None)
    df_test.index = df_test.index.tz_localize(None)
    
    # Identify Future Covariates (known ahead of time: calendar, holidays)
    future_covariates_cols = [
        col for col in X_train.columns 
        if 'is_holiday' in col or 'sin' in col or 'cos' in col or 'day_type' in col
    ]
    
    # Identify Past Covariates (known up to today: weather, lags, rolling means)
    past_covariates_cols = [
        col for col in X_train.columns 
        if col not in future_covariates_cols and col != 'target'
    ]
    
    print(f"Identified Future Covariates: {len(future_covariates_cols)} columns")
    print(f"Identified Past Covariates: {len(past_covariates_cols)} columns")
    
    # Convert Pandas to Darts TimeSeries
    train_target = TimeSeries.from_dataframe(df_train, value_cols='target')
    test_target = TimeSeries.from_dataframe(df_test, value_cols='target')
    
    train_past = TimeSeries.from_dataframe(df_train, value_cols=past_covariates_cols)
    test_past = TimeSeries.from_dataframe(df_test, value_cols=past_covariates_cols)
    
    train_future = TimeSeries.from_dataframe(df_train, value_cols=future_covariates_cols)
    test_future = TimeSeries.from_dataframe(df_test, value_cols=future_covariates_cols)
    
    # Scaling (Fit on Train, Transform on Test to avoid data leakage)
    print("Scaling TimeSeries data...")
    scaler_target = Scaler()
    train_target_scaled = scaler_target.fit_transform(train_target)
    test_target_scaled = scaler_target.transform(test_target)
    
    scaler_past = Scaler()
    train_past_scaled = scaler_past.fit_transform(train_past)
    test_past_scaled = scaler_past.transform(test_past)
    
    scaler_future = Scaler()
    train_future_scaled = scaler_future.fit_transform(train_future)
    test_future_scaled = scaler_future.transform(test_future)
    
    print("✅ Darts Data Preparation Complete.\n")
    return {
        'train_target_scaled': train_target_scaled,
        'test_target_scaled': test_target_scaled,
        'train_past_scaled': train_past_scaled,
        'test_past_scaled': test_past_scaled,
        'train_future_scaled': train_future_scaled,
        'test_future_scaled': test_future_scaled,
        'test_target_unscaled': test_target, 
        'scaler_target': scaler_target
    }

# ==========================================
# 2. TRAIN & EVALUATE DEEP LEARNING FLEET
# ==========================================
def train_and_evaluate_deep_learning_fleet(darts_data, lookback_hours=168, horizon=24, epochs=5):
    print("="*50)
    print("🚀 LAUNCHING DEEP LEARNING FLEET (6 MODELS)")
    print("="*50)
    
    # Unpack scaled data
    train_tgt = darts_data['train_target_scaled']
    test_tgt = darts_data['test_target_scaled']
    train_past = darts_data['train_past_scaled']
    test_past = darts_data['test_past_scaled']
    full_past = train_past.append(test_past)
    train_future = darts_data['train_future_scaled']
    test_future = darts_data['test_future_scaled']
    full_future = train_future.append(test_future)
    
    # Define models dictionary
    models_dict = {
        'LSTM': {
            'model': RNNModel(model='LSTM', input_chunk_length=lookback_hours, training_length=lookback_hours+horizon, 
                              n_epochs=epochs, random_state=42),
            'uses_future': True
        },
        'GRU': {
            'model': RNNModel(model='GRU', input_chunk_length=lookback_hours, training_length=lookback_hours+horizon, 
                              n_epochs=epochs, random_state=42),
            'uses_future': True
        },
        'TCN': {
            'model': TCNModel(input_chunk_length=lookback_hours, output_chunk_length=horizon, 
                              n_epochs=epochs, random_state=42),
            'uses_future': False
        },
        'N-BEATS': {
            'model': NBEATSModel(input_chunk_length=lookback_hours, output_chunk_length=horizon, 
                                 n_epochs=epochs, random_state=42),
            'uses_future': False
        },
        'N-HiTS': {
            'model': NHiTSModel(input_chunk_length=lookback_hours, output_chunk_length=horizon, 
                                n_epochs=epochs, random_state=42),
            'uses_future': False
        },
        'TFT': {
            'model': TFTModel(input_chunk_length=lookback_hours, output_chunk_length=horizon, 
                              add_relative_index=True, n_epochs=epochs, random_state=42),
            'uses_future': True
        }
    }
    
    results = {}
    predictions_dict = {}
    actuals = darts_data['test_target_unscaled']
    scaler = darts_data['scaler_target']
    
    for name, config in models_dict.items():
        print(f"\n--- 🧠 TRAINING {name} ---")
        model = config['model']
        
        # FIT
        if config['uses_future']:
            model.fit(series=train_tgt, past_covariates=train_past, future_covariates=train_future, verbose=False)
        else:
            model.fit(series=train_tgt, past_covariates=train_past, verbose=False)
            
        print(f"Predicting 2018 with {name} (Historical Forecasts simulating Day-Ahead)...")
        
        # PREDICT
        kwargs = {'past_covariates': full_past}
        if config['uses_future']:
            kwargs['future_covariates'] = full_future
            
        pred_scaled = model.historical_forecasts(
            series=train_tgt.append(test_tgt),
            start=test_tgt.start_time(),
            forecast_horizon=horizon,
            stride=horizon,
            retrain=False,
            verbose=False,
            **kwargs
        )
        
        # Inverse Transform and Calculate Metrics
        pred_unscaled = scaler.inverse_transform(pred_scaled)
        aligned_actuals = actuals.intersect(pred_unscaled)
        
        m_mae = mae(aligned_actuals, pred_unscaled)
        m_mape = mape(aligned_actuals, pred_unscaled)
        m_rmse = rmse(aligned_actuals, pred_unscaled)
        
        results[name] = {'MAE': m_mae, 'MAPE': m_mape, 'RMSE': m_rmse}
        predictions_dict[name] = pred_unscaled
        
        print(f"✅ {name} Completed. MAE: {m_mae:,.2f} MWh")

    # LEADERBOARD
    print("\n🏆 DEEP LEARNING LEADERBOARD (TEST SET 2018) 🏆")
    print(f"{'Model':<12} | {'MAE (MWh)':<12} | {'MAPE (%)':<10} | {'RMSE (MWh)':<12}")
    print("-" * 55)
    print(f"{'TSO (Goal)':<12} | {269.85:<12.2f} | {0.926:<10.3f} | {389.32:<12.2f}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['MAE'])
    for name, metrics in sorted_results:
        print(f"{name:<12} | {metrics['MAE']:<12.2f} | {metrics['MAPE']:<10.3f} | {metrics['RMSE']:<12.2f}")
        
    return predictions_dict, sorted_results