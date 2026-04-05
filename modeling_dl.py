import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
def train_and_evaluate_deep_learning_fleet(darts_data, lookback_hours=168, horizon=24):
    print("="*50)
    print("🚀 LAUNCHING DEEP LEARNING FLEET (6 MODELS)")
    print("="*50)

    # Create a directory to store models and configs
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Dictionary to keep track of configurations to save later
    fleet_configs = {}

    # Define the Early Stopping rule (wait 3 epochs for improvement before quitting)
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=3,
        min_delta=0.001,
        mode='min',
    )
    
    trainer_kwargs = {"callbacks": [early_stopper]}
    max_epochs = 60 # Set a high ceiling, the early stopper will cut it off naturally!
    
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
                              n_epochs=max_epochs, pl_trainer_kwargs=trainer_kwargs, random_state=42),
            'covariate_type': 'future_only'  # RNNModel does NOT accept past_covariates
        },
        'GRU': {
            'model': RNNModel(model='GRU', input_chunk_length=lookback_hours, training_length=lookback_hours+horizon, 
                              n_epochs=max_epochs, pl_trainer_kwargs=trainer_kwargs, random_state=42),
            'covariate_type': 'future_only'  # RNNModel does NOT accept past_covariates
        },
        'TCN': {
            'model': TCNModel(input_chunk_length=lookback_hours, output_chunk_length=horizon, 
                              n_epochs=max_epochs, pl_trainer_kwargs=trainer_kwargs, random_state=42),
            'covariate_type': 'past_only'
        },
        'N-BEATS': {
            'model': NBEATSModel(input_chunk_length=lookback_hours, output_chunk_length=horizon, 
                                 n_epochs=max_epochs, pl_trainer_kwargs=trainer_kwargs, random_state=42),
            'covariate_type': 'past_only'
        },
        'N-HiTS': {
            'model': NHiTSModel(input_chunk_length=lookback_hours, output_chunk_length=horizon, 
                                n_epochs=max_epochs, pl_trainer_kwargs=trainer_kwargs, random_state=42),
            'covariate_type': 'past_only'
        },
        'TFT': {
            'model': TFTModel(input_chunk_length=lookback_hours, output_chunk_length=horizon, 
                              add_relative_index=True, n_epochs=max_epochs, pl_trainer_kwargs=trainer_kwargs, random_state=42),
            'covariate_type': 'both'  # TFT supports both past and future covariates
        }
    }
    
    results = {}
    predictions_dict = {}
    actuals = darts_data['test_target_unscaled']
    scaler = darts_data['scaler_target']
    
    for name, config in models_dict.items():
        print(f"\n--- 🧠 TRAINING {name} ---")
        model = config['model']
        cov_type = config['covariate_type']

        # Create a validation split (e.g., last 15% of training data) for the early stopper to monitor
        train_split, val_split = train_tgt.split_before(0.85)

        # FIT — pass the val_series so Early Stopping has something to monitor
        if cov_type == 'future_only':
            model.fit(series=train_split, val_series=val_split, 
                      future_covariates=train_future, val_future_covariates=train_future, verbose=False)
        elif cov_type == 'past_only':
            model.fit(series=train_split, val_series=val_split, 
                      past_covariates=train_past, val_past_covariates=train_past, verbose=False)
        else:  # 'both' → TFT
            model.fit(series=train_split, val_series=val_split, 
                      past_covariates=train_past, val_past_covariates=train_past,
                      future_covariates=train_future, val_future_covariates=train_future, verbose=False)

        # --- SAVE THE MODEL ---
        model_path = os.path.join(save_dir, f"{name}_model.pt")
        model.save(model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # --- STORE CONFIGURATION ---
        fleet_configs[name] = {
            "lookback_hours": lookback_hours,
            "horizon": horizon,
            "covariate_type": cov_type,
            "model_path": model_path
            # You can add more params here like learning_rate if you define them!
        }
           
        print(f"Predicting 2018 with {name} (Historical Forecasts simulating Day-Ahead)...")
        
        # PREDICT
        if cov_type == 'future_only':
            kwargs = {'future_covariates': full_future}
        elif cov_type == 'past_only':
            kwargs = {'past_covariates': full_past}
        else:  # 'both' → TFT
            kwargs = {'past_covariates': full_past, 'future_covariates': full_future}
            
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
        
        # 1. Use slice_intersect for BOTH to ensure identical lengths
        aligned_actuals = actuals.slice_intersect(pred_unscaled)
        aligned_preds   = pred_unscaled.slice_intersect(actuals)
        
        # 2. Use the aligned versions for ALL metric calculations
        m_mae = mae(aligned_actuals, aligned_preds)
        m_mape = mape(aligned_actuals, aligned_preds)
        m_rmse = rmse(aligned_actuals, aligned_preds)
        
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
        
    config_path = os.path.join(save_dir, "fleet_configs.json")
    with open(config_path, "w") as f:
        json.dump(fleet_configs, f, indent=4)
    print(f"\n All configurations saved to: {config_path}")

    return predictions_dict, sorted_results