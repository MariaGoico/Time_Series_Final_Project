import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

def tune_xgboost_with_cv(X_train, y_train):
    """
    Performs Time Series Cross-Validation to find the best hyperparameters 
    for the XGBoost model without leaking future data.
    """
    print("--- 4. HYPERPARAMETER TUNING WITH TIME SERIES CV ---")
    
    # 1. Define the correct Cross-Validation strategy for Time Series
    # n_splits=3 means we will have 3 validation periods progressively
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 2. Define the base model
    model_xgb = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # 3. Define the hyperparameter grid to search
    # (Kept relatively small to run in a reasonable time)
    param_grid = {
        'n_estimators': [300, 500, 800],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    # 4. We want to minimize MAE, so we create a custom scorer
    # greater_is_better=False because it's an error metric (lower is better)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    # 5. Set up the Randomized Search
    # We use RandomizedSearchCV instead of GridSearchCV to save time.
    # n_iter=10 means it will test 10 random combinations from the grid.
    random_search = RandomizedSearchCV(
        estimator=model_xgb,
        param_distributions=param_grid,
        n_iter=10, 
        scoring=mae_scorer,
        cv=tscv,               # <--- CRUCIAL: Passing our Time Series Split here!
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    print("Starting cross-validation search (this might take a few minutes)...")
    random_search.fit(X_train, y_train)
    
    print("\n✅ Cross-Validation Complete!")
    print(f"Best Hyperparameters Found:")
    for param, value in random_search.best_params_.items():
        print(f"   - {param}: {value}")
        
    # The score is negative MAE, so we multiply by -1 to print the actual MAE
    best_cv_mae = -random_search.best_score_
    print(f"\nExpected Validation MAE (based on CV): {best_cv_mae:,.2f} MWh")
    
    # Return the model trained on the full X_train with the best parameters
    return random_search.best_estimator_


def train_and_evaluate_lgbm_quantiles(X_train, y_train, X_test, y_test):
    """
    Trains three LightGBM models using Quantile Regression to generate 
    a median prediction (50th percentile) and an 80% prediction interval 
    (10th and 90th percentiles).
    """
    print("--- 4. TRAINING LIGHTGBM WITH QUANTILE REGRESSION ---")
    
    # Common hyperparameters for LightGBM
    lgbm_params = {
        'n_estimators': 800,
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # 1. Train Median Model (50th Percentile) - This is our main forecast
    print("Training Median Model (Alpha = 0.50)...")
    model_median = lgb.LGBMRegressor(objective='quantile', alpha=0.5, **lgbm_params)
    model_median.fit(X_train, y_train)
    pred_median = model_median.predict(X_test)
    
    # 2. Train Lower Bound Model (10th Percentile)
    print("Training Lower Bound Model (Alpha = 0.10)...")
    model_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.1, **lgbm_params)
    model_lower.fit(X_train, y_train)
    pred_lower = model_lower.predict(X_test)
    
    # 3. Train Upper Bound Model (90th Percentile)
    print("Training Upper Bound Model (Alpha = 0.90)...")
    model_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.9, **lgbm_params)
    model_upper.fit(X_train, y_train)
    pred_upper = model_upper.predict(X_test)
    
    # Format predictions into a DataFrame for easier handling
    df_preds = pd.DataFrame({
        'Actual': y_test,
        'Pred_Median': pred_median,
        'Pred_Lower': pred_lower,
        'Pred_Upper': pred_upper
    }, index=y_test.index)
    
    # Calculate deterministic metrics for the main forecast (Median)
    mae_lgbm = mean_absolute_error(y_test, pred_median)
    mape_lgbm = mean_absolute_percentage_error(y_test, pred_median) * 100
    rmse_lgbm = np.sqrt(mean_squared_error(y_test, pred_median))
    
    print("\n📊 LIGHTGBM MEDIAN FORECAST RESULTS (2018):")
    print(f"   MAE:  {mae_lgbm:,.2f} MWh")
    print(f"   MAPE: {mape_lgbm:.3f} %")
    print(f"   RMSE: {rmse_lgbm:,.2f} MWh")
    
    return df_preds, model_median

def evaluate_prediction_intervals(df_preds):
    """
    Calculates advanced metrics to measure the precision and reliability 
    of the prediction intervals (PICP and MPIW).
    """
    print("\n--- 5. EVALUATING PREDICTION INTERVALS ---")
    
    y_true = df_preds['Actual']
    lower = df_preds['Pred_Lower']
    upper = df_preds['Pred_Upper']
    
    # PICP (Prediction Interval Coverage Probability)
    # Measures the percentage of true values that fall within the bounds
    is_covered = (y_true >= lower) & (y_true <= upper)
    picp = is_covered.mean() * 100
    
    # MPIW (Mean Prediction Interval Width)
    # Measures how wide (uncertain) the intervals are on average
    mpiw = (upper - lower).mean()
    
    print(f"🎯 INTERVAL METRICS (Target Coverage: ~80%):")
    print(f"   PICP (Coverage): {picp:.2f}% of actual values fell inside the interval.")
    print(f"   MPIW (Width):    {mpiw:,.2f} MWh average width of the uncertainty band.")
    
    return picp, mpiw

def plot_lightgbm_intervals(df_preds, window_hours=168, start_idx=0):
    """
    Visualizes the actual demand, the median forecast, and the prediction 
    intervals as a shaded region.
    """
    plot_data = df_preds.iloc[start_idx:start_idx+window_hours]
    
    plt.figure(figsize=(16, 6))
    
    # Plot Actual and Median Forecast
    plt.plot(plot_data.index, plot_data['Actual'], label='Actual Demand', color='black', linewidth=1.5)
    plt.plot(plot_data.index, plot_data['Pred_Median'], label='LGBM Median Forecast', color='blue', linestyle='--')
    
    # Fill the area between Lower and Upper bounds
    plt.fill_between(plot_data.index, 
                     plot_data['Pred_Lower'], 
                     plot_data['Pred_Upper'], 
                     color='dodgerblue', alpha=0.25, 
                     label='80% Prediction Interval (10th - 90th percentile)')
    
    plt.title(f'LightGBM Probabilistic Forecast vs Actual Demand ({window_hours} hours)')
    plt.ylabel('Total Load (MWh)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_alternative_models(X_train, y_train, X_test, y_test):
    """
    Trains and compares three different model architectures:
    CatBoost, a Neural Network (MLP), and Random Forest.
    """
    print("--- 4. TRAINING ALTERNATIVE MODELS ---")
    
    # 1. SCALING THE DATA
    # Neural Networks strictly require scaled data to converge.
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ==========================================
    # MODEL A: CATBOOST REGRESSOR
    # ==========================================
    print("\nTraining Model A: CatBoost...")
    model_cat = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        verbose=0, # Set to 0 to avoid massive console output
        random_seed=42
    )
    model_cat.fit(X_train, y_train) # CatBoost doesn't strictly need scaled data
    pred_cat = model_cat.predict(X_test)
    
    mae_cat = mean_absolute_error(y_test, pred_cat)
    mape_cat = mean_absolute_percentage_error(y_test, pred_cat) * 100
    rmse_cat = np.sqrt(mean_squared_error(y_test, pred_cat))

    # ==========================================
    # MODEL B: NEURAL NETWORK (MLP)
    # ==========================================
    print("Training Model B: Neural Network (MLPRegressor)...")
    # Architecture: 2 hidden layers with 128 and 64 neurons
    model_mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.01, # L2 penalty (regularization)
        max_iter=200,
        early_stopping=True,
        random_state=42
    )
    model_mlp.fit(X_train_scaled, y_train) # MLP MUST use scaled data
    pred_mlp = model_mlp.predict(X_test_scaled)
    
    mae_mlp = mean_absolute_error(y_test, pred_mlp)
    mape_mlp = mean_absolute_percentage_error(y_test, pred_mlp) * 100
    rmse_mlp = np.sqrt(mean_squared_error(y_test, pred_mlp))

    # ==========================================
    # MODEL C: RANDOM FOREST
    # ==========================================
    print("Training Model C: Random Forest...")
    # Using slightly fewer trees and depth to keep training time reasonable
    model_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42
    )
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)
    
    mae_rf = mean_absolute_error(y_test, pred_rf)
    mape_rf = mean_absolute_percentage_error(y_test, pred_rf) * 100
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))

    # ==========================================
    # RESULTS COMPARISON
    # ==========================================
    print("\n📊 ALTERNATIVE MODELS COMPARISON (TEST SET 2018):")
    print(f"{'Metric':<10} | {'TSO (Goal)':<12} | {'CatBoost':<12} | {'Neural Net':<12} | {'RandomForest':<12}")
    print("-" * 75)
    print(f"{'MAE (MWh)':<10} | {269.85:<12.2f} | {mae_cat:<12.2f} | {mae_mlp:<12.2f} | {mae_rf:<12.2f}")
    print(f"{'MAPE (%)':<10} | {0.926:<12.3f} | {mape_cat:<12.3f} | {mape_mlp:<12.3f} | {mape_rf:<12.3f}")
    print(f"{'RMSE (MWh)':<10} | {389.32:<12.2f} | {rmse_cat:<12.2f} | {rmse_mlp:<12.2f} | {rmse_rf:<12.2f}")
    
    return pred_cat, pred_mlp, pred_rf


def train_and_evaluate_prophet(X_train, y_train, X_test, y_test):
    """
    Trains a Meta Prophet model with external regressors (weather, lags).
    Prophet natively provides prediction intervals (uncertainty bounds).
    """
    print("--- 5. TRAINING META PROPHET (ADDITIVE MODEL) ---")
    
    # 1. PREPARE THE DATA FOR PROPHET
    # Prophet requires the datetime column to be named 'ds' and the target 'y'.
    # It also strictly hates timezone-aware datetimes, so we must remove the UTC timezone.
    
    print("Formatting data for Prophet (ds, y)...")
    train_df = X_train.copy()
    train_df['ds'] = train_df.index.tz_localize(None)
    train_df['y'] = y_train.values
    
    test_df = X_test.copy()
    test_df['ds'] = test_df.index.tz_localize(None)
    test_df['y'] = y_test.values
    
    # 2. INITIALIZE PROPHET
    # We set interval_width=0.80 to get the 10th and 90th percentiles (80% confidence interval)
    print("Initializing model and adding external regressors...")
    model_prophet = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        interval_width=0.80, # 80% Prediction Interval
    )
    
    # 3. ADD EXTERNAL COVARIATES (Regressors)
    # We must explicitly tell Prophet to use our weather, lags, and holiday features
    # (Excluding 'ds' and 'y' which are base columns)
    feature_columns = [col for col in train_df.columns if col not in ['ds', 'y']]
    for col in feature_columns:
        model_prophet.add_regressor(col)
        
    # 4. FIT THE MODEL
    print("Fitting Prophet model (this may take a minute or two on hourly data)...")
    model_prophet.fit(train_df)
    
    # 5. PREDICT ON TEST SET
    print("Making predictions on the 2018 Test Set...")
    # Prophet prediction outputs a dataframe with yhat, yhat_lower, yhat_upper, etc.
    forecast = model_prophet.predict(test_df)
    
    # 6. CALCULATE METRICS
    pred_yhat = forecast['yhat'].values
    mae = mean_absolute_error(y_test, pred_yhat)
    mape = mean_absolute_percentage_error(y_test, pred_yhat) * 100
    rmse = np.sqrt(mean_squared_error(y_test, pred_yhat))
    
    # Calculate Prediction Interval Coverage
    y_true = test_df['y'].values
    lower = forecast['yhat_lower'].values
    upper = forecast['yhat_upper'].values
    picp = ((y_true >= lower) & (y_true <= upper)).mean() * 100
    
    print("\n📊 PROPHET MODEL RESULTS (TEST SET 2018):")
    print("-" * 45)
    print(f"   MAE:  {mae:,.2f} MWh")
    print(f"   MAPE: {mape:.3f} %")
    print(f"   RMSE: {rmse:,.2f} MWh")
    print(f"   PICP: {picp:.2f}% (Target: 80% coverage)")
    
    return model_prophet, forecast