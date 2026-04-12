import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer, mean_absolute_error,
    mean_squared_error, mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# XGBoost with Time-Series CV
def tune_xgboost_with_cv(X_train, y_train, n_iter=25):
    print("--- XGBoost: TIME SERIES CV HYPERPARAMETER SEARCH ---")

    tscv = TimeSeriesSplit(n_splits=5, gap=24)

    param_grid = {
        'n_estimators':     [800, 1200, 1500],
        'learning_rate':    [0.01, 0.03, 0.05],
        'max_depth':        [6, 8, 10, 12],
        'subsample':        [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=mae_scorer,
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    print(f"Running {n_iter} random combinations × 5 folds …")
    search.fit(X_train, y_train)

    print("\nCross-validation complete.")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"CV MAE (best): {-search.best_score_:,.2f} MWh")

    return search.best_estimator_

def evaluate_xgboost(best_xgb, X_test, y_test, tso_metrics=None):
    y_pred = best_xgb.predict(X_test)
    mae_  = mean_absolute_error(y_test, y_pred)
    mape_ = mean_absolute_percentage_error(y_test, y_pred) * 100
    rmse_ = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nXGBoost — TEST SET 2018")
    _print_metrics(mae_, mape_, rmse_, tso_metrics)

    return pd.Series(y_pred, index=y_test.index, name='XGBoost'), \
           {'MAE': mae_, 'MAPE': mape_, 'RMSE': rmse_}

def plot_xgb_feature_importance(model, X_train, top_n=20):
    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    imp = imp.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, top_n * 0.38))
    imp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'XGBoost — Top {top_n} Feature Importances', fontweight='bold')
    ax.set_xlabel('Importance score')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# LightGBM
def train_and_evaluate_lgbm_quantiles(
    X_train, y_train, X_test, y_test,
    tso_metrics=None,
    lower_q=0.10, upper_q=0.90,
):

    print("\n--- LightGBM: QUANTILE REGRESSION ---")
    lgbm_params = dict(
        n_estimators=1000, learning_rate=0.03, max_depth=10,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )

    print(f"Training median model (α=0.50)…")
    m_med = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **lgbm_params)
    m_med.fit(X_train, y_train)
    pred_med = m_med.predict(X_test)

    print(f"Training lower bound  (α={lower_q:.2f})…")
    m_low = lgb.LGBMRegressor(objective='quantile', alpha=lower_q, **lgbm_params)
    m_low.fit(X_train, y_train)
    pred_low = m_low.predict(X_test)

    print(f"Training upper bound  (α={upper_q:.2f})…")
    m_up = lgb.LGBMRegressor(objective='quantile', alpha=upper_q, **lgbm_params)
    m_up.fit(X_train, y_train)
    pred_up = m_up.predict(X_test)

    df_preds = pd.DataFrame({
        'Actual':      y_test.values,
        'Pred_Median': pred_med,
        'Pred_Lower':  pred_low,
        'Pred_Upper':  pred_up,
    }, index=y_test.index)

    mae_  = mean_absolute_error(y_test, pred_med)
    mape_ = mean_absolute_percentage_error(y_test, pred_med) * 100
    rmse_ = np.sqrt(mean_squared_error(y_test, pred_med))

    print("\n📊 LightGBM Median — TEST SET 2018")
    _print_metrics(mae_, mape_, rmse_, tso_metrics)

    metrics = {'MAE': mae_, 'MAPE': mape_, 'RMSE': rmse_}
    return df_preds, m_med, metrics

def evaluate_prediction_intervals(df_preds, target_coverage=0.80):
    print("\n--- INTERVAL METRICS ---")
    y   = df_preds['Actual']
    lo  = df_preds['Pred_Lower']
    hi  = df_preds['Pred_Upper']
    α   = 1.0 - target_coverage

    covered = (y >= lo) & (y <= hi)
    picp = covered.mean() * 100
    mpiw = (hi - lo).mean()

    width     = hi - lo
    penalty_l = (2 / α) * (lo - y).clip(lower=0)
    penalty_u = (2 / α) * (y - hi).clip(lower=0)
    winkler   = (width + penalty_l + penalty_u).mean()

    print(f"  Target coverage : {target_coverage*100:.0f}%")
    print(f"  PICP            : {picp:.2f}%  {'✅' if picp >= target_coverage*100 else '⚠️  below target'}")
    print(f"  MPIW            : {mpiw:,.2f} MWh")
    print(f"  Winkler score   : {winkler:,.2f} MWh  (lower = better)")

    return {'PICP': picp, 'MPIW': mpiw, 'Winkler': winkler}

def plot_lightgbm_intervals(df_preds, window_hours=168, start_idx=0):
    plot_data = df_preds.iloc[start_idx: start_idx + window_hours]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(plot_data.index, plot_data['Actual'],
            label='Actual Demand', color='black', linewidth=1.5)
    ax.plot(plot_data.index, plot_data['Pred_Median'],
            label='LightGBM Median', color='royalblue', linestyle='--')
    ax.fill_between(
        plot_data.index, plot_data['Pred_Lower'], plot_data['Pred_Upper'],
        color='dodgerblue', alpha=0.25,
        label='80% Prediction Interval (P10–P90)',
    )
    ax.set_title(f'LightGBM Probabilistic Forecast vs Actual ({window_hours}h window)',
                 fontweight='bold')
    ax.set_ylabel('Total Load (MWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_lgbm_feature_importance(model, X_train, top_n=20):
    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    imp = imp.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, top_n * 0.38))
    imp.plot(kind='barh', ax=ax, color='darkorange')
    ax.set_title(f'LightGBM — Top {top_n} Feature Importances', fontweight='bold')
    ax.set_xlabel('Importance score')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Alternative models: CatBoost / MLP / Random Forest
def train_alternative_models(X_train, y_train, X_test, y_test, tso_metrics=None):
    print("\n--- ALTERNATIVE MODELS ---")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    results = {}

    # ── CatBoost ─────────────────────────────────────────────────────────
    print("Training CatBoost…")
    m_cat = CatBoostRegressor(
        iterations=1500, learning_rate=0.03, depth=8,
        loss_function='RMSE', verbose=0, random_seed=42,
    )
    m_cat.fit(X_train, y_train)
    p_cat = m_cat.predict(X_test)
    results['CatBoost'] = _metrics(y_test, p_cat)

    # ── MLP ───────────────────────────────────────────────────────────────
    print("Training MLP (Neural Network)…")
    m_mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
        alpha=0.01, max_iter=200, early_stopping=True, random_state=42,
    )
    m_mlp.fit(X_tr_s, y_train)
    p_mlp = m_mlp.predict(X_te_s)
    results['MLP'] = _metrics(y_test, p_mlp)

    # ── Random Forest ─────────────────────────────────────────────────────
    print("Training Random Forest…")
    m_rf = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=10,
        n_jobs=-1, random_state=42,
    )
    m_rf.fit(X_train, y_train)
    p_rf = m_rf.predict(X_test)
    results['RandomForest'] = _metrics(y_test, p_rf)

    # ── Comparison table ──────────────────────────────────────────────────
    print("\nALTERNATIVE MODELS — TEST SET 2018")
    header = f"{'Model':<14} | {'MAE':>10} | {'MAPE (%)':>9} | {'RMSE':>10}"
    print(header)
    print("-" * len(header))
    if tso_metrics:
        print(f"{'TSO (goal)':<14} | {tso_metrics['MAE']:>10.2f} | "
              f"{tso_metrics['MAPE']:>9.3f} | {tso_metrics['RMSE']:>10.2f}  ← baseline")
        print("-" * len(header))
    for name, m in results.items():
        beat = "  ✅" if tso_metrics and m['MAE'] < tso_metrics['MAE'] else ""
        print(f"{name:<14} | {m['MAE']:>10.2f} | {m['MAPE']:>9.3f} | {m['RMSE']:>10.2f}{beat}")

    preds = {
        'CatBoost':    pd.Series(p_cat, index=y_test.index),
        'MLP':         pd.Series(p_mlp, index=y_test.index),
        'RandomForest':pd.Series(p_rf,  index=y_test.index),
    }
    return preds, results

# Internal helpers
def _metrics(y_true, y_pred):
    return {
        'MAE':  mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
    }

def _print_metrics(mae_, mape_, rmse_, tso_metrics=None):
    print(f"  MAE  : {mae_:,.2f} MWh", end="")
    if tso_metrics:
        delta = mae_ - tso_metrics['MAE']
        sign  = "▼" if delta < 0 else "▲"
        print(f"  ({sign}{abs(delta):,.2f} vs TSO)", end="")
    print()
    print(f"  MAPE : {mape_:.3f} %")
    print(f"  RMSE : {rmse_:,.2f} MWh")