import os
import json
import gc
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.utils.timeseries_generation import concatenate
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.models import RNNModel, TCNModel, NBEATSModel, NHiTSModel, TFTModel
from darts.metrics import mae, mape, rmse
import warnings

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# LOSS LOGGER CALLBACK
# ---------------------------------------------------------------------------

class LossLogger(pl.Callback):
    """Records train_loss and val_loss at the end of every epoch."""
    def __init__(self):
        self.train_loss = []
        self.val_loss   = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_loss.append(float(loss))

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_loss.append(float(loss))


# ---------------------------------------------------------------------------
# STEP 1 — PREPARE DARTS DATA
# ---------------------------------------------------------------------------

def prepare_darts_from_split(X_train, y_train, X_test, y_test):
    """
    Converts the existing Pandas train/test split into scaled Darts TimeSeries
    objects, correctly separating past vs future covariates.

    Covariate classification rules
    ───────────────────────────────
    FUTURE (known at prediction time — calendar & astronomy):
      • anything with 'is_holiday' or 'next_day' in the name
      • cyclical encodings: 'sin' / 'cos'
      • 'day_type', 'hour', 'month', 'week_of_year'
      • 'sunrise_hour', 'sunset_hour', 'day_length_h', 'is_daylight'

    PAST (only known up to the current hour):
      • everything else — weather measurements, lags, rolling stats,
        generation lags, price lag, HDH/CDH
    """
    print("--- 1. PREPARING DARTS DATA FROM EXISTING TRAIN/TEST SPLIT ---")

    df_train = X_train.copy()
    df_train['target'] = y_train.values

    df_test = X_test.copy()
    df_test['target'] = y_test.values

    # Darts requires timezone-naive indices
    df_train.index = df_train.index.tz_localize(None)
    df_test.index  = df_test.index.tz_localize(None)

    # ── Covariate classification ──────────────────────────────────────────
    FUTURE_KEYWORDS = (
        'is_holiday', 'next_day', '_sin', '_cos',
        'day_type', 'sunrise_hour', 'sunset_hour', 'day_length_h', 'is_daylight',
    )
    # 'hour' and 'month' raw integers are deterministic calendar features too
    FUTURE_EXACT = {'hour', 'month', 'week_of_year'}

    def is_future(col):
        if col in FUTURE_EXACT:
            return True
        return any(kw in col for kw in FUTURE_KEYWORDS)

    feature_cols       = [c for c in X_train.columns]
    future_cov_cols    = [c for c in feature_cols if is_future(c)]
    past_cov_cols      = [c for c in feature_cols if not is_future(c)]

    print(f"  Future covariates : {len(future_cov_cols)} columns")
    print(f"  Past covariates   : {len(past_cov_cols)} columns")

    # ── Build Darts TimeSeries ────────────────────────────────────────────
    train_target  = TimeSeries.from_dataframe(df_train, value_cols='target', freq='h')
    test_target   = TimeSeries.from_dataframe(df_test,  value_cols='target', freq='h')

    train_past    = TimeSeries.from_dataframe(df_train, value_cols=past_cov_cols,   freq='h')
    test_past     = TimeSeries.from_dataframe(df_test,  value_cols=past_cov_cols,   freq='h')

    train_future  = TimeSeries.from_dataframe(df_train, value_cols=future_cov_cols, freq='h')
    test_future   = TimeSeries.from_dataframe(df_test,  value_cols=future_cov_cols, freq='h')

    # ── Scale (fit on train only) ─────────────────────────────────────────
    print("  Scaling (fit on train, transform test)...")

    scaler_target = Scaler()
    train_target_scaled = scaler_target.fit_transform(train_target)
    test_target_scaled  = scaler_target.transform(test_target)

    scaler_past = Scaler()
    train_past_scaled = scaler_past.fit_transform(train_past)
    test_past_scaled  = scaler_past.transform(test_past)

    scaler_future = Scaler()
    train_future_scaled = scaler_future.fit_transform(train_future)
    test_future_scaled  = scaler_future.transform(test_future)

    print("✅ Darts data preparation complete.\n")
    return {
        'train_target_scaled'  : train_target_scaled,
        'test_target_scaled'   : test_target_scaled,
        'train_past_scaled'    : train_past_scaled,
        'test_past_scaled'     : test_past_scaled,
        'train_future_scaled'  : train_future_scaled,
        'test_future_scaled'   : test_future_scaled,
        'test_target_unscaled' : test_target,
        'scaler_target'        : scaler_target,
    }


# ---------------------------------------------------------------------------
# STEP 2 — TRAIN & EVALUATE FLEET
# ---------------------------------------------------------------------------

def train_and_evaluate_deep_learning_fleet(
    darts_data,
    lookback_hours: int = 168,
    horizon: int = 24,
    max_epochs: int = 60,
    tso_metrics: dict = None,
):
    """
    Trains LSTM, GRU, TCN, N-HiTS, N-BEATS, TFT and evaluates each against
    the 2018 test set using historical_forecasts (true day-ahead simulation).

    N-BEATS and N-HiTS: Darts only supports FUTURE covariates for these models
    (not past). They are trained without covariates here and rely purely on the
    target series — which is the intended use case per the original papers.
    TFT: supports both past and future covariates.
    RNN / TCN: support past OR future covariates respectively.

    Parameters
    ----------
    tso_metrics : dict, optional
        Dict with keys 'MAE', 'MAPE', 'RMSE' from evaluate_tso_baseline().
        Used to print a live comparison in the leaderboard.
    """
    print("=" * 55)
    print("🚀 LAUNCHING DEEP LEARNING FLEET")
    print("=" * 55)

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    fleet_configs = {}

    # ── Helper: trainer kwargs ────────────────────────────────────────────
    def get_trainer_kwargs(patience=5):
        logger = LossLogger()
        kwargs = {
            "callbacks": [
                EarlyStopping(
                    monitor="val_loss", patience=patience,
                    min_delta=1e-4, mode='min'
                ),
                logger,
            ],
            "enable_checkpointing": True,
            "gradient_clip_val": 1.0,
        }
        return kwargs, logger

    # ── Unpack data ───────────────────────────────────────────────────────
    train_tgt    = darts_data['train_target_scaled']
    test_tgt     = darts_data['test_target_scaled']
    train_past   = darts_data['train_past_scaled']
    test_past    = darts_data['test_past_scaled']
    train_future = darts_data['train_future_scaled']
    test_future  = darts_data['test_future_scaled']
    actuals      = darts_data['test_target_unscaled']
    scaler       = darts_data['scaler_target']

    # Full series for historical_forecasts (train + test window)
    full_past   = concatenate([train_past,   test_past],   axis=0)
    full_future = concatenate([train_future, test_future], axis=0)
    full_series = concatenate([train_tgt,    test_tgt],    axis=0)

    # ── Build trainer kwargs & loggers ────────────────────────────────────
    loggers = {}
    trainer_kwargs = {}
    for name, pat in [('LSTM',5),('GRU',5),('TCN',5),('N-BEATS',5),('N-HiTS',5),('TFT',7)]:
        trainer_kwargs[name], loggers[name] = get_trainer_kwargs(patience=pat)

    # ── Model definitions ─────────────────────────────────────────────────
    #
    # covariate_type rules:
    #   'future_only' → RNNModel (LSTM/GRU)
    #   'past_only'   → TCNModel
    #   'none'        → N-BEATS, N-HiTS  (Darts does NOT support past covariates
    #                                     for these; future covariates are optional
    #                                     but omitted here for cleaner baselines)
    #   'both'        → TFTModel
    #
    common = dict(n_epochs=max_epochs, batch_size=32, random_state=42)

    models_dict = {
        'LSTM': {
            'model': RNNModel(
                model='LSTM',
                input_chunk_length=lookback_hours,
                training_length=lookback_hours + horizon,
                hidden_dim=64, n_rnn_layers=2, dropout=0.1,
                pl_trainer_kwargs=trainer_kwargs['LSTM'],
                **common,
            ),
            'covariate_type': 'future_only',
        },
        'GRU': {
            'model': RNNModel(
                model='GRU',
                input_chunk_length=lookback_hours,
                training_length=lookback_hours + horizon,
                hidden_dim=64, n_rnn_layers=2, dropout=0.1,
                pl_trainer_kwargs=trainer_kwargs['GRU'],
                **common,
            ),
            'covariate_type': 'future_only',
        },
        'TCN': {
            'model': TCNModel(
                input_chunk_length=lookback_hours,
                output_chunk_length=horizon,
                num_filters=32, kernel_size=3, num_layers=4, dropout=0.1,
                pl_trainer_kwargs=trainer_kwargs['TCN'],
                **common,
            ),
            'covariate_type': 'past_only',
        },
        'N-BEATS': {
            # Pure univariate — no covariates (as per original paper design)
            'model': NBEATSModel(
                input_chunk_length=lookback_hours,
                output_chunk_length=horizon,
                num_stacks=2, num_blocks=2, num_layers=2, layer_widths=128,
                pl_trainer_kwargs=trainer_kwargs['N-BEATS'],
                **common,
            ),
            'covariate_type': 'none',
        },
        'N-HiTS': {
            # Pure univariate — no covariates (as per original paper design)
            'model': NHiTSModel(
                input_chunk_length=lookback_hours,
                output_chunk_length=horizon,
                num_stacks=2, num_blocks=2, num_layers=2, layer_widths=128,
                pl_trainer_kwargs=trainer_kwargs['N-HiTS'],
                **common,
            ),
            'covariate_type': 'none',
        },
        'TFT': {
            'model': TFTModel(
                input_chunk_length=lookback_hours,
                output_chunk_length=horizon,
                hidden_size=64, lstm_layers=2, num_attention_heads=4,
                dropout=0.1, add_relative_index=True,
                pl_trainer_kwargs=trainer_kwargs['TFT'],
                **common,
            ),
            'covariate_type': 'both',
        },
    }

    # ── Training loop ─────────────────────────────────────────────────────
    results = {}
    predictions_dict = {}

    for name, config in models_dict.items():
        print(f"\n{'─'*50}")
        print(f"🧠 TRAINING {name}")
        print(f"{'─'*50}")

        model    = config['model']
        cov_type = config['covariate_type']

        # Validation split (last 15% of training data)
        VAL_SPLIT = 0.85
        tr_tgt,  vl_tgt  = train_tgt.split_before(VAL_SPLIT)
        tr_past, vl_past = train_past.split_before(VAL_SPLIT)
        tr_fut,  vl_fut  = train_future.split_before(VAL_SPLIT)

        # FIT
        if cov_type == 'future_only':
            model.fit(series=tr_tgt, val_series=vl_tgt,
                      future_covariates=tr_fut, val_future_covariates=vl_fut,
                      verbose=True)
        elif cov_type == 'past_only':
            model.fit(series=tr_tgt, val_series=vl_tgt,
                      past_covariates=tr_past, val_past_covariates=vl_past,
                      verbose=True)
        elif cov_type == 'none':
            model.fit(series=tr_tgt, val_series=vl_tgt, verbose=True)
        else:  # 'both' — TFT
            model.fit(series=tr_tgt, val_series=vl_tgt,
                      past_covariates=tr_past, val_past_covariates=vl_past,
                      future_covariates=tr_fut, val_future_covariates=vl_fut,
                      verbose=True)

        # SAVE
        epochs_run = model.epochs_trained
        model_path = os.path.join(save_dir, f"{name}_model.pt")
        model.save(model_path)
        fleet_configs[name] = {
            "lookback_hours": lookback_hours,
            "horizon": horizon,
            "covariate_type": cov_type,
            "model_path": model_path,
            "epochs_trained": epochs_run,
        }
        print(f"💾 Saved to {model_path} ({epochs_run} epochs)")

        # HISTORICAL FORECASTS (true day-ahead simulation)
        print(f"Running historical_forecasts for {name}...")

        hf_kwargs = {}
        if cov_type == 'future_only':
            hf_kwargs['future_covariates'] = full_future
        elif cov_type == 'past_only':
            hf_kwargs['past_covariates'] = full_past
        elif cov_type == 'both':
            hf_kwargs['past_covariates']   = full_past
            hf_kwargs['future_covariates'] = full_future
        # 'none' → no kwargs

        pred_scaled_list = model.historical_forecasts(
            series=full_series,
            start=test_tgt.start_time(),
            forecast_horizon=horizon,
            stride=horizon,
            last_points_only=False,
            retrain=False,
            verbose=False,
            **hf_kwargs,
        )

        # Stitch chunks — use Darts concatenate (handles index gaps safely)
        if isinstance(pred_scaled_list, list):
            pred_scaled = concatenate(pred_scaled_list, axis=0)
        else:
            pred_scaled = pred_scaled_list

        # Inverse transform
        pred_unscaled = scaler.inverse_transform(pred_scaled)

        # Align (length may differ by a few points at boundaries)
        aligned_actual = actuals.slice_intersect(pred_unscaled)
        aligned_pred   = pred_unscaled.slice_intersect(actuals)

        m_mae  = mae(aligned_actual,  aligned_pred)
        m_mape = mape(aligned_actual, aligned_pred)
        m_rmse = rmse(aligned_actual, aligned_pred)

        results[name]          = {'MAE': m_mae, 'MAPE': m_mape, 'RMSE': m_rmse}
        predictions_dict[name] = pred_unscaled

        print(f"✅ {name} | MAE: {m_mae:,.2f} MWh | MAPE: {m_mape:.3f}% | RMSE: {m_rmse:,.2f} MWh")

        # Save loss curves
        ll = loggers[name]
        n  = len(ll.train_loss)
        loss_df = pd.DataFrame({
            'epoch':      range(1, n + 1),
            'train_loss': ll.train_loss,
            'val_loss':   (ll.val_loss[:n] + [float('nan')] * max(0, n - len(ll.val_loss))),
        })
        loss_path = os.path.join(save_dir, f"{name}_loss_curves.csv")
        loss_df.to_csv(loss_path, index=False)
        print(f"📈 Loss curves → {loss_path}")

        # Free GPU memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Leaderboard ───────────────────────────────────────────────────────
    sorted_results = sorted(results.items(), key=lambda x: x[1]['MAE'])

    print("\n" + "=" * 60)
    print("🏆 DEEP LEARNING LEADERBOARD — TEST SET 2018")
    print("=" * 60)
    print(f"{'Model':<12} | {'MAE (MWh)':>10} | {'MAPE (%)':>9} | {'RMSE (MWh)':>11}")
    print("-" * 60)

    if tso_metrics:
        print(f"{'TSO (goal)':<12} | {tso_metrics['MAE']:>10.2f} | "
              f"{tso_metrics['MAPE']:>9.3f} | {tso_metrics['RMSE']:>11.2f}  ← baseline")
        print("-" * 60)

    for name, m in sorted_results:
        beat = ""
        if tso_metrics and m['MAE'] < tso_metrics['MAE']:
            beat = "  ✅ beats TSO"
        print(f"{name:<12} | {m['MAE']:>10.2f} | {m['MAPE']:>9.3f} | {m['RMSE']:>11.2f}{beat}")

    # Save configs
    config_path = os.path.join(save_dir, "fleet_configs.json")
    with open(config_path, "w") as f:
        json.dump(fleet_configs, f, indent=4)
    print(f"\n📁 Configs saved → {config_path}")

    return predictions_dict, sorted_results


# ---------------------------------------------------------------------------
# LOSS CURVE VISUALISATION (call from notebook after training)
# ---------------------------------------------------------------------------

def plot_saved_loss_curves(save_dir="saved_models"):
    csv_files = sorted(f for f in os.listdir(save_dir) if f.endswith('_loss_curves.csv'))
    if not csv_files:
        print(f"No loss curve CSVs found in {save_dir}")
        return

    ncols = 3
    nrows = (len(csv_files) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, fname in enumerate(csv_files):
        ax = axes[i]
        model_name = fname.replace("_loss_curves.csv", "")
        df = pd.read_csv(os.path.join(save_dir, fname))

        train_d = df[['epoch', 'train_loss']].dropna()
        val_d   = df[['epoch', 'val_loss']].dropna()

        ax.plot(train_d['epoch'], train_d['train_loss'],
                label='Train', color='steelblue', linewidth=2)
        ax.plot(val_d['epoch'], val_d['val_loss'],
                label='Val', color='tomato', linewidth=2, linestyle='--')

        if not val_d.empty:
            best_ep = val_d.loc[val_d['val_loss'].idxmin(), 'epoch']
            ax.axvline(best_ep, color='grey', linestyle=':', linewidth=1.5,
                       label=f'Best ({int(best_ep)})')

        ax.set_title(model_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Training vs Validation Loss — Deep Learning Fleet',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()