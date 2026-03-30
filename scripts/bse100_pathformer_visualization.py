"""
BSE100 Stock Data Visualization using PathFormer model for predictions.
Downloads real BSE100 stock data and uses trained PathFormer for forecasting.

Run:
  python scripts/bse100_pathformer_visualization.py --csv_path "./stock-data/bse100 - Sheet1.csv" --output ./bse100_pathformer_results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import copy
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from visualize_results import (
    plot_prediction_comparison,
    plot_multivariate_predictions,
    plot_error_distribution,
)
from exp.exp_main import Exp_Main


COL_COMPANY = 'Company Name'
COL_TICKER = 'NSE Ticker'
COL_DATA_POINTS = 'Data Points'


def read_bse100_csv(csv_path: str) -> pd.DataFrame:
    """Read and clean BSE100 universe from CSV."""
    df = pd.read_csv(csv_path)

    required_cols = [COL_COMPANY, COL_TICKER]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    cleaned = df.copy()
    cleaned[COL_COMPANY] = cleaned[COL_COMPANY].astype(str).str.strip()
    cleaned[COL_TICKER] = cleaned[COL_TICKER].fillna('').astype(str).str.strip().str.upper()

    before = len(cleaned)

    # Keep rows with valid ticker format
    cleaned = cleaned[(cleaned[COL_TICKER] != '') & (cleaned[COL_TICKER].str.endswith('.NS'))]

    # Filter unusable rows if Data Points column exists
    if COL_DATA_POINTS in cleaned.columns:
        cleaned[COL_DATA_POINTS] = pd.to_numeric(cleaned[COL_DATA_POINTS], errors='coerce').fillna(0)
        cleaned = cleaned[cleaned[COL_DATA_POINTS] > 0]

    # Drop duplicates by ticker, keep first row
    cleaned = cleaned.drop_duplicates(subset=[COL_TICKER], keep='first').reset_index(drop=True)

    removed = before - len(cleaned)
    print(f"Loaded {before} rows from CSV")
    print(f"Retained {len(cleaned)} clean rows, removed {removed} invalid/duplicate rows")

    if len(cleaned) == 0:
        raise ValueError("No valid rows after cleaning CSV. Check 'NSE Ticker' and 'Data Points' columns.")

    return cleaned


def download_bse_stock_data(universe_df: pd.DataFrame, period: str = '2y', interval: str = '1d') -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Download stock data using cleaned NSE tickers from CSV."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Please run: pip install yfinance")
        sys.exit(1)
    
    print(f"Downloading BSE data for {len(universe_df)} companies...")
    all_data = []
    successful_tickers = []
    ticker_to_company = {}

    for i, row in universe_df.iterrows():
        company_name = row[COL_COMPANY]
        ticker = row[COL_TICKER]
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if len(hist) > 100:  # Need at least 100 data points
                hist['Ticker'] = ticker
                all_data.append(hist[['Close', 'Ticker']])
                successful_tickers.append(ticker)
                ticker_to_company[ticker] = company_name
                print(f"  [{i+1}/{len(universe_df)}] {company_name} ({ticker}): {len(hist)} days")
            else:
                print(f"  [{i+1}/{len(universe_df)}] {company_name} ({ticker}): Insufficient data ({len(hist)} days)")
        except Exception as e:
            print(f"  [{i+1}/{len(universe_df)}] {company_name} ({ticker}): Error - {str(e)[:50]}")
    
    if not all_data:
        raise ValueError("No data downloaded for any ticker")
    
    print(f"\nSuccessfully downloaded {len(successful_tickers)} companies")
    return pd.concat(all_data), successful_tickers, ticker_to_company


def build_price_matrix(df: pd.DataFrame, tickers: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Build aligned raw price matrix with columns ordered by available tickers."""
    pivot_df = df.pivot_table(index=df.index, columns='Ticker', values='Close')
    present_tickers = [t for t in tickers if t in pivot_df.columns]
    missing_tickers = [t for t in tickers if t not in pivot_df.columns]
    if missing_tickers:
        print(f"Warning: {len(missing_tickers)} tickers missing in pivot data and will be skipped")
    if len(present_tickers) == 0:
        raise ValueError("No valid tickers available after pivoting downloaded data")
    pivot_df = pivot_df[present_tickers].ffill().bfill()
    return pivot_df.values.astype(np.float32), present_tickers


def create_price_windows(
    price_matrix: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create windows for direct price forecasting mode."""
    total_len = seq_len + pred_len
    num_samples = (len(price_matrix) - total_len) // stride + 1
    if num_samples <= 0:
        raise ValueError(
            f"Insufficient timeline length ({len(price_matrix)}) for seq_len={seq_len}, pred_len={pred_len}, stride={stride}"
        )

    inputs_price = []
    targets_price = []
    anchors = []

    for i in range(num_samples):
        start_idx = i * stride
        end_idx = start_idx + total_len
        if end_idx <= len(price_matrix):
            window = price_matrix[start_idx:end_idx]
            inputs_price.append(window[:seq_len])
            targets_price.append(window[seq_len:])
            anchors.append(window[seq_len - 1])

    return (
        np.array(inputs_price, dtype=np.float32),
        np.array(targets_price, dtype=np.float32),
        np.array(anchors, dtype=np.float32),
    )


def create_log_return_windows(
    price_matrix: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create windows for log-return forecasting mode and keep price targets for fair evaluation."""
    safe_price = np.clip(price_matrix, eps, None)
    returns = np.diff(np.log(safe_price), axis=0).astype(np.float32)

    total_len = seq_len + pred_len
    num_samples = (len(returns) - total_len) // stride + 1
    if num_samples <= 0:
        raise ValueError(
            f"Insufficient return timeline length ({len(returns)}) for seq_len={seq_len}, pred_len={pred_len}, stride={stride}"
        )

    inputs_ret = []
    targets_ret = []
    anchors = []
    targets_price = []

    for i in range(num_samples):
        start_idx = i * stride
        end_idx = start_idx + total_len
        if end_idx <= len(returns):
            ret_window = returns[start_idx:end_idx]
            inputs_ret.append(ret_window[:seq_len])
            targets_ret.append(ret_window[seq_len:])

            anchor_price = safe_price[start_idx + seq_len]
            future_price = safe_price[start_idx + seq_len + 1:start_idx + seq_len + 1 + pred_len]
            anchors.append(anchor_price)
            targets_price.append(future_price)

    return (
        np.array(inputs_ret, dtype=np.float32),
        np.array(targets_ret, dtype=np.float32),
        np.array(anchors, dtype=np.float32),
        np.array(targets_price, dtype=np.float32),
    )


def denormalize_global(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse global z-score normalization for arrays shaped (samples, time, features)."""
    return data * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def normalize_global(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Global z-score normalization for arrays shaped (samples, time, features)."""
    return (data - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def reconstruct_prices_from_log_returns(anchor_prices: np.ndarray, log_returns: np.ndarray) -> np.ndarray:
    """Reconstruct future prices from anchor price and predicted log-returns."""
    cumulative = np.cumsum(log_returns, axis=1)
    return anchor_prices[:, None, :] * np.exp(cumulative)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute common forecasting metrics."""
    mae = float(np.mean(np.abs(predictions - targets)))
    mse = float(np.mean((predictions - targets) ** 2))
    rmse = float(np.sqrt(mse))
    return {'mae': mae, 'mse': mse, 'rmse': rmse}


def compute_per_ticker_metrics(predictions: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """Compute MAE/MSE/RMSE per feature (ticker)."""
    errors = predictions - targets  # (samples, horizon, features)
    mae = np.mean(np.abs(errors), axis=(0, 1))
    mse = np.mean(errors ** 2, axis=(0, 1))
    rmse = np.sqrt(mse)
    return pd.DataFrame({
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
    })


def compute_walk_forward_summary(predictions: np.ndarray, targets: np.ndarray, folds: int) -> Tuple[dict, List[dict]]:
    """Compute fold-wise walk-forward metrics and aggregate mean/std."""
    sample_count = len(predictions)
    fold_count = max(1, min(folds, sample_count))
    split_indices = np.array_split(np.arange(sample_count), fold_count)

    fold_rows = []
    for fold_id, indices in enumerate(split_indices):
        if len(indices) == 0:
            continue
        fold_metrics = compute_metrics(predictions[indices], targets[indices])
        fold_rows.append({
            'fold': fold_id,
            'sample_count': int(len(indices)),
            'mae': fold_metrics['mae'],
            'mse': fold_metrics['mse'],
            'rmse': fold_metrics['rmse'],
        })

    mae_values = np.array([row['mae'] for row in fold_rows], dtype=np.float32)
    mse_values = np.array([row['mse'] for row in fold_rows], dtype=np.float32)
    rmse_values = np.array([row['rmse'] for row in fold_rows], dtype=np.float32)
    summary = {
        'folds': fold_count,
        'mae_mean': float(mae_values.mean()),
        'mae_std': float(mae_values.std(ddof=0)),
        'mse_mean': float(mse_values.mean()),
        'mse_std': float(mse_values.std(ddof=0)),
        'rmse_mean': float(rmse_values.mean()),
        'rmse_std': float(rmse_values.std(ddof=0)),
    }
    return summary, fold_rows


def get_pathformer_predictions(
    inputs: np.ndarray,
    seq_len: int,
    pred_len: int,
    num_features: int,
    use_revin: bool,
    revin_affine: bool,
    revin_subtract_last: bool,
    config_name: str,
    model_state_dict: Optional[dict] = None,
    device: str = 'cpu',
    checkpoint_path: Optional[str] = None,
) -> np.ndarray:
    """Generate predictions using PathFormer model."""
    
    print(f"\nLoading PathFormer model for config: {config_name}")
    
    # Create args for PathFormer
    class Args:
        pass
    
    args = Args()
    args.is_training = 0
    args.model = 'PathFormer'
    args.data = 'custom'
    args.root_path = './'
    args.data_path = 'bse100'
    args.features = 'M'  # Multivariate
    args.target = 'Close'
    args.freq = 'd'
    args.checkpoints = './checkpoints/'
    args.seq_len = seq_len
    args.pred_len = pred_len
    args.individual = False
    args.d_model = 16
    args.d_ff = 64
    args.num_nodes = num_features
    args.layer_nums = 3
    args.k = 2
    args.num_experts_list = [4, 4, 4]
    args.patch_size_list = np.array([16,12,8,32,12,8,6,4,8,6,4,2]).reshape(3, -1).tolist()
    args.revin = 1 if use_revin else 0
    args.revin_affine = revin_affine
    args.revin_subtract_last = revin_subtract_last
    args.drop = 0.1
    args.embed = 'timeF'
    args.residual_connection = 0
    args.metric = 'mae'
    args.batch_norm = 0
    args.num_workers = 0
    args.batch_size = 32
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    args.device_ids = [0]
    
    try:
        exp = Exp_Main(args)
        if model_state_dict is not None:
            exp.model.load_state_dict(model_state_dict, strict=False)
        exp.model.eval()
        
        # Create predictions
        batch_size = inputs.shape[0]
        all_predictions = []
        
        print(f"Generating PathFormer predictions for {batch_size} samples...")
        
        with torch.no_grad():
            for i in range(0, batch_size, args.batch_size):
                batch_inputs = inputs[i:i+args.batch_size]
                batch_tensor = torch.from_numpy(batch_inputs).to(device)
                
                # Reshape for model: (batch, seq_len, features) -> (batch, seq_len, features)
                # PathFormer expects input shape suitable for the model
                
                try:
                    batch_out = exp.model(batch_tensor)
                    batch_pred_tensor = batch_out[0] if isinstance(batch_out, tuple) else batch_out
                    batch_pred = batch_pred_tensor.detach().cpu().numpy()
                    all_predictions.append(batch_pred)
                    print(f"  Processed batch {i//args.batch_size + 1}")
                except Exception as e:
                    print(f"  Warning: Could not get model predictions: {e}")
                    print(f"  Using naive forecast fallback for remaining batches")
                    # Fallback to naive forecast
                    all_predictions.append(naive_forecast_batch(batch_inputs, pred_len))
        
        if all_predictions:
            predictions = np.concatenate(all_predictions, axis=0)
            print(f"Generated predictions shape: {predictions.shape}")
            return predictions
    
    except Exception as e:
        print(f"Warning: Could not load PathFormer model: {e}")
        print(f"Using naive forecast instead...")
    
    return naive_forecast_batch(inputs, pred_len)


def naive_forecast_batch(inputs: np.ndarray, pred_len: int) -> np.ndarray:
    """Fallback deterministic forecast if model unavailable."""
    batch_size, seq_len, num_features = inputs.shape
    predictions = np.zeros((batch_size, pred_len, num_features), dtype=np.float32)
    
    for i in range(batch_size):
        lookback = min(5, seq_len)
        recent = inputs[i, -lookback:]
        
        x = np.arange(lookback)
        for j in range(num_features):
            y = recent[:, j]
            slope = np.polyfit(x, y, 1)[0]
            last_val = inputs[i, -1, j]
            forecast = last_val + slope * np.arange(1, pred_len + 1)
            predictions[i, :, j] = forecast
    
    return predictions


def fine_tune_pathformer(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    model_cfg: dict,
    train_cfg: dict,
    device: str,
) -> Optional[dict]:
    """Fine-tune PathFormer on train split and return model weights."""
    epochs = int(train_cfg['epochs'])
    learning_rate = float(train_cfg['learning_rate'])
    batch_size = int(train_cfg['batch_size'])
    balance_loss_weight = float(train_cfg['balance_loss_weight'])

    if epochs <= 0 or len(train_inputs) == 0:
        return None

    class Args:
        pass

    args = Args()
    args.is_training = 0
    args.model = 'PathFormer'
    args.data = 'custom'
    args.root_path = './'
    args.data_path = 'bse100'
    args.features = 'M'
    args.target = 'Close'
    args.freq = 'd'
    args.checkpoints = './checkpoints/'
    args.seq_len = int(model_cfg['seq_len'])
    args.pred_len = int(model_cfg['pred_len'])
    args.individual = False
    args.d_model = 16
    args.d_ff = 64
    args.num_nodes = int(model_cfg['num_features'])
    args.layer_nums = 3
    args.k = 2
    args.num_experts_list = [4, 4, 4]
    args.patch_size_list = np.array([16,12,8,32,12,8,6,4,8,6,4,2]).reshape(3, -1).tolist()
    args.revin = 1 if model_cfg['use_revin'] else 0
    args.revin_affine = bool(model_cfg['revin_affine'])
    args.revin_subtract_last = bool(model_cfg['revin_subtract_last'])
    args.drop = 0.1
    args.embed = 'timeF'
    args.residual_connection = 0
    args.metric = 'mae'
    args.batch_norm = 0
    args.num_workers = 0
    args.batch_size = batch_size
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    args.device_ids = [0]

    exp = Exp_Main(args)
    model = exp.model.to(device)
    model.train()

    x_tensor = torch.from_numpy(train_inputs).float().to(device)
    y_tensor = torch.from_numpy(train_targets).float().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    total_samples = x_tensor.shape[0]
    print(f"Fine-tuning {model_cfg['config_name']}: samples={total_samples}, epochs={epochs}")

    for epoch in range(epochs):
        perm = torch.randperm(total_samples, device=device)
        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, total_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_tensor[idx]
            yb = y_tensor[idx]

            optimizer.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                pred, aux_loss = out
            else:
                pred, aux_loss = out, torch.tensor(0.0, device=device)

            loss = criterion(pred, yb) + balance_loss_weight * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  {model_cfg['config_name']} epoch {epoch + 1}/{epochs} - loss={epoch_loss / max(1, batch_count):.6f}")

    return copy.deepcopy(model.state_dict())


def evaluate_configuration(
    config: dict,
    datasets: dict,
    eval_cfg: dict,
    device: str,
) -> dict:
    """Run one configuration and evaluate on raw price scale."""
    target_mode = config['target_mode']
    use_revin = config['use_revin']
    folds = int(eval_cfg['folds'])
    train_ratio = float(eval_cfg['train_ratio'])

    price_matrix = datasets['price_matrix']
    price_inputs = datasets['price_inputs']
    price_targets = datasets['price_targets']
    price_anchors = datasets['price_anchors']
    ret_inputs = datasets['ret_inputs']
    ret_targets = datasets['ret_targets']
    ret_anchors = datasets['ret_anchors']
    ret_target_prices = datasets['ret_target_prices']

    if target_mode == 'price':
        raw_inputs = price_inputs
        raw_targets = price_targets
        base_matrix = price_matrix
        anchors = price_anchors
    else:
        raw_inputs = ret_inputs
        raw_targets = ret_targets
        safe_price = np.clip(price_matrix, 1e-8, None)
        base_matrix = np.diff(np.log(safe_price), axis=0).astype(np.float32)
        anchors = ret_anchors

    mean = base_matrix.mean(axis=0).astype(np.float32)
    std = (base_matrix.std(axis=0) + 1e-8).astype(np.float32)

    if use_revin:
        model_inputs = raw_inputs
        model_targets = raw_targets
    else:
        model_inputs = normalize_global(raw_inputs, mean, std)
        model_targets = normalize_global(raw_targets, mean, std)

    sample_count = len(model_inputs)
    split_idx = int(sample_count * train_ratio)
    split_idx = max(1, min(split_idx, sample_count - 1)) if sample_count > 1 else 0

    train_inputs = model_inputs[:split_idx]
    train_targets = model_targets[:split_idx]
    test_inputs = model_inputs[split_idx:]
    test_anchors = anchors[split_idx:]

    if target_mode == 'price':
        test_targets_price = price_targets[split_idx:]
        test_inputs_price_for_plot = price_inputs[split_idx:]
    else:
        test_targets_price = ret_target_prices[split_idx:]
        test_inputs_price_for_plot = price_inputs[split_idx:]

    trained_state = fine_tune_pathformer(
        train_inputs=train_inputs,
        train_targets=train_targets,
        model_cfg={
            'seq_len': eval_cfg['seq_len'],
            'pred_len': eval_cfg['pred_len'],
            'num_features': model_inputs.shape[2],
            'use_revin': use_revin,
            'revin_affine': config['revin_affine'],
            'revin_subtract_last': config['revin_subtract_last'],
            'config_name': config['name'],
        },
        train_cfg={
            'epochs': eval_cfg['finetune_epochs'],
            'learning_rate': eval_cfg['finetune_lr'],
            'batch_size': eval_cfg['finetune_batch_size'],
            'balance_loss_weight': eval_cfg['finetune_balance_loss_weight'],
        },
        device=device,
    )

    predictions_model = get_pathformer_predictions(
        test_inputs,
        eval_cfg['seq_len'],
        eval_cfg['pred_len'],
        model_inputs.shape[2],
        use_revin=use_revin,
        revin_affine=config['revin_affine'],
        revin_subtract_last=config['revin_subtract_last'],
        config_name=config['name'],
        model_state_dict=trained_state,
        device=device,
    )

    if use_revin:
        predictions_target_space = predictions_model
    else:
        predictions_target_space = denormalize_global(predictions_model, mean, std)

    if target_mode == 'price':
        predictions_price = predictions_target_space
        targets_price = test_targets_price
        inputs_price_for_plot = test_inputs_price_for_plot
    else:
        predictions_price = reconstruct_prices_from_log_returns(test_anchors, predictions_target_space)
        targets_price = test_targets_price
        inputs_price_for_plot = test_inputs_price_for_plot

    overall_metrics = compute_metrics(predictions_price, targets_price)
    walk_summary, fold_rows = compute_walk_forward_summary(predictions_price, targets_price, folds=folds)

    return {
        'config': config,
        'predictions_price': predictions_price,
        'targets_price': targets_price,
        'inputs_price_for_plot': inputs_price_for_plot,
        'overall_metrics': overall_metrics,
        'walk_summary': walk_summary,
        'fold_rows': fold_rows,
    }


def main():
    parser = argparse.ArgumentParser(description='BSE100 with PathFormer visualization')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to BSE100 CSV file')
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction horizon')
    parser.add_argument('--period', type=str, default='2y',
                        help='Data period to download')
    parser.add_argument('--stride', type=int, default=24,
                        help='Stride for sliding windows')
    parser.add_argument('--output', type=str, default='./bse100_pathformer_results',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to plot')
    parser.add_argument('--num_features_plot', type=int, default=10,
                        help='Number of features to plot')
    parser.add_argument('--walk_folds', type=int, default=3,
                        help='Number of walk-forward folds for evaluation summary')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train split ratio for fine-tuning before evaluation')
    parser.add_argument('--finetune_epochs', type=int, default=20,
                        help='Fine-tuning epochs per configuration')
    parser.add_argument('--finetune_lr', type=float, default=1e-3,
                        help='Fine-tuning learning rate')
    parser.add_argument('--finetune_batch_size', type=int, default=16,
                        help='Fine-tuning batch size')
    parser.add_argument('--finetune_balance_loss_weight', type=float, default=0.01,
                        help='Weight for PathFormer auxiliary balance loss during fine-tuning')
    
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Read and clean universe from CSV
    universe_df = read_bse100_csv(args.csv_path)
    
    # Download data
    df, successful_tickers, ticker_to_company = download_bse_stock_data(universe_df, period=args.period)

    # Build aligned matrix and both target modes
    price_matrix, aligned_tickers = build_price_matrix(df, successful_tickers)
    price_inputs, price_targets, price_anchors = create_price_windows(
        price_matrix, seq_len=args.seq_len, pred_len=args.pred_len, stride=args.stride
    )
    ret_inputs, ret_targets, ret_anchors, ret_target_prices = create_log_return_windows(
        price_matrix, seq_len=args.seq_len, pred_len=args.pred_len, stride=args.stride
    )

    print("\nPrepared data shapes:")
    print(f"  Price mode inputs/targets: {price_inputs.shape} / {price_targets.shape}")
    print(f"  Log-return mode inputs/targets: {ret_inputs.shape} / {ret_targets.shape}")

    configs = [
        {
            'name': 'baseline_price',
            'target_mode': 'price',
            'use_revin': False,
            'revin_affine': False,
            'revin_subtract_last': False,
        },
        {
            'name': 'revin_price_tuned',
            'target_mode': 'price',
            'use_revin': True,
            'revin_affine': True,
            'revin_subtract_last': True,
        },
        {
            'name': 'baseline_logreturn',
            'target_mode': 'log_return',
            'use_revin': False,
            'revin_affine': False,
            'revin_subtract_last': False,
        },
        {
            'name': 'revin_logreturn_tuned',
            'target_mode': 'log_return',
            'use_revin': True,
            'revin_affine': True,
            'revin_subtract_last': True,
        },
    ]

    config_results = []
    for cfg in configs:
        print(f"\n=== Evaluating {cfg['name']} ===")
        result = evaluate_configuration(
            cfg,
            datasets={
                'price_matrix': price_matrix,
                'price_inputs': price_inputs,
                'price_targets': price_targets,
                'price_anchors': price_anchors,
                'ret_inputs': ret_inputs,
                'ret_targets': ret_targets,
                'ret_anchors': ret_anchors,
                'ret_target_prices': ret_target_prices,
            },
            eval_cfg={
                'seq_len': args.seq_len,
                'pred_len': args.pred_len,
                'folds': args.walk_folds,
                'train_ratio': args.train_ratio,
                'finetune_epochs': args.finetune_epochs,
                'finetune_lr': args.finetune_lr,
                'finetune_batch_size': args.finetune_batch_size,
                'finetune_balance_loss_weight': args.finetune_balance_loss_weight,
            },
            device=device,
        )
        config_results.append(result)
    
    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build config comparison table
    comparison_rows = []
    fold_rows_all = []
    for result in config_results:
        cfg = result['config']
        overall = result['overall_metrics']
        walk = result['walk_summary']
        comparison_rows.append({
            'config_name': cfg['name'],
            'target_mode': cfg['target_mode'],
            'normalization': 'revin' if cfg['use_revin'] else 'simple_global',
            'revin_affine': cfg['revin_affine'],
            'revin_subtract_last': cfg['revin_subtract_last'],
            'overall_mae': overall['mae'],
            'overall_mse': overall['mse'],
            'overall_rmse': overall['rmse'],
            'walk_mae_mean': walk['mae_mean'],
            'walk_mae_std': walk['mae_std'],
            'walk_rmse_mean': walk['rmse_mean'],
            'walk_rmse_std': walk['rmse_std'],
        })

        for fold in result['fold_rows']:
            fold_rows_all.append({
                'config_name': cfg['name'],
                'target_mode': cfg['target_mode'],
                'normalization': 'revin' if cfg['use_revin'] else 'simple_global',
                **fold,
            })

    comparison_df = pd.DataFrame(comparison_rows).sort_values('overall_mae', ascending=True).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows_all)
    comparison_df.to_csv(out_dir / 'walk_forward_config_comparison.csv', index=False)
    fold_df.to_csv(out_dir / 'walk_forward_fold_metrics.csv', index=False)

    with open(out_dir / 'walk_forward_config_comparison.txt', 'w') as f:
        f.write('Config Comparison (sorted by overall MAE, lower is better)\n')
        f.write('========================================================\n\n')
        f.write(comparison_df.to_string(index=False))
        f.write('\n')

    best_idx = comparison_df['overall_mae'].idxmin()
    best_name = comparison_df.loc[best_idx, 'config_name']
    best_result = next(r for r in config_results if r['config']['name'] == best_name)

    revin_candidates = [r for r in config_results if r['config']['use_revin']]
    baseline_candidates = [r for r in config_results if not r['config']['use_revin']]
    best_revin = min(revin_candidates, key=lambda r: r['overall_metrics']['mae'])
    best_baseline = min(baseline_candidates, key=lambda r: r['overall_metrics']['mae'])

    # Save primary arrays as best config outputs
    np.save(out_dir / 'inputs.npy', best_result['inputs_price_for_plot'])
    np.save(out_dir / 'predictions.npy', best_result['predictions_price'])
    np.save(out_dir / 'ground_truth.npy', best_result['targets_price'])

    # Save best RevIN and baseline arrays for careful differentiation
    np.save(out_dir / 'predictions_best_revin.npy', best_revin['predictions_price'])
    np.save(out_dir / 'ground_truth_best_revin.npy', best_revin['targets_price'])
    np.save(out_dir / 'predictions_best_baseline.npy', best_baseline['predictions_price'])
    np.save(out_dir / 'ground_truth_best_baseline.npy', best_baseline['targets_price'])
    
    with open(out_dir / 'companies.txt', 'w') as f:
        for ticker in aligned_tickers:
            f.write(f"{ticker}\t{ticker_to_company.get(ticker, ticker)}\n")

    metrics_revin = best_revin['overall_metrics']
    metrics_baseline = best_baseline['overall_metrics']
    mae_diff = metrics_revin['mae'] - metrics_baseline['mae']
    mse_diff = metrics_revin['mse'] - metrics_baseline['mse']
    rmse_diff = metrics_revin['rmse'] - metrics_baseline['rmse']

    print("\nBest-config summary:")
    print(f"  Winner: {best_name}")
    print(f"  Winner MAE: {best_result['overall_metrics']['mae']:.6f}")
    print(f"  Best RevIN: {best_revin['config']['name']} (MAE={metrics_revin['mae']:.6f})")
    print(f"  Best Baseline: {best_baseline['config']['name']} (MAE={metrics_baseline['mae']:.6f})")
    print(f"  ΔMAE (best RevIN - best Baseline): {mae_diff:.6f}")

    with open(out_dir / 'normalization_comparison.txt', 'w') as f:
        f.write('Normalization Differentiation Report\n')
        f.write('===================================\n\n')
        f.write(f"Best RevIN config: {best_revin['config']['name']}\n")
        f.write(f"Best Baseline config: {best_baseline['config']['name']}\n")
        f.write(f"Overall winner: {best_name}\n\n")
        f.write('Metrics on raw-price scale:\n')
        f.write(f"  Best RevIN MAE: {metrics_revin['mae']:.6f}\n")
        f.write(f"  Best RevIN MSE: {metrics_revin['mse']:.6f}\n")
        f.write(f"  Best RevIN RMSE: {metrics_revin['rmse']:.6f}\n")
        f.write(f"  Best Baseline MAE: {metrics_baseline['mae']:.6f}\n")
        f.write(f"  Best Baseline MSE: {metrics_baseline['mse']:.6f}\n")
        f.write(f"  Best Baseline RMSE: {metrics_baseline['rmse']:.6f}\n")
        f.write(f"  Delta MAE (RevIN - Baseline): {mae_diff:.6f}\n")
        f.write(f"  Delta MSE (RevIN - Baseline): {mse_diff:.6f}\n")
        f.write(f"  Delta RMSE (RevIN - Baseline): {rmse_diff:.6f}\n")

    # Per-ticker differentiation between best RevIN and best baseline
    revin_pred = best_revin['predictions_price']
    revin_tgt = best_revin['targets_price']
    base_pred = best_baseline['predictions_price']
    base_tgt = best_baseline['targets_price']
    common_samples = min(len(revin_pred), len(base_pred))

    per_ticker_revin = compute_per_ticker_metrics(revin_pred[:common_samples], revin_tgt[:common_samples])
    per_ticker_baseline = compute_per_ticker_metrics(base_pred[:common_samples], base_tgt[:common_samples])

    per_ticker_df = pd.DataFrame({
        'ticker': aligned_tickers,
        'company': [ticker_to_company.get(t, t) for t in aligned_tickers],
        'revin_mae': per_ticker_revin['mae'].values,
        'baseline_mae': per_ticker_baseline['mae'].values,
        'delta_mae_revin_minus_baseline': per_ticker_revin['mae'].values - per_ticker_baseline['mae'].values,
        'revin_rmse': per_ticker_revin['rmse'].values,
        'baseline_rmse': per_ticker_baseline['rmse'].values,
        'delta_rmse_revin_minus_baseline': per_ticker_revin['rmse'].values - per_ticker_baseline['rmse'].values,
        'revin_mse': per_ticker_revin['mse'].values,
        'baseline_mse': per_ticker_baseline['mse'].values,
        'delta_mse_revin_minus_baseline': per_ticker_revin['mse'].values - per_ticker_baseline['mse'].values,
    })
    per_ticker_df = per_ticker_df.sort_values('delta_mae_revin_minus_baseline', ascending=True).reset_index(drop=True)
    per_ticker_df.to_csv(out_dir / 'per_ticker_normalization_comparison.csv', index=False)

    with open(out_dir / 'per_ticker_normalization_comparison.txt', 'w') as f:
        f.write('Per-Ticker Normalization Differentiation (best RevIN vs best Baseline)\n')
        f.write('====================================================================\n')
        f.write(f"Common sample count used for fair ticker comparison: {common_samples}\n\n")
        f.write(
            per_ticker_df[
                ['ticker', 'company', 'revin_mae', 'baseline_mae', 'delta_mae_revin_minus_baseline',
                 'revin_rmse', 'baseline_rmse', 'delta_rmse_revin_minus_baseline']
            ].to_string(index=False)
        )
        f.write('\n')

    with open(out_dir / 'best_config_report.txt', 'w') as f:
        f.write('Automatic Best-Config Selection Report\n')
        f.write('=====================================\n\n')
        f.write(f"Selected winner: {best_name}\n")
        f.write(f"Winner target mode: {best_result['config']['target_mode']}\n")
        f.write(f"Winner normalization: {'revin' if best_result['config']['use_revin'] else 'simple_global'}\n")
        f.write(f"Winner overall MAE: {best_result['overall_metrics']['mae']:.6f}\n")
        f.write(f"Winner walk-forward MAE mean+/-std: {best_result['walk_summary']['mae_mean']:.6f} +/- {best_result['walk_summary']['mae_std']:.6f}\n\n")
        f.write(f"Best RevIN config: {best_revin['config']['name']} (MAE={metrics_revin['mae']:.6f})\n")
        f.write(f"Best Baseline config: {best_baseline['config']['name']} (MAE={metrics_baseline['mae']:.6f})\n")
        f.write(f"Delta MAE (best RevIN - best Baseline): {mae_diff:.6f}\n")
        f.write(f"Delta RMSE (best RevIN - best Baseline): {rmse_diff:.6f}\n")
    
    # Visualizations
    print("\nCreating visualizations...")
    
    plot_error_distribution(
        best_result['predictions_price'], best_result['targets_price'],
        save_path=str(out_dir / 'error_distribution.png')
    )
    
    num_plot = min(args.num_samples, len(best_result['inputs_price_for_plot']))
    for i in range(num_plot):
        plot_prediction_comparison(
            best_result['inputs_price_for_plot'], best_result['predictions_price'], best_result['targets_price'],
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            sample_idx=i,
            feature_idx=0,
            save_path=str(out_dir / f'sample_{i}_feature_0.png')
        )
        
        num_features = min(args.num_features_plot, best_result['inputs_price_for_plot'].shape[2])
        plot_multivariate_predictions(
            best_result['inputs_price_for_plot'], best_result['predictions_price'], best_result['targets_price'],
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            sample_idx=i,
            num_features=num_features,
            save_path=str(out_dir / f'sample_{i}_multivariate.png')
        )
    
    print(f"\nResults saved to: {out_dir}")
    print(f"Tickers: {', '.join(aligned_tickers[:10])}...")
    print(f"Per-ticker differentiation: {out_dir / 'per_ticker_normalization_comparison.csv'}")
    print(f"Best-config report: {out_dir / 'best_config_report.txt'}")


if __name__ == '__main__':
    main()
