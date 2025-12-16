"""
BSE 100 Stock Data Visualization using CSV mapping to NSE tickers.

Usage:
  python scripts/bse100_visualization.py --csv_path ./stock-data/bse100\ -\ Sheet1.csv --output ./bse100_results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from visualize_results import (
    plot_prediction_comparison,
    plot_multivariate_predictions,
    plot_error_distribution,
)

# Mapping of BSE company names to NSE tickers
COMPANY_TO_TICKER = {
    'Reliance Industries Ltd.': 'RELIANCE.NS',
    'Tata Consultancy Services (TCS)': 'TCS.NS',
    'HDFC Bank Ltd.': 'HDFCBANK.NS',
    'ICICI Bank Ltd.': 'ICICIBANK.NS',
    'Hindustan Unilever Ltd. (HUL)': 'HINDUNILVR.NS',
    'Infosys Ltd.': 'INFY.NS',
    'Life Insurance Corporation (LIC)': 'LIC.NS',
    'State Bank of India (SBI)': 'SBIN.NS',
    'Bharti Airtel Ltd.': 'BHARTIARTL.NS',
    'Bajaj Finance Ltd.': 'BAJAJFINSV.NS',
    'Larsen & Toubro (L&T)': 'LT.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Adani Enterprises Ltd.': 'ADANIENTERP.NS',
    'Asian Paints Ltd.': 'ASIANPAINT.NS',
    'HCL Technologies Ltd.': 'HCLTECH.NS',
    'ITC Ltd.': 'ITC.NS',
    'Power Grid Corporation': 'POWERGRID.NS',
    'Wipro Ltd.': 'WIPRO.NS',
    'Axis Bank Ltd.': 'AXISBANK.NS',
    'Sun Pharmaceutical Industries': 'SUNPHARMA.NS',
}


def read_bse_csv(csv_path: str) -> Dict[str, str]:
    """Read BSE CSV and extract company names."""
    df = pd.read_csv(csv_path)
    company_tickers = {}
    
    print(f"Reading BSE data from: {csv_path}")
    print(f"Total companies in CSV: {len(df)}")
    
    for _, row in df.iterrows():
        company_name = row['Company Name'].strip()
        if company_name in COMPANY_TO_TICKER:
            ticker = COMPANY_TO_TICKER[company_name]
            company_tickers[company_name] = ticker
        else:
            print(f"  WARNING: No ticker mapping for '{company_name}'")
    
    print(f"Successfully mapped {len(company_tickers)} companies to tickers")
    return company_tickers


def download_stock_data(
    company_tickers: Dict[str, str], 
    period: str = '2y', 
    interval: str = '1d'
) -> Tuple[pd.DataFrame, List[str]]:
    """Download stock data for BSE companies using NSE tickers."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Please run: pip install yfinance")
        sys.exit(1)
    
    tickers = list(company_tickers.values())
    print(f"\nDownloading data for {len(tickers)} companies...")
    all_data = []
    successful_tickers = []
    
    for i, (company_name, ticker) in enumerate(company_tickers.items()):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if len(hist) > 0:
                hist['Ticker'] = ticker
                hist['Company'] = company_name
                all_data.append(hist[['Close', 'Ticker', 'Company']])
                successful_tickers.append(ticker)
                print(f"  [{i+1}/{len(company_tickers)}] {company_name:40} ({ticker:15}): {len(hist):4} days")
            else:
                print(f"  [{i+1}/{len(company_tickers)}] {company_name:40} ({ticker:15}): No data")
        except Exception as e:
            print(f"  [{i+1}/{len(company_tickers)}] {company_name:40} ({ticker:15}): Error - {str(e)[:30]}")
    
    if not all_data:
        raise ValueError("No data downloaded for any company")
    
    print(f"\n✓ Successfully downloaded {len(successful_tickers)} companies")
    return pd.concat(all_data), successful_tickers


def prepare_timeseries_data(
    df: pd.DataFrame,
    tickers: List[str],
    seq_len: int = 60,
    pred_len: int = 30,
    stride: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for time series forecasting."""
    
    # Pivot to get companies as columns
    pivot_df = df.pivot_table(index=df.index, columns='Ticker', values='Close')
    pivot_df = pivot_df[tickers]  # Ensure correct order
    pivot_df = pivot_df.ffill().bfill()  # Fill missing values
    
    # Normalize each series
    data_normalized = (pivot_df - pivot_df.mean()) / pivot_df.std()
    data_array = data_normalized.values
    
    # Create sliding windows
    total_len = seq_len + pred_len
    num_samples = max(1, (len(data_array) - total_len) // stride + 1)
    
    inputs = []
    targets = []
    
    for i in range(num_samples):
        start_idx = i * stride
        end_idx = start_idx + total_len
        
        if end_idx <= len(data_array):
            window = data_array[start_idx:end_idx]
            inputs.append(window[:seq_len])
            targets.append(window[seq_len:])
    
    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    print(f"\nPrepared data shapes:")
    print(f"  Inputs: {inputs.shape} (samples, seq_len, companies)")
    print(f"  Targets: {targets.shape} (samples, pred_len, companies)")
    
    return inputs, targets


def naive_forecast(inputs: np.ndarray, pred_len: int) -> np.ndarray:
    """Create naive forecast using last value + trend."""
    batch_size, seq_len, num_features = inputs.shape
    predictions = np.zeros((batch_size, pred_len, num_features), dtype=np.float32)
    
    for i in range(batch_size):
        # Use last 5 points to estimate trend
        lookback = min(5, seq_len)
        recent = inputs[i, -lookback:]
        
        # Linear regression for trend
        x = np.arange(lookback)
        for j in range(num_features):
            y = recent[:, j]
            slope = np.polyfit(x, y, 1)[0]
            
            # Forecast with trend + small noise
            last_val = inputs[i, -1, j]
            forecast = last_val + slope * np.arange(1, pred_len + 1)
            noise = np.random.normal(0, 0.1, pred_len)
            predictions[i, :, j] = forecast + noise
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='BSE 100 Stock visualization')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to BSE CSV file')
    parser.add_argument('--seq_len', type=int, default=60, 
                        help='Input sequence length (days)')
    parser.add_argument('--pred_len', type=int, default=30, 
                        help='Prediction horizon (days)')
    parser.add_argument('--period', type=str, default='2y', 
                        help='Data period (1y, 2y, 5y, max)')
    parser.add_argument('--stride', type=int, default=10, 
                        help='Stride for sliding window')
    parser.add_argument('--output', type=str, default='./bse100_results', 
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=3, 
                        help='Number of samples to plot')
    parser.add_argument('--num_features_plot', type=int, default=10, 
                        help='Number of stocks to plot in multivariate')
    
    args = parser.parse_args()
    
    np.random.seed(42)
    
    # Read CSV and get company-to-ticker mapping
    company_tickers = read_bse_csv(args.csv_path)
    
    if not company_tickers:
        print("ERROR: No companies mapped from CSV")
        sys.exit(1)
    
    # Download data
    df, successful_tickers = download_stock_data(
        company_tickers,
        period=args.period,
        interval='1d'
    )
    
    # Prepare time series data
    inputs, targets = prepare_timeseries_data(
        df, 
        successful_tickers,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride
    )
    
    # Generate naive predictions
    print("\nGenerating naive forecasts...")
    predictions = naive_forecast(inputs, args.pred_len)
    
    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(out_dir / 'inputs.npy', inputs)
    np.save(out_dir / 'predictions.npy', predictions)
    np.save(out_dir / 'ground_truth.npy', targets)
    
    # Save ticker info
    with open(out_dir / 'tickers.txt', 'w') as f:
        f.write('\n'.join(successful_tickers))
    
    print(f"\n✓ Saved arrays to {out_dir}")
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\nOverall Metrics:")
    print(f"  MAE:  {mae:.6f}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Error distribution
    plot_error_distribution(
        predictions, targets,
        save_path=str(out_dir / 'error_distribution.png')
    )
    
    # Individual samples
    num_plot = min(args.num_samples, len(inputs))
    for i in range(num_plot):
        # Single stock (first one)
        plot_prediction_comparison(
            inputs, predictions, targets,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            sample_idx=i,
            feature_idx=0,
            save_path=str(out_dir / f'sample_{i}_single_stock.png')
        )
        
        # Multivariate (multiple stocks)
        num_features_to_plot = min(args.num_features_plot, len(successful_tickers))
        plot_multivariate_predictions(
            inputs, predictions, targets,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            sample_idx=i,
            num_features=num_features_to_plot,
            save_path=str(out_dir / f'sample_{i}_multiple_stocks.png')
        )
    
    print(f"\n✓ All visualizations saved to: {out_dir}")
    print(f"\nCompanies included ({len(successful_tickers)}):")
    for ticker in successful_tickers:
        print(f"  • {ticker}")


if __name__ == '__main__':
    main()
