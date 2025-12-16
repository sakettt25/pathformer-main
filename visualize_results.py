"""
Standalone script to visualize saved predictions
Usage: python visualize_results.py --result_path ./test_results/your_experiment_folder/
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_prediction_comparison(inputs, predictions, ground_truth, seq_len, pred_len, 
                                sample_idx=0, feature_idx=-1, save_path=None):
    """
    Plot input sequence, predictions, and ground truth for a single sample
    """
    plt.figure(figsize=(15, 5))
    
    # Create x-axis
    x_input = np.arange(seq_len)
    x_pred = np.arange(seq_len, seq_len + pred_len)
    
    # Plot input sequence
    plt.plot(x_input, inputs[sample_idx, :, feature_idx], 
             label='Input Sequence', color='blue', linewidth=2)
    
    # Plot ground truth
    plt.plot(x_pred, ground_truth[sample_idx, :, feature_idx], 
             label='Ground Truth', color='green', linewidth=2)
    
    # Plot predictions
    plt.plot(x_pred, predictions[sample_idx, :, feature_idx], 
             label='Prediction', color='red', linewidth=2, linestyle='--')
    
    # Add vertical line
    plt.axvline(x=seq_len, color='black', linestyle=':', linewidth=1.5, 
                label='Forecast Start')
    
    # Calculate error metrics for this sample
    mae = np.mean(np.abs(predictions[sample_idx, :, feature_idx] - 
                          ground_truth[sample_idx, :, feature_idx]))
    mse = np.mean((predictions[sample_idx, :, feature_idx] - 
                   ground_truth[sample_idx, :, feature_idx])**2)
    rmse = np.sqrt(mse)
    
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Sample {sample_idx} - Feature {feature_idx}\n' + 
              f'MAE: {mae:.4f}, RMSE: {rmse:.4f}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    else:
        plt.show()
    plt.close()


def plot_multivariate_predictions(inputs, predictions, ground_truth, seq_len, pred_len,
                                   sample_idx=0, num_features=5, save_path=None):
    """
    Plot multiple features for a single sample
    """
    num_features = min(num_features, predictions.shape[2])
    
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 3*num_features))
    if num_features == 1:
        axes = [axes]
    
    x_input = np.arange(seq_len)
    x_pred = np.arange(seq_len, seq_len + pred_len)
    
    for i in range(num_features):
        ax = axes[i]
        
        # Plot input
        ax.plot(x_input, inputs[sample_idx, :, i], 
                label='Input', color='blue', linewidth=1.5)
        
        # Plot ground truth
        ax.plot(x_pred, ground_truth[sample_idx, :, i], 
                label='Ground Truth', color='green', linewidth=1.5)
        
        # Plot predictions
        ax.plot(x_pred, predictions[sample_idx, :, i], 
                label='Prediction', color='red', linewidth=1.5, linestyle='--')
        
        # Add vertical line
        ax.axvline(x=seq_len, color='black', linestyle=':', linewidth=1)
        
        # Calculate error
        mae = np.mean(np.abs(predictions[sample_idx, :, i] - 
                              ground_truth[sample_idx, :, i]))
        
        ax.set_ylabel(f'Feature {i} (MAE: {mae:.3f})', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.set_title(f'Multivariate Forecasting - Sample {sample_idx}', 
                        fontsize=14, fontweight='bold')
        if i == num_features - 1:
            ax.set_xlabel('Time Steps', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    else:
        plt.show()
    plt.close()


def plot_error_distribution(predictions, ground_truth, save_path=None):
    """
    Plot error distribution across all predictions
    """
    errors = predictions - ground_truth
    mae_per_sample = np.mean(np.abs(errors), axis=(1, 2))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of errors
    axes[0].hist(errors.flatten(), bins=100, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Prediction Errors', fontsize=14)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].grid(True, alpha=0.3)
    
    # MAE per sample
    axes[1].plot(mae_per_sample, linewidth=1.5)
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('MAE per Test Sample', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize PathFormer predictions')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to test results folder')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--num_features', type=int, default=5,
                        help='Number of features to visualize for multivariate')
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction sequence length')
    
    args = parser.parse_args()
    
    # Load data
    print(f'Loading data from {args.result_path}')
    inputs = np.load(os.path.join(args.result_path, 'inputs.npy'))
    predictions = np.load(os.path.join(args.result_path, 'predictions.npy'))
    ground_truth = np.load(os.path.join(args.result_path, 'ground_truth.npy'))
    
    print(f'Data shapes:')
    print(f'  Inputs: {inputs.shape}')
    print(f'  Predictions: {predictions.shape}')
    print(f'  Ground Truth: {ground_truth.shape}')
    
    # Create output folder
    vis_folder = os.path.join(args.result_path, 'visualizations')
    os.makedirs(vis_folder, exist_ok=True)
    
    # Overall statistics
    mae = np.mean(np.abs(predictions - ground_truth))
    mse = np.mean((predictions - ground_truth)**2)
    rmse = np.sqrt(mse)
    
    print(f'\nOverall Metrics:')
    print(f'  MAE: {mae:.6f}')
    print(f'  MSE: {mse:.6f}')
    print(f'  RMSE: {rmse:.6f}')
    
    # Plot error distribution
    print('\nCreating error distribution plot...')
    plot_error_distribution(predictions, ground_truth, 
                           save_path=os.path.join(vis_folder, 'error_distribution.png'))
    
    # Plot individual samples
    print(f'\nCreating plots for {args.num_samples} samples...')
    for i in range(min(args.num_samples, len(predictions))):
        # Single feature plot (last feature)
        plot_prediction_comparison(
            inputs, predictions, ground_truth,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            sample_idx=i,
            feature_idx=-1,
            save_path=os.path.join(vis_folder, f'sample_{i}_single.png')
        )
        
        # Multivariate plot
        if predictions.shape[2] > 1:
            plot_multivariate_predictions(
                inputs, predictions, ground_truth,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                sample_idx=i,
                num_features=args.num_features,
                save_path=os.path.join(vis_folder, f'sample_{i}_multivariate.png')
            )
    
    print(f'\nAll visualizations saved to: {vis_folder}')


if __name__ == '__main__':
    main()
