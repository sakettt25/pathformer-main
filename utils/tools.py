import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def adjust_learning_rate(optimizer,scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 2))}
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 3))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 4))}
    elif args.lradj == 'type5':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend(loc="upper right")
    plt.savefig(name,bbox_inches='tight')
    plt.close()


def visual_predictions(true, preds, input_data, pred_len, name='./pic/prediction.png'):
    """
    Enhanced visualization for predictions showing input, ground truth, and predictions
    """
    plt.figure(figsize=(15, 5))
    
    # Total length for x-axis
    input_len = len(input_data)
    total_len = input_len + pred_len
    
    # Create x-axis
    x_input = np.arange(input_len)
    x_pred = np.arange(input_len, total_len)
    
    # Plot input sequence
    plt.plot(x_input, input_data, label='Input Sequence', color='blue', linewidth=2)
    
    # Plot ground truth
    plt.plot(x_pred, true, label='Ground Truth', color='green', linewidth=2)
    
    # Plot predictions
    plt.plot(x_pred, preds, label='Prediction', color='red', linewidth=2, linestyle='--')
    
    # Add vertical line to separate input and prediction
    plt.axvline(x=input_len, color='black', linestyle=':', linewidth=1.5, label='Forecast Start')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('Time Series Forecasting: Input → Prediction vs Ground Truth')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches='tight')
    plt.close()


def visual_multivariate(true, preds, input_data, pred_len, feature_names=None, 
                        num_features=5, name='./pic/multivariate_prediction.png'):
    """
    Visualization for multiple features in multivariate time series
    """
    num_features = min(num_features, true.shape[1])
    
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 3*num_features))
    if num_features == 1:
        axes = [axes]
    
    input_len = input_data.shape[0]
    total_len = input_len + pred_len
    
    x_input = np.arange(input_len)
    x_pred = np.arange(input_len, total_len)
    
    for i in range(num_features):
        ax = axes[i]
        
        # Plot input
        ax.plot(x_input, input_data[:, i], label='Input', color='blue', linewidth=1.5)
        
        # Plot ground truth
        ax.plot(x_pred, true[:, i], label='Ground Truth', color='green', linewidth=1.5)
        
        # Plot predictions
        ax.plot(x_pred, preds[:, i], label='Prediction', color='red', linewidth=1.5, linestyle='--')
        
        # Add vertical line
        ax.axvline(x=input_len, color='black', linestyle=':', linewidth=1)
        
        feature_label = f'Feature {i+1}' if feature_names is None else feature_names[i]
        ax.set_ylabel(feature_label)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.set_title('Multivariate Time Series Forecasting')
        if i == num_features - 1:
            ax.set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches='tight')
    plt.close()

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


