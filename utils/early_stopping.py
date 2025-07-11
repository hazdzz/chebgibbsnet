import math
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, delta: float = 0.0, patience: int = 7, verbose: bool = True, path: str = 'checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = math.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        """
        Call method to check if early stopping should be triggered.
        
        Args:
            val_loss (float): Current validation loss
            model: The model to potentially save
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, first_save=True)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, first_save=False):
        """
        Save model checkpoint when validation loss improves.
        
        Args:
            val_loss (float): Current validation loss
            model: The model to save
            first_save (bool): Whether this is the first save
        """
        if self.verbose:
            if first_save:
                print(f'Initial model saved with validation loss: {val_loss:.6f}')
            else:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss