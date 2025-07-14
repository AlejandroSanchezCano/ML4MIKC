"""
===============================================================================
Title:      Tracker
Outline:    Tracker class to track and plot metrics during training and 
            evaluation.
Author:     Alejandro Sánchez Cano
Date:       2024-10-11
Version:    2025-03-03
License:    MIT
===============================================================================
"""

# Third-party modules
import matplotlib.pyplot as plt

class Tracker():

    def __init__(self):
        # Initialize output directory
        self.output_dir = None
        # Initialize metrics
        self.metrics = {
            'learning_rate': [],
            'train_loss': [],
            'val_loss': [],
            'train_bal_accuracy': [],
            'val_bal_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'train_mcc': [],
            'val_mcc': []
        }

    def track(self, **kwargs) -> None:
        '''Record metrics'''
        # Update metrics
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def plot(self, *metrics: str, output_dir: str) -> None:
        '''
        Decide which metrics to plot and save the plots.

        Parameters
        ----------
        metrics : str
            Metrics to plot
        output_dir : str
            Plotting output directory
        '''
        # Set output directory
        self.output_dir = output_dir

        # Plot metrics
        if 'learning_rate' in metrics:
            self.__plot_learning_rate()
        if 'losses' in metrics:
            self.__plot_losses()
        if 'performance' in metrics:
            self.__plot_performance()
    
    def __plot_learning_rate(self) -> None:
        '''Plot learning rate'''	
        for lr_parameter_group in zip(*self.metrics['learning_rate']):
            plt.plot(lr_parameter_group)
        plt.xlabel('Epoch')
        plt.ylabel('Learning rate')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/learning_rate.png')
        plt.clf()

    def __plot_losses(self) -> None:
        '''Plot losses'''
        plt.plot(self.metrics['train_loss'], label='Train loss')
        plt.plot(self.metrics['val_loss'], label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/losses.png')
        plt.clf()
    
    def __plot_performance(self) -> None:
        '''Plot performance metrics'''
        plt.plot(self.metrics['train_bal_accuracy'], label='Train balanced accuracy', linestyle='--', color = 'red')
        plt.plot(self.metrics['val_bal_accuracy'], label='Validation balanced accuracy', color = 'red')
        plt.plot(self.metrics['train_f1'], label='Train F1 score', linestyle='--', color = 'blue')
        plt.plot(self.metrics['val_f1'], label='Validation F1 score', color = 'blue')
        plt.plot(self.metrics['train_mcc'], label='Train MCC', linestyle='--', color = 'green')
        plt.plot(self.metrics['val_mcc'], label='Validation MCC', color = 'green')
        plt.xlabel('Epoch')
        plt.ylabel('Performance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance.png')
        plt.clf()
