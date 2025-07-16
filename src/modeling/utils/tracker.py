"""
===============================================================================
Title:      Tracker
Outline:    Tracker class to track and plot metrics during training and 
            evaluation. This allows for easy monitoring of model performance
            during the training process.
Author:     Alejandro Sánchez Cano
Date:       15/07/2025
===============================================================================
"""

# Third-party modules
import pandas as pd
import matplotlib.pyplot as plt

# Custom modules
from src.misc.logger import logger

#TODO: pick up from checkpoint
#TODO: improve plotting aesthetics
#NOTE: this code my be substituted by wandb in the future

class Tracker:

    def __init__(self, output_dir: str = '.'):
        self.output_dir = output_dir
        self.metrics = pd.DataFrame()
        self.model = None

    def track(self, **kwargs) -> None:
        '''Record metrics'''
        df = pd.DataFrame([kwargs])
        self.metrics = pd.concat([self.metrics, df], ignore_index=True)
        if self.metrics['val_loss'].min() == kwargs.get('val_loss', float('inf')):
            self.model = kwargs.get('model', None)

    def best_epoch(self) -> tuple[int, pd.Series]:
        '''
        Find best epoch based on validation loss and report the metrics

        Returns
        -------
            best_epoch : int
                The epoch with the best validation loss.
            best_metrics : pd.Series
                The metrics of the best epoch.
        '''
        if 'val_loss' not in self.metrics.columns:
            raise ValueError("Validation loss not found in metrics.")
        best_epoch = self.metrics['val_loss'].idxmin()
        best_metrics = self.metrics.iloc[best_epoch]
        logger.info(f"Best epoch: {best_epoch + 1}/{len(self.metrics)}")
        logger.info("Metrics:")
        for key, value in best_metrics.items():
            logger.info(f"{key}: {value}")

        return best_epoch, best_metrics
    
    def plot(self) -> None:
        '''Plot metrics based on the fields tracked'''
        if any(col in self.metrics.columns for col in ['learning_rate', 'lr',]):
            self.__plot_learning_rate()
        if any(col in self.metrics.columns for col in ['train_loss', 'val_loss']):
            self.__plot_losses()
        if any(col in self.metrics.columns for col in ['train_bal_accuracy', 'val_bal_accuracy', 'train_f1', 'val_f1', 'train_mcc', 'val_mcc']):
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
