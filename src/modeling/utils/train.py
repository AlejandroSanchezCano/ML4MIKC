"""
===============================================================================
Title:      Train
Outline:    Train class for deep learning models.
Author:     Alejandro Sánchez Cano
Date:       2024-10-01
Version:    2025-03-03
License:    MIT
===============================================================================
"""

# Third-party modules
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Custom modules
from src.misc.logger import logger

def train(
        model: nn.Module, 
        train_loader: DataLoader,
        criterion: nn. Module,
        optimizer: optim.Optimizer,
        device: torch.device
        ) -> None:
    '''
    Train deep learning model.

    Parameters
    ----------
    model : nn.Module
        Deep learning model
    train_loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimization algorithm
    device : torch.device
        Device to run the model (CPU or GPU)
    '''
    
    model.train()                           # Train mode
    running_loss = 0.0                      # Accumulate loss

    for _, labels, *inputs in train_loader:
        inputs = [input.to(device) for input in inputs] # Send to device
        labels = labels.to(device)          # Send to device
        labels = labels.float()             # Convert to float
        optimizer.zero_grad()               # Zero gradients
        outputs = model(*inputs)             # Forward pass
        outputs = outputs.squeeze(1)        # Squeeze output
        logger.debug(f'Outputs shape: {outputs.shape}')
        logger.debug(f'Labels type: {labels.dtype}')
        logger.debug(f'Outputs type: {outputs.dtype}')
        loss = criterion(outputs, labels)   # Compute loss
        loss.backward()                     # Backward pass
        optimizer.step()                    # Update weights
        running_loss += loss.item()         # Accumulate loss

def evaluate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device    
        ) -> dict[str, float | np.ndarray]:
    '''
    Evaluate deep learning model.

    Parameters
    ----------
    model : nn.Module
        Deep learning model
    loader : DataLoader
        Data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to run the model (CPU or GPU)

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        Dictionary with loss, labels, label indeces, and logits.
    '''
    
    model.eval()                            # Evaluation mode
    running_loss = 0.0                      # Accumulate loss
    total_labels = np.array([])             # Accumulate labels
    total_idx = np.array([])                # Accumulate indices
    total_logits = np.array([])             # Accumulate logits

    with torch.no_grad():
        for indeces, labels, *inputs in loader:
            inputs = [input.to(device) for input in inputs] # Send to device
            labels = labels.to(device)                  # Send to device
            labels = labels.float()                     # Convert to float
            logits = model(*inputs)                      # Forward pass
            logits = logits.squeeze(1)                  # Squeeze logits
            loss = criterion(logits, labels)            # Compute loss
            running_loss += loss.item()                 # Accumulate loss

            total_labels = np.concatenate((total_labels, labels.cpu().numpy()))
            total_idx = np.concatenate((total_idx, indeces.cpu().numpy()))
            total_logits = np.concatenate((total_logits, logits.cpu().numpy()))

            logger.debug(f'Labels: {labels}')
            logger.debug(f'Logits: {logits}')
            logger.debug(f'Labels shape: {labels.shape}')
            logger.debug(f'Logits shape: {logits.shape}')

    return {
        'loss': running_loss / len(loader),
        'labels': total_labels,
        'indeces': total_idx,
        'logits': total_logits
    }