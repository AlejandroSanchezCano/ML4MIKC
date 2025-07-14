"""
===============================================================================
Title:      ModelHub
Outline:    Collection of deep learning models.
Author:     Alejandro Sánchez Cano
Date:       2024-10-01
Version:    2025-03-12
License:    MIT
===============================================================================
"""

# Third-party modules
import torch
import torch.nn as nn

# Custom modules
from src.misc.logger import logger

class CCL(nn.Module):
    def __init__(self, max_shape: int):
        # Internal
        super(CCL, self).__init__()
        # External
        self.max_shape = max_shape
        # Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * int(self.max_shape/2/2)**2, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):

        logger.debug(f'Input shape: {x.shape}')
        x = self.pool1(self.relu1(self.conv1(x)))           # Convolution 1
        logger.debug(f'Conv1 shape: {x.shape}')
        x = self.pool2(self.relu2(self.conv2(x)))           # Convolution 2
        logger.debug(f'Conv2 shape: {x.shape}')
        x = x.view(-1, 64 * int(self.max_shape/2/2)**2)     # Flatten
        logger.debug(f'Flatten shape: {x.shape}')
        x = self.fc1(x)                                     # Fully connected 1
        logger.debug(f'Output shape: {x.shape}')

        return x