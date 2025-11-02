"""
===============================================================================
Title:      Path module
Outline:    Important path variables for the project to avoid hardcoding them
            in each individual file.
Author:     Alejandro Sánchez Cano
Date:       02/10/2024
===============================================================================
"""

# Built-in modules
from pathlib import Path

# Top directories
CHONKY = Path('/home/asanchez/chonky')
PROJECT = CHONKY / 'ML4MIKC'
TOOLS = CHONKY / 'tools'
DATA = CHONKY / 'data'