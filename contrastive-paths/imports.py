"""
Common imports for the contrastive-paths module.
This module consolidates all necessary imports for the codebase.
"""

# Standard library imports
import itertools
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Scikit-learn imports
from sklearn.metrics import roc_auc_score, roc_curve

# Mogra module imports (from parent directory)
import sys
import os

# Add parent directory to path to access mogra
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import from mogra
from mogra.datatypes import Shruti, SSwar, SAPTAK_MARKS
from mogra.tonnetz import EFGenus, Tonnetz

# Optional visualization imports (may not be available in all environments)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

# Export commonly used items
__all__ = [
    'itertools', 'np', 'plt', 'Fraction',
    'torch', 'nn', 'F', 'DataLoader', 'Dataset',
    'roc_auc_score', 'roc_curve',
    'Shruti', 'SSwar', 'SAPTAK_MARKS',
    'EFGenus', 'Tonnetz',
    'UMAP_AVAILABLE'
]

if UMAP_AVAILABLE:
    __all__.append('umap')