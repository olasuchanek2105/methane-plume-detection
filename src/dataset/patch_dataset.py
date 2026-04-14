"""
PyTorch Dataset for methane plume patches.

Loads precomputed image patches (X) and labels (Y) from .npy files.
Each sample is converted to float32 and reshaped from (H, W, C) to (C, H, W)
to match PyTorch input format.

Optionally loads metadata from a CSV file.

Used for training and evaluation of models on methane detection patches.
"""

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

class MethanePatchDataset(Dataset):
    def __init__(self, x_path, y_path, meta_path=None):
        self.X = np.load(x_path)
        self.Y = np.load(y_path)
        
        if meta_path is not None:
            self.meta = pd.read_csv(meta_path)
        else:
            self.meta = None

        if len(self.X) != len(self.Y):
            raise ValueError("x i y maja różne dlugości")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y = self.Y[idx]
        x = self.X[idx].astype(np.float32)
        x = np.transpose(x, (2,0,1)) # z (H, W, C) na (C, H, W)
        y = torch.tensor(y, dtype=torch.long)
        
        return x,y
    