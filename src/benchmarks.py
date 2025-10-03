from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ParityDataset(Dataset):
    """
    Generate random binary sequences of length seq_len.
    Label is parity (sum mod 2).
    Input features are float32 in {0.0, 1.0}.
    """

    def __init__(self, num_samples: int, seq_len: int = 32, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.x = rng.randint(0, 2, size=(num_samples, seq_len)).astype(np.float32)
        self.y = (self.x.sum(axis=1) % 2).astype(np.int64)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.x[idx])
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return features, label 